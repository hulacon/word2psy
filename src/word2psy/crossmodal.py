"""Cross-modal similarity between word2psy text and viz2psy image embeddings.

word2psy's ``clip_text`` model and viz2psy's ``clip`` model use the same
OpenCLIP checkpoint (ViT-B-32, ``laion2b_s34b_b79k``), so their L2-normalized
512-d embeddings live in one shared space: the cosine similarity between a
word (or passage) and an image is meaningful. This module joins the two
tools' output CSVs — ``clip_text_000..511`` columns from a word2psy chunks
file and ``clip_000..511`` columns from a viz2psy image file — into a
text x image similarity matrix.

Note that raw CLIP text-image cosine similarities are compressed into a
narrow band (matches typically land around 0.2-0.3, non-matches around
0.1-0.2); relative comparisons within a stimulus set are what carry signal.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

_TEXT_COL = re.compile(r"^clip_text_(\d{3})$")
_IMAGE_COL = re.compile(r"^clip_(\d{3})$")

# Identifier candidates for image rows in a viz2psy CSV, in preference order
_IMAGE_ID_COLS = ["filename", "filepath", "image_idx", "time"]


def clip_columns(df: pd.DataFrame, kind: str) -> list[str]:
    """Ordered CLIP embedding columns of the given kind ("text" or "image").

    The image pattern (``clip_###``) does not match text columns
    (``clip_text_###``), so a combined CSV is handled correctly.
    """
    pattern = _TEXT_COL if kind == "text" else _IMAGE_COL
    cols = [c for c in df.columns if pattern.match(c)]
    return sorted(cols, key=lambda c: int(pattern.match(c).group(1)))


def _embedding_matrix(df: pd.DataFrame, kind: str) -> np.ndarray:
    """Extract and re-L2-normalize the embedding matrix."""
    cols = clip_columns(df, kind)
    if not cols:
        which = "clip_text_*" if kind == "text" else "clip_*"
        raise ValueError(f"No {which} embedding columns found in the CSV.")
    X = df[cols].to_numpy(dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def _text_labels(df: pd.DataFrame) -> list[str]:
    if "chunk_label" in df.columns:
        return df["chunk_label"].astype(str).tolist()
    if "chunk_idx" in df.columns:
        return [f"chunk_{i}" for i in df["chunk_idx"]]
    return [f"row_{i}" for i in range(len(df))]


def _image_labels(df: pd.DataFrame) -> list[str]:
    for col in _IMAGE_ID_COLS:
        if col in df.columns:
            return df[col].astype(str).tolist()
    return [f"image_{i}" for i in range(len(df))]


def cross_modal_similarity(
    text_df: pd.DataFrame,
    image_df: pd.DataFrame,
) -> pd.DataFrame:
    """Cosine similarity between every text chunk and every image.

    Parameters
    ----------
    text_df : pd.DataFrame
        word2psy chunks output containing ``clip_text_000..511`` columns.
    image_df : pd.DataFrame
        viz2psy output containing ``clip_000..511`` columns.

    Returns
    -------
    pd.DataFrame
        Similarity matrix: one row per text chunk (indexed by chunk label),
        one column per image (named by filename where available).
    """
    T = _embedding_matrix(text_df, "text")
    V = _embedding_matrix(image_df, "image")
    if T.shape[1] != V.shape[1]:
        raise ValueError(
            f"Embedding dimensions differ: text {T.shape[1]} vs image "
            f"{V.shape[1]}. Were both produced with the shared "
            "ViT-B-32 checkpoint?"
        )

    sim = T @ V.T
    return pd.DataFrame(
        sim, index=_text_labels(text_df), columns=_image_labels(image_df)
    ).rename_axis(index="text", columns="image")


def top_matches(sim: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Long-format top-k images per text row, ranked by similarity."""
    records = []
    for text_label, row in sim.iterrows():
        ranked = row.sort_values(ascending=False).head(k)
        for rank, (image_label, value) in enumerate(ranked.items(), start=1):
            records.append(
                {
                    "text": text_label,
                    "rank": rank,
                    "image": image_label,
                    "similarity": round(float(value), 4),
                }
            )
    return pd.DataFrame.from_records(records)
