"""Batch inference pipeline: tokenize text, run models, return scores."""

import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from word2psy.exceptions import InferenceError, ModelLoadError
from word2psy.models.base import BaseModel
from word2psy.tokenize import tokenize_text


def score_text(
    text: str | list[str],
    models: list[BaseModel],
    *,
    chunk_labels: list[str] | None = None,
    keep_punctuation: bool = False,
    batch_size: int = 64,
    quiet: bool = False,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Score text with one or more feature models.

    Parameters
    ----------
    text : str or list[str]
        A single string (one chunk) or list of strings (multiple chunks).
    models : list of BaseModel
        Instantiated (but not yet loaded) model wrappers.
    chunk_labels : list[str], optional
        Labels for each chunk.
    keep_punctuation : bool
        If True, keep punctuation tokens in the word table.
    batch_size : int
        Number of items per forward pass.
    quiet : bool
        If True, suppress progress output.

    Returns
    -------
    df : pd.DataFrame
        Word-per-row table with index columns and all scalar features.
    embeddings : dict[str, np.ndarray]
        Mapping of model name to embedding arrays. Chunk-level embeddings
        have shape ``(n_chunks, dim)``; not included in the CSV.
    """
    # Build the word-per-row table
    df = tokenize_text(text, chunk_labels=chunk_labels, keep_punctuation=keep_punctuation)
    chunks = [text] if isinstance(text, str) else list(text)

    embeddings: dict[str, np.ndarray] = {}

    for model in models:
        if not quiet:
            print(f"Loading {model.name} on {model.device}...")
        try:
            model.load()
        except Exception as e:
            raise ModelLoadError(model.name, str(e)) from e

        start_time = time.time()

        if model.level == "word":
            _score_word_level(df, model, batch_size=batch_size, quiet=quiet)
        elif model.level == "chunk":
            chunk_emb = _score_chunk_level(
                df, chunks, model, batch_size=batch_size, quiet=quiet
            )
            if chunk_emb is not None:
                embeddings[model.name] = chunk_emb
        else:
            raise InferenceError(
                model.name, f"Unknown model level: {model.level!r}"
            )

        elapsed = time.time() - start_time
        if not quiet:
            print(f"  {model.name} completed in {elapsed:.1f}s")

    return df, embeddings


def _score_word_level(
    df: pd.DataFrame,
    model: BaseModel,
    *,
    batch_size: int = 64,
    quiet: bool = False,
) -> None:
    """Add word-level features to the DataFrame in-place."""
    words = df["word"].tolist()
    unique_words = list(dict.fromkeys(words))

    # Predict in batches
    all_scores: list[dict[str, float]] = []
    iterator = range(0, len(unique_words), batch_size)
    if not quiet:
        iterator = tqdm(iterator, desc=model.name)

    for batch_start in iterator:
        batch = unique_words[batch_start : batch_start + batch_size]
        try:
            scores = model.predict_batch(batch)
        except Exception as e:
            raise InferenceError(model.name, str(e)) from e
        all_scores.extend(scores)

    # Build lookup from unique word -> scores
    word_to_scores = dict(zip(unique_words, all_scores))

    # Map back to DataFrame rows
    feature_names = list(all_scores[0].keys()) if all_scores else []
    for feat in feature_names:
        df[feat] = [word_to_scores[w][feat] for w in words]


def _score_chunk_level(
    df: pd.DataFrame,
    chunks: list[str],
    model: BaseModel,
    *,
    batch_size: int = 64,
    quiet: bool = False,
) -> np.ndarray | None:
    """Score chunks and return embedding array.

    Returns shape ``(n_chunks, dim)`` or None if no chunks.
    """
    if not chunks:
        return None

    all_scores: list[dict[str, float]] = []
    iterator = range(0, len(chunks), batch_size)
    if not quiet:
        iterator = tqdm(iterator, desc=model.name)

    for batch_start in iterator:
        batch = chunks[batch_start : batch_start + batch_size]
        try:
            scores = model.predict_batch(batch)
        except Exception as e:
            raise InferenceError(model.name, str(e)) from e
        all_scores.extend(scores)

    # Convert to numpy array
    feature_names = sorted(all_scores[0].keys())
    emb_array = np.array(
        [[s[f] for f in feature_names] for s in all_scores], dtype=np.float32
    )

    return emb_array


def save_embeddings(
    embeddings: dict[str, np.ndarray],
    output_path: str | Path,
    df: pd.DataFrame,
) -> Path:
    """Save embedding arrays to HDF5 alongside the CSV.

    Parameters
    ----------
    embeddings : dict[str, np.ndarray]
        Model name -> array of shape ``(n_chunks, dim)``.
    output_path : str or Path
        Path to the CSV output file. HDF5 gets the same stem with ``.h5``.
    df : pd.DataFrame
        The word-per-row DataFrame (used to build chunk_index).

    Returns
    -------
    Path to the saved HDF5 file.
    """
    h5_path = Path(output_path).with_suffix(".h5")

    with h5py.File(h5_path, "w") as f:
        # Write chunk_index: maps each word row to its chunk
        f.create_dataset("chunk_index", data=df["chunk_idx"].values)

        for model_name, arr in embeddings.items():
            f.create_dataset(
                f"{model_name}_embeddings",
                data=arr,
                compression="gzip",
                compression_opts=4,
            )

    return h5_path
