"""Batch inference pipeline: tokenize text, run models, return scores.

Produces two tables mirroring viz2psy's row-per-stimulus CSV layout:

- a words table (one row per word token) holding word-level features, and
- a chunks table (one row per chunk) holding chunk-level features,
  including embeddings flat as columns.
"""

import time

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
    passthrough: pd.DataFrame | None = None,
    keep_punctuation: bool = False,
    batch_size: int = 64,
    quiet: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score text with one or more feature models.

    Parameters
    ----------
    text : str or list[str]
        A single string (one chunk) or list of strings (multiple chunks).
    models : list of BaseModel
        Instantiated (but not yet loaded) model wrappers.
    chunk_labels : list[str], optional
        Labels for each chunk.
    passthrough : pd.DataFrame, optional
        Extra per-chunk columns (e.g. stimulus IDs, conditions from a CSV
        input) carried into the chunks table. Must have one row per chunk.
    keep_punctuation : bool
        If True, keep punctuation tokens in the word table.
    batch_size : int
        Number of items per forward pass.
    quiet : bool
        If True, suppress progress output.

    Returns
    -------
    words_df : pd.DataFrame
        Word-per-row table: index columns (word_idx, word, sentence_idx,
        chunk_idx, chunk_label, onset, offset) plus one column per
        word-level feature.
    chunks_df : pd.DataFrame
        Chunk-per-row table: chunk_idx, chunk_label, n_words, any
        passthrough columns, plus one column per chunk-level feature
        (embeddings appear flat, e.g. clip_text_000...clip_text_511).

    Notes
    -----
    After scoring, each model instance carries a ``feature_names_``
    attribute listing the columns it produced.
    """
    # Build the word-per-row table
    words_df = tokenize_text(
        text, chunk_labels=chunk_labels, keep_punctuation=keep_punctuation
    )
    chunks = [text] if isinstance(text, str) else list(text)

    # Build the chunk-per-row table skeleton
    if chunk_labels is None:
        labels = [f"chunk_{i}" for i in range(len(chunks))]
    else:
        labels = [str(lab) for lab in chunk_labels]

    words_per_chunk = (
        words_df.groupby("chunk_idx").size() if len(words_df) else pd.Series(dtype=int)
    )
    chunks_df = pd.DataFrame(
        {
            "chunk_idx": range(len(chunks)),
            "chunk_label": labels,
            "n_words": [int(words_per_chunk.get(i, 0)) for i in range(len(chunks))],
        }
    )

    if passthrough is not None:
        if len(passthrough) != len(chunks):
            raise ValueError(
                f"passthrough has {len(passthrough)} rows but there are "
                f"{len(chunks)} chunks"
            )
        pt = passthrough.reset_index(drop=True)
        for col in pt.columns:
            if col not in chunks_df.columns:
                chunks_df[col] = pt[col].values

    for model in models:
        if not quiet:
            print(f"Loading {model.name} on {model.device}...")
        try:
            model.load()
        except Exception as e:
            raise ModelLoadError(model.name, str(e)) from e

        start_time = time.time()

        if model.level == "word":
            _score_word_level(words_df, model, batch_size=batch_size, quiet=quiet)
        elif model.level == "chunk":
            _score_chunk_level(
                chunks_df, chunks, model, batch_size=batch_size, quiet=quiet
            )
        elif model.level == "context":
            _score_context_level(words_df, chunks, model, quiet=quiet)
        else:
            raise InferenceError(
                model.name, f"Unknown model level: {model.level!r}"
            )

        # Free weights so large models are not resident simultaneously
        # (fastText alone is ~7 GB); load() re-runs on next use.
        model.unload()

        elapsed = time.time() - start_time
        if not quiet:
            print(f"  {model.name} completed in {elapsed:.1f}s")

    return words_df, chunks_df


def _predict_in_batches(
    items: list[str],
    model: BaseModel,
    *,
    batch_size: int,
    quiet: bool,
) -> list[dict[str, float]]:
    """Run model.predict_batch over items, batched, with progress."""
    all_scores: list[dict[str, float]] = []
    iterator = range(0, len(items), batch_size)
    if not quiet:
        iterator = tqdm(iterator, desc=model.name)

    for batch_start in iterator:
        batch = items[batch_start : batch_start + batch_size]
        try:
            scores = model.predict_batch(batch)
        except Exception as e:
            raise InferenceError(model.name, str(e)) from e
        all_scores.extend(scores)

    return all_scores


def _score_word_level(
    words_df: pd.DataFrame,
    model: BaseModel,
    *,
    batch_size: int = 64,
    quiet: bool = False,
) -> None:
    """Add word-level features to the words DataFrame in-place."""
    words = words_df["word"].tolist()
    unique_words = list(dict.fromkeys(words))

    all_scores = _predict_in_batches(
        unique_words, model, batch_size=batch_size, quiet=quiet
    )

    # Build lookup from unique word -> scores
    word_to_scores = dict(zip(unique_words, all_scores))

    # Map back to DataFrame rows
    feature_names = list(all_scores[0].keys()) if all_scores else []
    for feat in feature_names:
        words_df[feat] = [word_to_scores[w][feat] for w in words]

    model.feature_names_ = feature_names


def _score_context_level(
    words_df: pd.DataFrame,
    chunks: list[str],
    model: BaseModel,
    *,
    quiet: bool = False,
) -> None:
    """Add context-dependent word-level features to the words DataFrame.

    Unlike word-level scoring there is no deduplication: the same word
    gets different values at different positions.
    """
    feature_names: list[str] = []

    iterator = list(enumerate(chunks))
    if not quiet:
        iterator = tqdm(iterator, desc=model.name)

    for chunk_idx, chunk_text in iterator:
        rows = words_df.index[words_df["chunk_idx"] == chunk_idx]
        words = words_df.loc[rows, "word"].tolist()
        if not words:
            continue
        try:
            scores = model.predict_context(chunk_text, words)
        except Exception as e:
            raise InferenceError(model.name, str(e)) from e

        if not feature_names and scores:
            feature_names = list(scores[0].keys())
            for feat in feature_names:
                if feat not in words_df.columns:
                    words_df[feat] = np.nan
        for row, s in zip(rows, scores):
            for feat in feature_names:
                words_df.loc[row, feat] = s[feat]

    model.feature_names_ = feature_names


def _score_chunk_level(
    chunks_df: pd.DataFrame,
    chunks: list[str],
    model: BaseModel,
    *,
    batch_size: int = 64,
    quiet: bool = False,
) -> None:
    """Add chunk-level features to the chunks DataFrame in-place."""
    if not chunks:
        model.feature_names_ = []
        return

    all_scores = _predict_in_batches(
        chunks, model, batch_size=batch_size, quiet=quiet
    )

    # Preserve the model's own feature ordering (dict insertion order)
    feature_names = list(all_scores[0].keys()) if all_scores else []
    for feat in feature_names:
        chunks_df[feat] = [s[feat] for s in all_scores]

    model.feature_names_ = feature_names
