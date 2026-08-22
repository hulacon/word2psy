"""Batch inference pipeline: tokenize text, run models, return scores.

Produces two tables mirroring viz2psy's row-per-stimulus CSV layout:

- a words table (one row per word token) holding word-level features, and
- a chunks table (one row per chunk) holding chunk-level features,
  including embeddings flat as columns.
"""

import re
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from word2psy.exceptions import InferenceError, ModelLoadError
from word2psy.models.base import BaseModel
from word2psy.tokenize import split_by_sentence, tokenize_text

# Structural columns of the words table (everything else is a feature)
_WORD_INDEX_COLS = {
    "word_idx", "word", "sentence_idx", "chunk_idx", "chunk_label",
    "onset", "offset",
}
# Embedding dimension columns look like fasttext_000, word2vec_299, ...
# Contract B §4.1 pads the index to a fixed width, so spaces wider than
# 1,000 dimensions use four digits (ebind_text_1023) -- match 3 or more.
_EMBEDDING_COL = re.compile(r"_\d{3,}$")

_AGGREGATE_STATS = ["mean", "sd", "min", "max"]


def score_text(
    text: str | list[str],
    models: list[BaseModel],
    *,
    chunk_labels: list[str] | None = None,
    passthrough: pd.DataFrame | None = None,
    keep_punctuation: bool = False,
    by_sentence: bool = False,
    aggregate_words: bool = True,
    pool_embeddings: bool = True,
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
    by_sentence : bool
        If True, re-chunk the input so every sentence becomes its own
        chunk (labels ``"{original_label}/s{j}"``; passthrough rows are
        repeated across a chunk's sentences). Chunk-level models then
        score individual sentences.
    aggregate_words : bool
        If True (default), append per-chunk aggregates of word-level
        scalar features to the chunks table as ``{feature}_{stat}``
        columns (mean/sd/min/max, NaN-aware; sd is NaN for single-word
        chunks). Embedding columns are not aggregated -- see
        ``pool_embeddings``.
    pool_embeddings : bool
        If True (default), mean-pool word-level embedding columns
        (``{model}_{NNN}``) within each chunk and write them to the
        chunks table under the same names, so a static word embedding
        becomes a chunk-level space comparable to CLIP/MiniLM/EBind.
        Adds a ``{model}_n_pooled`` count per space.
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
    if by_sentence:
        chunks_in = [text] if isinstance(text, str) else list(text)
        text, chunk_labels, origin = split_by_sentence(chunks_in, chunk_labels)
        if passthrough is not None:
            passthrough = passthrough.iloc[origin].reset_index(drop=True)

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

    if aggregate_words:
        aggregate_word_features(words_df, chunks_df)
    if pool_embeddings:
        pooled = pool_word_embeddings(words_df, chunks_df)
        # Attribute pooled columns back to the model that produced them,
        # mirroring the ``feature_names_`` convention above.
        for model in models:
            own = [c for c in pooled if c.startswith(f"{model.name}_")]
            if own:
                model.pooled_features_ = own

    return words_df, chunks_df


def aggregate_word_features(
    words_df: pd.DataFrame,
    chunks_df: pd.DataFrame,
) -> list[str]:
    """Append per-chunk aggregates of word-level scalar features in-place.

    For every numeric feature column in the words table (embedding
    dimensions excluded), adds ``{feature}_mean``, ``{feature}_sd``,
    ``{feature}_min``, and ``{feature}_max`` columns to the chunks table.
    Aggregation is NaN-aware (e.g. word2vec OOV rows are skipped); sd uses
    ddof=1 and is therefore NaN for single-word chunks.

    Returns the list of columns added.
    """
    feature_cols = [
        c
        for c in words_df.columns
        if c not in _WORD_INDEX_COLS
        and not _EMBEDDING_COL.search(c)
        and pd.api.types.is_numeric_dtype(words_df[c])
    ]
    if not feature_cols or not len(words_df) or not len(chunks_df):
        return []

    grouped = words_df.groupby("chunk_idx")[feature_cols]
    tables = {
        "mean": grouped.mean(),
        "sd": grouped.std(),
        "min": grouped.min(),
        "max": grouped.max(),
    }

    added = []
    new_cols = {}
    for feat in feature_cols:
        for stat in _AGGREGATE_STATS:
            col = f"{feat}_{stat}"
            new_cols[col] = chunks_df["chunk_idx"].map(tables[stat][feat])
            added.append(col)
    chunks_df[added] = pd.DataFrame(new_cols, index=chunks_df.index)
    return added


def pool_word_embeddings(
    words_df: pd.DataFrame,
    chunks_df: pd.DataFrame,
) -> list[str]:
    """Mean-pool word-level embedding columns into the chunks table in-place.

    Static word embeddings (``word2vec``, ``fasttext``) are the only
    spaces that exist per token and not per chunk, so without pooling
    they cannot be compared against chunk-level spaces such as
    ``clip_text`` or ``minilm``. This writes one pooled vector per chunk
    under the *same* column names the words table uses
    (``word2vec_000``...), which is what makes the result a first-class
    embedding space to a Contract B consumer: psytwill detects a space by
    the ``{prefix}_{NNN}`` pattern and nothing else.

    Pooling is the unweighted, NaN-aware mean over a chunk's words, so
    out-of-vocabulary tokens (word2vec returns NaN for these; fastText's
    subwords never do) drop out rather than poisoning the vector. A
    ``{prefix}_n_pooled`` column records how many words actually
    contributed, since a vector pooled over 2 of 9 words is not the same
    evidence as one pooled over all 9. A chunk with no in-vocabulary word
    gets an all-NaN vector and ``n_pooled = 0``.

    Deliberately *not* mean/sd/min/max: per-dimension spread of an
    embedding is not interpretable the way it is for a lexical norm, and
    the four-stat form would emit 1,200 columns for a 300-d space while
    breaking the prefix pattern consumers match on.

    Returns the list of columns added.
    """
    embed_cols = [
        c
        for c in words_df.columns
        if c not in _WORD_INDEX_COLS
        and _EMBEDDING_COL.search(c)
        and pd.api.types.is_numeric_dtype(words_df[c])
    ]
    if not embed_cols or not len(words_df) or not len(chunks_df):
        return []

    # Group columns by their model prefix (everything before _NNN)
    by_prefix: dict[str, list[str]] = {}
    for col in embed_cols:
        by_prefix.setdefault(_EMBEDDING_COL.sub("", col), []).append(col)

    grouped = words_df.groupby("chunk_idx")
    means = grouped[embed_cols].mean()  # NaN-aware

    added = []
    new_cols = {}
    for prefix, cols in by_prefix.items():
        # Count words carrying a real vector for this space, not merely
        # present in the chunk -- word2vec OOV rows are all-NaN.
        valid = words_df[cols[0]].notna()
        n_pooled = valid.groupby(words_df["chunk_idx"]).sum()
        count_col = f"{prefix}_n_pooled"
        new_cols[count_col] = (
            chunks_df["chunk_idx"].map(n_pooled).fillna(0).astype(int)
        )
        added.append(count_col)
        for col in cols:
            new_cols[col] = chunks_df["chunk_idx"].map(means[col])
            added.append(col)

    chunks_df[added] = pd.DataFrame(new_cols, index=chunks_df.index)
    return added


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
