"""Merge standalone per-model scores CSVs into one dashboard table pair.

Each word2psy model writes its own ``*_words.csv`` / ``*_chunks.csv`` pair,
all sharing the same scaffold (``stimulus_id``, ``word_idx`` / ``chunk_idx``,
``word``, ``onset``, ...). The dashboard renders every model it detects in a
single DataFrame, so an all-models view is a merge problem, solved here:
words tables merge on ``(stimulus_id, word_idx)``, chunks tables on
``(stimulus_id, chunk_idx)``; scaffold columns are kept from the first input
that carries them.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Scaffold columns shared by every model's output; first occurrence wins.
CARRIER_COLUMNS = frozenset({
    "word", "sentence_idx", "chunk_label", "n_words", "onset", "offset",
    "word_idx", "chunk_idx", "time",
})

WORDS_KEYS = ["stimulus_id", "word_idx"]
CHUNKS_KEYS = ["stimulus_id", "chunk_idx"]


def collect_scores_csvs(directory: Path) -> tuple[list[Path], list[Path]]:
    """(words, chunks) CSVs directly inside ``directory``."""
    words = [p for p in sorted(directory.glob("*_words.csv"))
             if not p.name.startswith(".")]
    chunks = [p for p in sorted(directory.glob("*_chunks.csv"))
              if not p.name.startswith(".")]
    if not words and not chunks:
        raise FileNotFoundError(
            f"no *_words.csv / *_chunks.csv files found in {directory}")
    return words, chunks


def merge_scores(frames: dict[str, pd.DataFrame],
                 keys: list[str]) -> pd.DataFrame:
    """Merge per-model score DataFrames (name -> frame) on ``keys``."""
    if not frames:
        raise ValueError("no input frames to merge")

    owners: dict[str, str] = {}
    for name, df in frames.items():
        missing = [k for k in keys if k not in df.columns]
        if missing:
            raise ValueError(
                f"{name} lacks {'/'.join(missing)}; scores CSVs share the "
                f"{'/'.join(keys)} scaffold")
        if df.duplicated(subset=keys).any():
            raise ValueError(
                f"{name} has several rows per ({', '.join(keys)}); merge "
                "would multiply rows — exclude this file")
        for col in df.columns:
            if col in keys or col in CARRIER_COLUMNS:
                continue
            if col in owners:
                raise ValueError(
                    f"feature column {col!r} appears in both {owners[col]} "
                    f"and {name}; rename or drop one file")
            owners[col] = name

    merged: pd.DataFrame | None = None
    for name, df in frames.items():
        if merged is None:
            merged = df
            continue
        dupes = [c for c in df.columns
                 if c not in keys and c in merged.columns]
        merged = merged.merge(df.drop(columns=dupes), on=keys, how="outer")
    return (merged.sort_values(keys, kind="stable").reset_index(drop=True))


def load_and_merge_dir(directory: Path) -> tuple[pd.DataFrame | None,
                                                 pd.DataFrame | None]:
    """Read and merge every model's (words, chunks) tables in a directory."""
    words_paths, chunks_paths = collect_scores_csvs(directory)
    words = merge_scores(
        {str(p): pd.read_csv(p) for p in words_paths}, WORDS_KEYS
    ) if words_paths else None
    chunks = merge_scores(
        {str(p): pd.read_csv(p) for p in chunks_paths}, CHUNKS_KEYS
    ) if chunks_paths else None
    return words, chunks
