"""Tests for word2psy.viz.merge — multi-model scores merging for the dashboard."""

import numpy as np
import pandas as pd
import pytest

from word2psy.viz.merge import (
    CHUNKS_KEYS,
    WORDS_KEYS,
    collect_scores_csvs,
    load_and_merge_dir,
    merge_scores,
)


def _words(model, n=4, features=True):
    df = pd.DataFrame({
        "stimulus_id": [f"w{i:02d}" for i in range(n)],
        "word_idx": [0] * n,
        "word": [f"word{i}" for i in range(n)],
        "onset": np.linspace(0, 1, n),
        "offset": np.linspace(0.1, 1.1, n),
    })
    if features:
        df[f"{model}_score"] = np.arange(n, dtype=float)
    return df


def _chunks(model, n=4):
    return pd.DataFrame({
        "stimulus_id": [f"w{i:02d}" for i in range(n)],
        "chunk_idx": [0] * n,
        "chunk_label": [f"chunk{i}" for i in range(n)],
        "n_words": [1] * n,
        f"{model}_val": np.arange(n, dtype=float),
    })


def test_words_merge_dedupes_scaffold():
    merged = merge_scores(
        {"a_words.csv": _words("alpha"), "b_words.csv": _words("beta")},
        WORDS_KEYS)
    assert len(merged) == 4
    assert list(merged.columns).count("word") == 1
    assert {"alpha_score", "beta_score"} <= set(merged.columns)


def test_scaffold_only_words_file_contributes_no_columns():
    merged = merge_scores(
        {"a_words.csv": _words("alpha"),
         "b_words.csv": _words("beta", features=False)},
        WORDS_KEYS)
    assert "alpha_score" in merged.columns
    assert len(merged) == 4


def test_chunks_merge_on_stimulus_and_chunk():
    merged = merge_scores(
        {"a_chunks.csv": _chunks("alpha"), "b_chunks.csv": _chunks("beta")},
        CHUNKS_KEYS)
    assert len(merged) == 4
    assert {"alpha_val", "beta_val"} <= set(merged.columns)


def test_duplicate_scaffold_rows_are_an_error():
    dup = pd.concat([_chunks("alpha"), _chunks("alpha")])
    with pytest.raises(ValueError, match="several rows per"):
        merge_scores({"a_chunks.csv": dup}, CHUNKS_KEYS)


def test_column_collision_names_both_files():
    with pytest.raises(ValueError, match="a_chunks.csv.*b_chunks.csv"):
        merge_scores(
            {"a_chunks.csv": _chunks("alpha"), "b_chunks.csv": _chunks("alpha")},
            CHUNKS_KEYS)


def test_missing_keys_is_an_error():
    with pytest.raises(ValueError, match="word_idx"):
        merge_scores({"a_words.csv": pd.DataFrame({"stimulus_id": ["x"]})},
                     WORDS_KEYS)


def test_collect_and_merge_directory(tmp_path):
    _words("alpha").to_csv(tmp_path / "alpha_words.csv", index=False)
    _words("beta").to_csv(tmp_path / "beta_words.csv", index=False)
    _chunks("alpha").to_csv(tmp_path / "alpha_chunks.csv", index=False)
    words, chunks = load_and_merge_dir(tmp_path)
    assert {"alpha_score", "beta_score"} <= set(words.columns)
    assert "alpha_val" in chunks.columns


def test_collect_empty_dir_errors_with_path(tmp_path):
    with pytest.raises(FileNotFoundError, match=str(tmp_path)):
        collect_scores_csvs(tmp_path)
