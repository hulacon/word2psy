"""Tests for the interactive HTML dashboard.

Fully offline: synthetic DataFrames, no model loading.
"""

import numpy as np
import pandas as pd
import pytest

from word2psy.cli import resolve_scores_paths
from word2psy.viz.dashboard import _is_scalar_col, create_dashboard


def _make_words_df(n=40, with_embeddings=True, rng_seed=0):
    rng = np.random.RandomState(rng_seed)
    data = {
        "word_idx": np.arange(n),
        "word": [f"word{i}" for i in range(n)],
        "sentence_idx": np.repeat(np.arange(n // 10), 10),
        "chunk_idx": np.repeat(np.arange(n // 20), 20),
        "chunk_label": np.repeat([f"chunk{i}" for i in range(n // 20)], 20),
        "onset": np.nan,
        "offset": np.nan,
        "concreteness": rng.uniform(1, 5, n),
        "valence": rng.uniform(0, 1, n),
        "arousal": rng.uniform(0, 1, n),
        "zipf_frequency": rng.uniform(1, 7, n),
        "length": rng.randint(2, 12, n).astype(float),
        "n_syllables": rng.randint(1, 5, n).astype(float),
        "old20": rng.uniform(1, 3, n),
        "gpt2_surprisal": rng.uniform(0, 25, n),
    }
    for dim in ["touch", "hearing", "vision"]:
        data[f"sensorimotor_{dim}"] = rng.uniform(0, 5, n)
    if with_embeddings:
        for i in range(20):
            data[f"fasttext_{i:03d}"] = rng.randn(n)
            data[f"word2vec_{i:03d}"] = rng.randn(n)
    return pd.DataFrame(data)


def _make_chunks_df(n=8, rng_seed=0):
    rng = np.random.RandomState(rng_seed)
    data = {
        "chunk_idx": np.arange(n),
        "chunk_label": [f"chunk{i}" for i in range(n)],
        "n_words": np.full(n, 20),
        "condition": ["a", "b"] * (n // 2),
        "sentiment_negative": rng.uniform(0, 1, n),
        "sentiment_neutral": rng.uniform(0, 1, n),
        "sentiment_positive": rng.uniform(0, 1, n),
        "readability_flesch_ease": rng.uniform(0, 100, n),
        "readability_smog": rng.uniform(0, 20, n),
    }
    for emo in ["joy", "fear", "sadness", "neutral"]:
        data[f"emotion_{emo}"] = rng.uniform(0, 1, n)
    for i in range(16):
        data[f"clip_text_{i:03d}"] = rng.randn(n)
    return pd.DataFrame(data)


class TestScalarColDetection:
    def test_embedding_cols_excluded(self):
        assert not _is_scalar_col("fasttext_000")
        assert not _is_scalar_col("clip_text_511")

    def test_scalar_cols_included(self):
        assert _is_scalar_col("concreteness")
        assert _is_scalar_col("sensorimotor_touch")
        assert _is_scalar_col("old20")
        assert _is_scalar_col("gpt2_surprisal")


class TestCreateDashboard:
    def test_returns_html(self):
        html = create_dashboard(_make_words_df(), _make_chunks_df())
        assert html.startswith("<!DOCTYPE html>")
        assert "plotly" in html.lower()

    def test_embeds_words_and_models(self):
        html = create_dashboard(_make_words_df(), _make_chunks_df())
        # Word data and detected model entries in the payload
        assert '"word0"' in html
        for model in [
            "lexical_norms", "wordform", "fasttext", "word2vec",
            "gpt2_surprisal", "sentiment", "emotion", "readability",
            "clip_text",
        ]:
            assert f'"{model}"' in html

    def test_detail_panels_configured(self):
        html = create_dashboard(_make_words_df(), _make_chunks_df())
        for panel in ["norms", "sensorimotor", "wordform",
                      "emotions", "sentiment", "readability"]:
            assert f'"id":"{panel}"' in html

    def test_words_only(self):
        html = create_dashboard(_make_words_df(), None)
        assert '"lexical_norms"' in html
        assert '"sentiment"' not in html

    def test_chunks_only(self):
        html = create_dashboard(None, _make_chunks_df())
        assert '"clip_text"' in html

    def test_no_input_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            create_dashboard(None, None)

    def test_no_models_raises(self):
        df = pd.DataFrame({"word_idx": [0, 1, 2], "word": ["a", "b", "c"],
                           "sentence_idx": [0, 0, 0], "chunk_idx": [0, 0, 0],
                           "chunk_label": ["x", "x", "x"]})
        with pytest.raises(ValueError, match="No word2psy model outputs"):
            create_dashboard(df, None)

    def test_truncation(self):
        words = _make_words_df(n=40)
        html = create_dashboard(words, None, max_points=20)
        assert '"words":40' in html  # recorded original size
        assert '"word39"' not in html

    def test_nan_rows_dropped_from_projection(self):
        words = _make_words_df()
        # Simulate word2vec OOV: NaN rows must be dropped, not crash
        w2v_cols = [c for c in words.columns if c.startswith("word2vec_")]
        words.loc[3, w2v_cols] = np.nan
        words.loc[7, w2v_cols] = np.nan
        html = create_dashboard(words, None)
        assert '"word2vec"' in html

    def test_chunk_text_reconstructed(self):
        words = _make_words_df(n=40)
        chunks = _make_chunks_df(n=2)
        html = create_dashboard(words, chunks)
        # chunk 0 text should join its first words
        assert "word0 word1" in html


class TestResolveScoresPaths:
    def _touch(self, tmp_path, *names):
        for name in names:
            (tmp_path / name).write_text("x")

    def test_base_path(self, tmp_path):
        self._touch(tmp_path, "scores_words.csv", "scores_chunks.csv")
        words, chunks = resolve_scores_paths(tmp_path / "scores.csv")
        assert words == tmp_path / "scores_words.csv"
        assert chunks == tmp_path / "scores_chunks.csv"

    def test_words_path(self, tmp_path):
        self._touch(tmp_path, "scores_words.csv")
        words, chunks = resolve_scores_paths(tmp_path / "scores_words.csv")
        assert words == tmp_path / "scores_words.csv"
        assert chunks is None

    def test_chunks_path(self, tmp_path):
        self._touch(tmp_path, "scores_chunks.csv")
        words, chunks = resolve_scores_paths(tmp_path / "scores_chunks.csv")
        assert words is None
        assert chunks == tmp_path / "scores_chunks.csv"

    def test_missing(self, tmp_path):
        words, chunks = resolve_scores_paths(tmp_path / "nope.csv")
        assert words is None and chunks is None
