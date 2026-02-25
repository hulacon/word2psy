"""Tests for visualization module.

Uses synthetic DataFrames to test visualization functions without model loading.
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for testing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from word2psy.viz.index_utils import detect_index_column, prepare_index_values
from word2psy.viz.feature_config import (
    FEATURE_CONFIGS,
    detect_models_in_dataframe,
    get_mds_features,
    get_timeseries_features,
    get_visualization_recommendations,
)
from word2psy.viz.projection import compute_projection, PROJECTION_METHODS
from word2psy.viz.timeseries import get_feature_columns, plot_timeseries
from word2psy.viz.heatmap import plot_heatmap
from word2psy.viz.scatter import plot_scatter


@pytest.fixture
def word_df():
    """Create a synthetic word-level DataFrame mimicking word2psy output."""
    rng = np.random.RandomState(42)
    n = 50
    return pd.DataFrame({
        "word_idx": np.arange(n),
        "word": [f"word_{i}" for i in range(n)],
        "sentence_idx": np.repeat(np.arange(10), 5),
        "chunk_idx": np.repeat(np.arange(5), 10),
        "chunk_label": np.repeat([f"chunk_{i}" for i in range(5)], 10),
        "onset": np.nan,
        "offset": np.nan,
        "concreteness": rng.uniform(1, 5, n),
        "valence": rng.uniform(0, 1, n),
        "arousal": rng.uniform(0, 1, n),
        "dominance": rng.uniform(0, 1, n),
        "age_of_acquisition": rng.uniform(2, 15, n),
        "imageability": rng.uniform(1, 7, n),
        "zipf_frequency": rng.uniform(1, 7, n),
        "sensorimotor_touch": rng.uniform(0, 5, n),
        "sensorimotor_hearing": rng.uniform(0, 5, n),
        "sensorimotor_smell": rng.uniform(0, 5, n),
        "sensorimotor_taste": rng.uniform(0, 5, n),
        "sensorimotor_vision": rng.uniform(0, 5, n),
        "sensorimotor_interoception": rng.uniform(0, 5, n),
        "sensorimotor_mouth": rng.uniform(0, 5, n),
        "sensorimotor_hand": rng.uniform(0, 5, n),
        "sensorimotor_foot": rng.uniform(0, 5, n),
        "sensorimotor_head": rng.uniform(0, 5, n),
        "sensorimotor_torso": rng.uniform(0, 5, n),
    })


@pytest.fixture
def chunk_df():
    """Create a synthetic chunk-level DataFrame with embedding columns."""
    rng = np.random.RandomState(42)
    n = 20
    data = {
        "chunk_idx": np.arange(n),
        "chunk_label": [f"segment_{i}" for i in range(n)],
    }
    # Add CLIP text embedding columns
    for i in range(512):
        data[f"clip_text_{i:03d}"] = rng.randn(n)
    return pd.DataFrame(data)


# --- Index Utils ---

class TestIndexUtils:
    def test_detect_word_idx(self, word_df):
        col, idx_type = detect_index_column(word_df)
        assert col == "word_idx"
        assert idx_type == "integer"

    def test_detect_chunk_idx(self):
        df = pd.DataFrame({"chunk_idx": [0, 1, 2], "val": [1.0, 2.0, 3.0]})
        col, idx_type = detect_index_column(df)
        assert col == "chunk_idx"
        assert idx_type == "integer"

    def test_detect_onset(self):
        df = pd.DataFrame({"onset": [0.0, 1.5, 3.0], "val": [1.0, 2.0, 3.0]})
        col, idx_type = detect_index_column(df)
        assert col == "onset"
        assert idx_type == "time"

    def test_detect_fallback(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        col, idx_type = detect_index_column(df)
        assert col is None
        assert idx_type == "integer"

    def test_prepare_index_none(self, word_df):
        x, info = prepare_index_values(word_df, None, "integer")
        assert len(x) == len(word_df)
        assert info["xlabel"] == "Index"

    def test_prepare_index_ordinal(self):
        df = pd.DataFrame({
            "chunk_label": [f"ch_{i}" for i in range(30)],
            "val": np.arange(30),
        })
        x, info = prepare_index_values(df, "chunk_label", "ordinal")
        assert len(x) == 30
        assert info["tickmode"] == "array"
        # Should subsample ticks for >20 points
        assert len(info["tickvals"]) < 30


# --- Feature Config ---

class TestFeatureConfig:
    def test_lexical_norms_config(self):
        cfg = FEATURE_CONFIGS["lexical_norms"]
        assert cfg.timeseries is True
        assert cfg.mds_clustering is True
        assert cfg.n_dims == 18

    def test_clip_text_config(self):
        cfg = FEATURE_CONFIGS["clip_text"]
        assert cfg.timeseries is False
        assert cfg.mds_clustering is True
        assert cfg.n_dims == 512

    def test_detect_models_lexical_norms(self, word_df):
        detected = detect_models_in_dataframe(word_df.columns.tolist())
        assert "lexical_norms" in detected

    def test_detect_models_clip_text(self, chunk_df):
        detected = detect_models_in_dataframe(chunk_df.columns.tolist())
        assert "clip_text" in detected

    def test_get_timeseries_features(self, word_df):
        ts_feats = get_timeseries_features(word_df.columns.tolist())
        assert "concreteness" in ts_feats
        assert "valence" in ts_feats
        assert "sensorimotor_touch" in ts_feats
        # Embeddings should not appear
        assert not any(f.startswith("clip_text_") for f in ts_feats)

    def test_get_mds_features(self, word_df):
        mds = get_mds_features(word_df.columns.tolist())
        assert "lexical_norms" in mds
        assert len(mds["lexical_norms"]) >= 2

    def test_get_mds_features_clip(self, chunk_df):
        mds = get_mds_features(chunk_df.columns.tolist())
        assert "clip_text" in mds
        assert len(mds["clip_text"]) == 512

    def test_recommendations(self, word_df):
        recs = get_visualization_recommendations(word_df.columns.tolist())
        assert recs["timeseries"]["available"]
        assert recs["heatmap"]["available"]
        assert recs["scatter"]["available"]


# --- Projection ---

class TestProjection:
    def test_pca_shape(self):
        X = np.random.randn(20, 5)
        X_proj, info = compute_projection(X, method="pca")
        assert X_proj.shape == (20, 2)
        assert "xlabel" in info
        assert "ylabel" in info

    def test_ppca_shape(self):
        X = np.random.randn(20, 5)
        X_proj, info = compute_projection(X, method="ppca")
        assert X_proj.shape == (20, 2)

    def test_ppca_missing_data(self):
        X = np.random.randn(20, 5)
        X[0, 2] = np.nan
        X[5, 1] = np.nan
        X_proj, info = compute_projection(X, method="ppca")
        assert X_proj.shape == (20, 2)
        assert info.get("missing_data_handled") is True

    def test_tsne_shape(self):
        X = np.random.randn(20, 5)
        X_proj, info = compute_projection(X, method="tsne")
        assert X_proj.shape == (20, 2)

    def test_mds_shape(self):
        X = np.random.randn(20, 5)
        X_proj, info = compute_projection(X, method="mds")
        assert X_proj.shape == (20, 2)

    def test_invalid_method(self):
        X = np.random.randn(20, 5)
        with pytest.raises(ValueError, match="Unknown method"):
            compute_projection(X, method="invalid")

    def test_too_few_samples(self):
        X = np.random.randn(2, 5)
        with pytest.raises(ValueError, match="at least 3 samples"):
            compute_projection(X, method="pca")


# --- Timeseries ---

class TestTimeseries:
    def test_get_feature_columns_default(self, word_df):
        cols = get_feature_columns(word_df)
        assert "concreteness" in cols
        assert "word_idx" not in cols
        assert "onset" not in cols

    def test_get_feature_columns_pattern(self, word_df):
        cols = get_feature_columns(word_df, patterns=["sensorimotor_*"])
        assert all(c.startswith("sensorimotor_") for c in cols)
        assert len(cols) == 11

    def test_plot_returns_figure(self, word_df):
        fig = plot_timeseries(word_df, features=["concreteness", "valence"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_auto_excludes_embeddings(self, chunk_df):
        # chunk_df has 512 clip_text_* columns; default should exclude them
        with pytest.raises(ValueError, match="No features found"):
            plot_timeseries(chunk_df)

    def test_plot_with_rolling(self, word_df):
        fig = plot_timeseries(word_df, features=["concreteness"], rolling_window=5)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_with_diff(self, word_df):
        fig = plot_timeseries(word_df, features=["concreteness"], show_diff=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_many_features(self, word_df):
        fig = plot_timeseries(word_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# --- Heatmap ---

class TestHeatmap:
    def test_plot_returns_figure(self, word_df):
        fig = plot_heatmap(word_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_with_patterns(self, word_df):
        fig = plot_heatmap(word_df, features=["concreteness", "valence", "arousal"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_spearman(self, word_df):
        fig = plot_heatmap(word_df, method="spearman")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_too_few_features(self):
        df = pd.DataFrame({"word_idx": [0, 1], "x": [1.0, 2.0]})
        with pytest.raises(ValueError, match="at least 2 features"):
            plot_heatmap(df)


# --- Scatter ---

class TestScatter:
    def test_plot_returns_figure(self, word_df):
        fig = plot_scatter(word_df, features=["concreteness", "valence", "arousal"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_pca_default(self, word_df):
        fig = plot_scatter(word_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_color_by(self, word_df):
        fig = plot_scatter(word_df, color_by="chunk_idx")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_too_few_features(self):
        df = pd.DataFrame({"word_idx": [0, 1, 2], "x": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="at least 2 features"):
            plot_scatter(df)
