"""Tests for cross-modal similarity (word2psy clip_text x viz2psy clip).

Fully offline: synthetic embeddings, no model loading.
"""

import numpy as np
import pandas as pd
import pytest

from word2psy.crossmodal import (
    clip_columns,
    cross_modal_similarity,
    top_matches,
)


def _text_df(vectors, labels=None, dim=8):
    data = {}
    if labels is not None:
        data["chunk_label"] = labels
        data["chunk_idx"] = np.arange(len(vectors))
    for i in range(dim):
        data[f"clip_text_{i:03d}"] = [v[i] for v in vectors]
    return pd.DataFrame(data)


def _image_df(vectors, filenames=None, dim=8):
    data = {}
    if filenames is not None:
        data["filename"] = filenames
    for i in range(dim):
        data[f"clip_{i:03d}"] = [v[i] for v in vectors]
    return pd.DataFrame(data)


def _unit(dim, hot):
    v = np.zeros(dim)
    v[hot] = 1.0
    return v


class TestClipColumns:
    def test_disambiguates_text_from_image(self):
        # A combined frame with both naming schemes must split cleanly
        df = pd.DataFrame({
            "clip_000": [0.1], "clip_001": [0.2],
            "clip_text_000": [0.3], "clip_text_001": [0.4],
            "filename": ["x.png"],
        })
        assert clip_columns(df, "image") == ["clip_000", "clip_001"]
        assert clip_columns(df, "text") == ["clip_text_000", "clip_text_001"]

    def test_numeric_ordering(self):
        cols = [f"clip_text_{i:03d}" for i in [2, 0, 10, 1]]
        df = pd.DataFrame({c: [0.0] for c in cols})
        assert clip_columns(df, "text") == [
            "clip_text_000", "clip_text_001", "clip_text_002", "clip_text_010"
        ]


class TestCrossModalSimilarity:
    def test_identical_vectors_give_one(self):
        v = [_unit(8, 0), _unit(8, 1)]
        sim = cross_modal_similarity(
            _text_df(v, labels=["a", "b"]),
            _image_df(v, filenames=["a.png", "b.png"]),
        )
        assert sim.shape == (2, 2)
        assert sim.loc["a", "a.png"] == pytest.approx(1.0)
        assert sim.loc["b", "b.png"] == pytest.approx(1.0)
        assert sim.loc["a", "b.png"] == pytest.approx(0.0)

    def test_renormalizes_unnormalized_input(self):
        sim = cross_modal_similarity(
            _text_df([_unit(8, 0) * 5.0], labels=["a"]),
            _image_df([_unit(8, 0) * 0.2], filenames=["a.png"]),
        )
        assert sim.loc["a", "a.png"] == pytest.approx(1.0)

    def test_labels_fall_back_to_indices(self):
        v = [_unit(8, 0), _unit(8, 1), _unit(8, 2)]
        sim = cross_modal_similarity(_text_df(v), _image_df(v))
        assert list(sim.index) == ["row_0", "row_1", "row_2"]
        assert list(sim.columns) == ["image_0", "image_1", "image_2"]

    def test_missing_text_columns_raises(self):
        with pytest.raises(ValueError, match="clip_text"):
            cross_modal_similarity(
                pd.DataFrame({"x": [1.0]}), _image_df([_unit(8, 0)])
            )

    def test_missing_image_columns_raises(self):
        with pytest.raises(ValueError, match=r"clip_\*"):
            cross_modal_similarity(
                _text_df([_unit(8, 0)]), pd.DataFrame({"x": [1.0]})
            )

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError, match="dimensions differ"):
            cross_modal_similarity(
                _text_df([_unit(8, 0)], dim=8),
                _image_df([_unit(4, 0)], dim=4),
            )


class TestTopMatches:
    def test_ranks_matching_pairs_first(self):
        rng = np.random.RandomState(0)
        base = [rng.randn(8) for _ in range(3)]
        # Images are noisy copies of the texts: diagonal should win
        noisy = [b + rng.randn(8) * 0.1 for b in base]
        sim = cross_modal_similarity(
            _text_df(base, labels=["t0", "t1", "t2"]),
            _image_df(noisy, filenames=["i0.png", "i1.png", "i2.png"]),
        )
        top = top_matches(sim, k=1)
        assert list(top["text"]) == ["t0", "t1", "t2"]
        assert list(top["image"]) == ["i0.png", "i1.png", "i2.png"]
        assert (top["rank"] == 1).all()

    def test_k_limits_rows(self):
        v = [_unit(8, i) for i in range(3)]
        sim = cross_modal_similarity(_text_df(v), _image_df(v))
        assert len(top_matches(sim, k=2)) == 6
