"""Tests for CLIP text model.

These tests require the CLIP model to be downloaded (~400MB on first run).
Mark as slow/integration tests if needed.
"""

import numpy as np
import pytest

from word2psy.models.clip_text import CLIPTextModel


@pytest.fixture(scope="module")
def clip_model():
    """Load the CLIP model once for all tests in this module."""
    model = CLIPTextModel(device="cpu")
    model.load()
    return model


class TestCLIPTextModel:
    def test_attributes(self):
        model = CLIPTextModel()
        assert model.name == "clip_text"
        assert model.level == "chunk"

    def test_predict_shape(self, clip_model):
        scores = clip_model.predict("A photo of a cat")
        assert len(scores) == 512
        assert all(k.startswith("clip_text_") for k in scores)

    def test_predict_l2_normalized(self, clip_model):
        scores = clip_model.predict("A photo of a cat")
        vec = np.array(list(scores.values()))
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 0.01

    def test_predict_batch(self, clip_model):
        texts = ["A photo of a cat", "A painting of a sunset", "Hello world"]
        results = clip_model.predict_batch(texts)
        assert len(results) == 3
        for r in results:
            assert len(r) == 512

    def test_different_texts_produce_different_embeddings(self, clip_model):
        r1 = clip_model.predict("cat")
        r2 = clip_model.predict("democracy")
        v1 = np.array(list(r1.values()))
        v2 = np.array(list(r2.values()))
        cosine_sim = np.dot(v1, v2)
        # Very different concepts should have low similarity
        assert cosine_sim < 0.9

    def test_column_naming(self, clip_model):
        scores = clip_model.predict("test")
        keys = sorted(scores.keys())
        assert keys[0] == "clip_text_000"
        assert keys[-1] == "clip_text_511"
