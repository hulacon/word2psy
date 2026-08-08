"""Tests for the CLAP text model.

Requires the LAION-CLAP checkpoint (~2 GB, shared with aud2psy via the
HF cache; warm on this machine).
"""

import numpy as np
import pytest

from word2psy.models.clap_text import CLAPTextModel


@pytest.fixture(scope="module")
def clap_model():
    model = CLAPTextModel(device="cpu")
    model.load()
    return model


class TestCLAPTextModel:
    def test_attributes(self):
        model = CLAPTextModel()
        assert model.name == "clap_text"
        assert model.level == "chunk"

    def test_predict_shape_and_norm(self, clap_model):
        scores = clap_model.predict("a person speaking calmly")
        assert len(scores) == 512
        assert all(k.startswith("clap_text_") for k in scores)
        vec = np.array([scores[f"clap_text_{i:03d}"] for i in range(512)])
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-4)

    def test_batch_matches_single(self, clap_model):
        texts = ["a drum beat", "rain falling on a roof"]
        batch = clap_model.predict_batch(texts)
        single = clap_model.predict(texts[0])
        v_batch = np.array([batch[0][f"clap_text_{i:03d}"] for i in range(512)])
        v_single = np.array([single[f"clap_text_{i:03d}"] for i in range(512)])
        assert np.allclose(v_batch, v_single, atol=1e-4)

    def test_semantic_ordering(self, clap_model):
        vecs = {
            t: np.array([v[f"clap_text_{i:03d}"] for i in range(512)])
            for t, v in zip(
                ["music", "speech", "melody"],
                clap_model.predict_batch(
                    ["a melody played on a piano", "a person giving a speech",
                     "an instrumental tune on a keyboard"]
                ),
            )
        }
        assert vecs["music"] @ vecs["melody"] > vecs["music"] @ vecs["speech"]
