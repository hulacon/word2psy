"""End-to-end pipeline tests.

These tests require all models and norm data to be available.
"""

import numpy as np
import pandas as pd
import pytest

from word2psy.pipeline import score_text
from word2psy.models.clip_text import CLIPTextModel
from word2psy.models.lexical_norms import LexicalNormsModel


@pytest.fixture(scope="module")
def models():
    """Pre-load both models."""
    clip = CLIPTextModel(device="cpu")
    norms = LexicalNormsModel(device="cpu")
    return [clip, norms]


class TestPipelineBasic:
    def test_single_string_input(self, models):
        df, emb = score_text("The dog chased the cat.", models, quiet=True)
        assert isinstance(df, pd.DataFrame)
        assert "word" in df.columns
        assert "concreteness" in df.columns
        assert len(df) > 0

    def test_embeddings_returned(self, models):
        df, emb = score_text("Hello world.", models, quiet=True)
        assert "clip_text" in emb
        arr = emb["clip_text"]
        assert arr.shape == (1, 512)  # 1 chunk, 512-d

    def test_multi_chunk_input(self, models):
        chunks = ["First sentence here.", "Second sentence there."]
        df, emb = score_text(chunks, models, quiet=True)
        assert df["chunk_idx"].nunique() == 2
        assert emb["clip_text"].shape[0] == 2

    def test_word_features_vary_per_word(self, models):
        df, _ = score_text("The enormous tiny cat.", models, quiet=True)
        # Different words should generally have different concreteness
        conc = df["concreteness"].tolist()
        assert len(set(conc)) > 1  # Not all identical

    def test_output_has_correct_structure(self, models):
        df, _ = score_text("A simple test.", models, quiet=True)
        required_cols = [
            "word_idx",
            "word",
            "sentence_idx",
            "chunk_idx",
            "chunk_label",
        ]
        for col in required_cols:
            assert col in df.columns


class TestPipelineWordOnly:
    def test_lexical_norms_only(self):
        norms = LexicalNormsModel(device="cpu")
        df, emb = score_text("bright dark loud quiet", [norms], quiet=True)
        assert "concreteness" in df.columns
        assert "zipf_frequency" in df.columns
        assert len(emb) == 0  # No chunk-level embeddings


class TestPipelineChunkOnly:
    def test_clip_only(self):
        clip = CLIPTextModel(device="cpu")
        df, emb = score_text("A photo of a sunset.", [clip], quiet=True)
        assert "clip_text" in emb
        # No word-level features added (only index columns)
        assert "concreteness" not in df.columns
