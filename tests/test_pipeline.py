"""End-to-end pipeline tests.

These tests require all models and norm data to be available.
"""

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
        words_df, chunks_df = score_text("The dog chased the cat.", models, quiet=True)
        assert isinstance(words_df, pd.DataFrame)
        assert "word" in words_df.columns
        assert "lexical_norms_concreteness" in words_df.columns
        assert len(words_df) > 0

    def test_chunk_embeddings_in_chunks_df(self, models):
        words_df, chunks_df = score_text("Hello world.", models, quiet=True)
        emb_cols = [c for c in chunks_df.columns if c.startswith("clip_text_")]
        assert len(emb_cols) == 512
        assert len(chunks_df) == 1

    def test_multi_chunk_input(self, models):
        chunks = ["First sentence here.", "Second sentence there."]
        words_df, chunks_df = score_text(chunks, models, quiet=True)
        assert words_df["chunk_idx"].nunique() == 2
        assert len(chunks_df) == 2
        assert list(chunks_df["chunk_idx"]) == [0, 1]

    def test_word_features_vary_per_word(self, models):
        words_df, _ = score_text("The enormous tiny cat.", models, quiet=True)
        # Different words should generally have different concreteness
        conc = words_df["lexical_norms_concreteness"].tolist()
        assert len(set(conc)) > 1  # Not all identical

    def test_output_has_correct_structure(self, models):
        words_df, chunks_df = score_text("A simple test.", models, quiet=True)
        for col in ["word_idx", "word", "sentence_idx", "chunk_idx", "chunk_label"]:
            assert col in words_df.columns
        for col in ["chunk_idx", "chunk_label", "n_words"]:
            assert col in chunks_df.columns
        assert chunks_df["n_words"].iloc[0] == len(words_df)

    def test_feature_names_recorded(self, models):
        score_text("A simple test.", models, quiet=True)
        for m in models:
            assert len(m.feature_names_) > 0


class TestPipelineWordOnly:
    def test_lexical_norms_only(self):
        norms = LexicalNormsModel(device="cpu")
        words_df, chunks_df = score_text(
            "bright dark loud quiet", [norms], quiet=True
        )
        assert "lexical_norms_concreteness" in words_df.columns
        assert "lexical_norms_zipf_frequency" in words_df.columns
        # No chunk-level model features, but word-feature aggregates are
        # appended by default
        extra = [c for c in chunks_df.columns
                 if c not in ("chunk_idx", "chunk_label", "n_words")]
        assert extra
        assert all(
            c.endswith(("_mean", "_sd", "_min", "_max")) for c in extra
        )
        assert "lexical_norms_concreteness_mean" in chunks_df.columns

    def test_lexical_norms_no_aggregates(self):
        norms = LexicalNormsModel(device="cpu")
        _, chunks_df = score_text(
            "bright dark loud quiet", [norms], aggregate_words=False,
            quiet=True,
        )
        assert list(chunks_df.columns) == ["chunk_idx", "chunk_label", "n_words"]


class TestPipelineChunkOnly:
    def test_clip_only(self):
        clip = CLIPTextModel(device="cpu")
        words_df, chunks_df = score_text("A photo of a sunset.", [clip], quiet=True)
        emb_cols = [c for c in chunks_df.columns if c.startswith("clip_text_")]
        assert len(emb_cols) == 512
        # No word-level features added (only index columns)
        assert "lexical_norms_concreteness" not in words_df.columns
