"""Tests for chunk-level pooling of word-level embedding spaces.

Fully offline: a fake word-level embedding model stands in for word2vec
and fasttext, including their differing OOV behaviour (word2vec returns
NaN for out-of-vocabulary words, fastText's subwords never do).
"""

import numpy as np
import pandas as pd
import pytest

from word2psy.metadata import MetadataBuilder
from word2psy.models.base import BaseModel
from word2psy.pipeline import pool_word_embeddings, score_text


class FakeEmbedModel(BaseModel):
    """Word-level 3-d embedding; the word "oov" has no vector."""

    name = "fakevec"
    level = "word"

    def load(self):
        self.model = object()

    def predict(self, text):
        if text.strip().lower() == "oov":
            return {f"fakevec_{i:03d}": np.nan for i in range(3)}
        n = float(len(text))
        return {f"fakevec_{i:03d}": n + i for i in range(3)}


TWO_CHUNKS = ["The cat sat.", "One lonely sentence here."]


class TestPoolWordEmbeddings:
    def _tables(self):
        words_df = pd.DataFrame({
            "word_idx": range(4),
            "word": ["a", "bb", "ccc", "dddd"],
            "sentence_idx": [0, 0, 1, 1],
            "chunk_idx": [0, 0, 1, 1],
            "chunk_label": ["c0", "c0", "c1", "c1"],
            "onset": np.nan,
            "offset": np.nan,
            "feat": [1.0, 3.0, 5.0, 7.0],          # scalar: not pooled
            "fakevec_000": [1.0, 3.0, 5.0, np.nan],  # embedding dim
            "fakevec_001": [0.0, 2.0, 4.0, np.nan],
        })
        chunks_df = pd.DataFrame({
            "chunk_idx": [0, 1], "chunk_label": ["c0", "c1"], "n_words": [2, 2],
        })
        return words_df, chunks_df

    def test_pools_mean_and_counts(self):
        words_df, chunks_df = self._tables()
        added = pool_word_embeddings(words_df, chunks_df)
        assert added == ["fakevec_n_pooled", "fakevec_000", "fakevec_001"]
        assert chunks_df["fakevec_000"].tolist() == [2.0, 5.0]  # NaN omitted
        assert chunks_df["fakevec_001"].tolist() == [1.0, 4.0]
        # chunk 1 had two words but only one in-vocabulary
        assert chunks_df["fakevec_n_pooled"].tolist() == [2, 1]

    def test_scalar_features_untouched(self):
        words_df, chunks_df = self._tables()
        pool_word_embeddings(words_df, chunks_df)
        assert "feat" not in chunks_df.columns

    def test_column_names_match_the_words_table(self):
        """Contract B §4.1: consumers detect a space by {prefix}_{NNN}."""
        words_df, chunks_df = self._tables()
        pool_word_embeddings(words_df, chunks_df)
        dims = [c for c in chunks_df.columns if c.startswith("fakevec_0")]
        assert dims == ["fakevec_000", "fakevec_001"]
        assert not any(c.endswith(("_mean", "_sd", "_min", "_max"))
                       for c in chunks_df.columns)

    def test_all_oov_chunk_is_nan_not_zero(self):
        words_df, chunks_df = self._tables()
        words_df.loc[[0, 1], ["fakevec_000", "fakevec_001"]] = np.nan
        pool_word_embeddings(words_df, chunks_df)
        assert np.isnan(chunks_df["fakevec_000"][0])
        assert chunks_df["fakevec_n_pooled"][0] == 0

    def test_wide_embeddings_pool(self):
        """>999-d spaces use four-digit indices and must still match."""
        words_df, chunks_df = self._tables()
        words_df["wide_1023"] = [1.0, 3.0, 5.0, 7.0]
        added = pool_word_embeddings(words_df, chunks_df)
        assert "wide_1023" in added
        assert chunks_df["wide_1023"].tolist() == [2.0, 6.0]

    def test_no_embeddings_noop(self):
        words_df, chunks_df = self._tables()
        words_df = words_df.drop(columns=["fakevec_000", "fakevec_001"])
        assert pool_word_embeddings(words_df, chunks_df) == []


class TestThroughPipeline:
    def test_pooled_by_default(self):
        _, chunks_df = score_text(
            TWO_CHUNKS, [FakeEmbedModel(device="cpu")], quiet=True
        )
        # "The cat sat." -> lengths 3, 3, 3 -> dim 0 mean = 3.0
        assert chunks_df["fakevec_000"].iloc[0] == pytest.approx(3.0)
        assert chunks_df["fakevec_002"].iloc[0] == pytest.approx(5.0)
        assert chunks_df["fakevec_n_pooled"].iloc[0] == 3

    def test_optout(self):
        _, chunks_df = score_text(
            TWO_CHUNKS, [FakeEmbedModel(device="cpu")],
            pool_embeddings=False, quiet=True,
        )
        assert not any(c.startswith("fakevec_") for c in chunks_df.columns)

    def test_model_carries_pooled_features(self):
        model = FakeEmbedModel(device="cpu")
        score_text(TWO_CHUNKS, [model], quiet=True)
        assert model.pooled_features_[0] == "fakevec_n_pooled"
        assert "fakevec_000" in model.pooled_features_

    def test_oov_words_drop_out_of_the_mean(self):
        _, chunks_df = score_text(
            ["cat oov cat"], [FakeEmbedModel(device="cpu")], quiet=True
        )
        # only the two "cat" tokens carry a vector; len("cat") == 3
        assert chunks_df["fakevec_000"].iloc[0] == pytest.approx(3.0)
        assert chunks_df["fakevec_n_pooled"].iloc[0] == 2
        assert chunks_df["n_words"].iloc[0] == 3


class TestSidecar:
    def test_pooling_is_recorded(self):
        meta = MetadataBuilder()
        meta.add_model(
            "word2vec",
            [f"word2vec_{i:03d}" for i in range(300)],
            1.0,
            level="word",
            pooled_features=["word2vec_n_pooled"]
            + [f"word2vec_{i:03d}" for i in range(300)],
        )
        entry = meta.build()["models"]["word2vec"]
        assert entry["features"]["level"] == "word"
        pooling = entry["chunk_pooling"]
        assert pooling["stat"] == "mean"
        assert pooling["nan_policy"] == "omit"
        assert pooling["count_column"] == "word2vec_n_pooled"
        assert pooling["features"]["count"] == 300
        assert pooling["features"]["level"] == "chunk"

    def test_absent_when_not_pooled(self):
        meta = MetadataBuilder()
        meta.add_model("sentiment", ["sentiment_pos"], 1.0, level="chunk")
        assert "chunk_pooling" not in meta.build()["models"]["sentiment"]
