"""Tests for --by-sentence chunking and word-feature aggregation.

Fully offline: a trivial fake model stands in for real word-level models.
"""

import numpy as np
import pandas as pd
import pytest

from word2psy.models.base import BaseModel
from word2psy.pipeline import aggregate_word_features, score_text
from word2psy.tokenize import split_by_sentence


class FakeWordModel(BaseModel):
    """Word-level model returning the word's character length."""

    name = "fake_wordlen"
    level = "word"

    def load(self):
        self.model = object()

    def predict(self, text):
        return {"fake_len": float(len(text))}

    def predict_batch(self, texts):
        return [self.predict(t) for t in texts]


TWO_CHUNKS = [
    "The cat sat. The dog ran away.",
    "One lonely sentence here.",
]


class TestSplitBySentence:
    def test_splits_and_labels(self):
        sentences, labels, origin = split_by_sentence(TWO_CHUNKS)
        assert sentences == [
            "The cat sat.", "The dog ran away.", "One lonely sentence here.",
        ]
        assert labels == ["chunk_0/s0", "chunk_0/s1", "chunk_1/s0"]
        assert origin == [0, 0, 1]

    def test_custom_labels(self):
        _, labels, _ = split_by_sentence(TWO_CHUNKS, ["A", "B"])
        assert labels == ["A/s0", "A/s1", "B/s0"]


class TestBySentenceScoring:
    def test_chunks_are_sentences(self):
        words_df, chunks_df = score_text(
            TWO_CHUNKS, [], by_sentence=True, quiet=True
        )
        assert len(chunks_df) == 3
        assert list(chunks_df["chunk_label"]) == [
            "chunk_0/s0", "chunk_0/s1", "chunk_1/s0",
        ]
        assert list(chunks_df["n_words"]) == [3, 4, 4]
        # words table chunk assignment follows sentences
        assert words_df.loc[words_df["word"] == "dog", "chunk_idx"].item() == 1

    def test_passthrough_expanded(self):
        passthrough = pd.DataFrame({"condition": ["x", "y"]})
        _, chunks_df = score_text(
            TWO_CHUNKS, [], by_sentence=True, passthrough=passthrough,
            quiet=True,
        )
        assert list(chunks_df["condition"]) == ["x", "x", "y"]

    def test_without_flag_unchanged(self):
        _, chunks_df = score_text(TWO_CHUNKS, [], quiet=True)
        assert len(chunks_df) == 2


class TestAggregateWordFeatures:
    def _tables(self):
        words_df = pd.DataFrame({
            "word_idx": range(4),
            "word": ["a", "bb", "ccc", "dddd"],
            "sentence_idx": [0, 0, 1, 1],
            "chunk_idx": [0, 0, 1, 1],
            "chunk_label": ["c0", "c0", "c1", "c1"],
            "onset": np.nan,
            "offset": np.nan,
            "feat": [1.0, 3.0, 5.0, np.nan],
            "fasttext_000": [0.1, 0.2, 0.3, 0.4],  # embedding: excluded
        })
        chunks_df = pd.DataFrame({
            "chunk_idx": [0, 1], "chunk_label": ["c0", "c1"], "n_words": [2, 2],
        })
        return words_df, chunks_df

    def test_stats_and_nan_handling(self):
        words_df, chunks_df = self._tables()
        added = aggregate_word_features(words_df, chunks_df)
        assert added == ["feat_mean", "feat_sd", "feat_min", "feat_max"]
        assert chunks_df["feat_mean"].tolist() == [2.0, 5.0]  # NaN skipped
        assert chunks_df["feat_min"].tolist() == [1.0, 5.0]
        assert chunks_df["feat_max"].tolist() == [3.0, 5.0]
        assert chunks_df["feat_sd"][0] == pytest.approx(np.sqrt(2))
        assert np.isnan(chunks_df["feat_sd"][1])  # single valid value, ddof=1

    def test_embeddings_excluded(self):
        words_df, chunks_df = self._tables()
        aggregate_word_features(words_df, chunks_df)
        assert "fasttext_000_mean" not in chunks_df.columns

    def test_no_features_noop(self):
        words_df, chunks_df = self._tables()
        words_df = words_df.drop(columns=["feat", "fasttext_000"])
        assert aggregate_word_features(words_df, chunks_df) == []

    def test_through_pipeline(self):
        _, chunks_df = score_text(
            TWO_CHUNKS, [FakeWordModel(device="cpu")],
            by_sentence=True, quiet=True,
        )
        assert "fake_len_mean" in chunks_df.columns
        # "One lonely sentence here." -> lengths 3, 6, 8, 4
        assert chunks_df["fake_len_mean"].iloc[2] == pytest.approx(21 / 4)
        assert chunks_df["fake_len_max"].iloc[2] == 8.0

    def test_pipeline_optout(self):
        _, chunks_df = score_text(
            TWO_CHUNKS, [FakeWordModel(device="cpu")],
            aggregate_words=False, quiet=True,
        )
        assert not any(c.endswith("_mean") for c in chunks_df.columns)
