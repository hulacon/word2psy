"""Tests for text tokenization and DataFrame construction."""

import math

import pandas as pd
import pytest

from word2psy.tokenize import tokenize_text


class TestTokenizeSingleChunk:
    def test_basic_sentence(self):
        df = tokenize_text("The cat sat on the mat.")
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == [
            "word_idx",
            "word",
            "sentence_idx",
            "chunk_idx",
            "chunk_label",
            "onset",
            "offset",
        ]
        # "The cat sat on the mat" = 6 words (punctuation stripped by default)
        assert len(df) == 6
        assert df["word"].tolist() == ["The", "cat", "sat", "on", "the", "mat"]

    def test_word_idx_sequential(self):
        df = tokenize_text("Hello world. Goodbye world.")
        assert df["word_idx"].tolist() == list(range(len(df)))

    def test_sentence_idx(self):
        df = tokenize_text("First sentence. Second sentence.")
        assert df["sentence_idx"].nunique() == 2
        # Words in the first sentence should have sentence_idx=0
        first_words = df[df["sentence_idx"] == 0]["word"].tolist()
        assert "First" in first_words

    def test_single_chunk_default_label(self):
        df = tokenize_text("Hello world.")
        assert (df["chunk_idx"] == 0).all()
        assert (df["chunk_label"] == "chunk_0").all()

    def test_onset_offset_nan(self):
        df = tokenize_text("Hello world.")
        assert df["onset"].isna().all()
        assert df["offset"].isna().all()


class TestTokenizeMultipleChunks:
    def test_list_input(self):
        df = tokenize_text(["First chunk.", "Second chunk."])
        assert df["chunk_idx"].nunique() == 2
        chunk_0 = df[df["chunk_idx"] == 0]["word"].tolist()
        chunk_1 = df[df["chunk_idx"] == 1]["word"].tolist()
        assert "First" in chunk_0
        assert "Second" in chunk_1

    def test_custom_labels(self):
        df = tokenize_text(
            ["Scene one.", "Scene two."],
            chunk_labels=["intro", "body"],
        )
        assert df[df["chunk_idx"] == 0]["chunk_label"].iloc[0] == "intro"
        assert df[df["chunk_idx"] == 1]["chunk_label"].iloc[0] == "body"

    def test_label_length_mismatch(self):
        with pytest.raises(ValueError, match="chunk_labels length"):
            tokenize_text(["a", "b"], chunk_labels=["only_one"])


class TestTokenizePunctuation:
    def test_punctuation_stripped_by_default(self):
        df = tokenize_text("Hello, world!")
        words = df["word"].tolist()
        assert "," not in words
        assert "!" not in words

    def test_keep_punctuation(self):
        df = tokenize_text("Hello, world!", keep_punctuation=True)
        words = df["word"].tolist()
        assert "," in words
        assert "!" in words


class TestTokenizeEdgeCases:
    def test_empty_string(self):
        df = tokenize_text("")
        assert len(df) == 0
        assert list(df.columns) == [
            "word_idx",
            "word",
            "sentence_idx",
            "chunk_idx",
            "chunk_label",
            "onset",
            "offset",
        ]

    def test_single_word(self):
        df = tokenize_text("hello")
        assert len(df) == 1
        assert df["word"].iloc[0] == "hello"

    def test_contractions(self):
        df = tokenize_text("I can't believe it's done.")
        words = df["word"].tolist()
        # nltk tokenizes "can't" as ["ca", "n't"] and "it's" as ["it", "'s"]
        assert "ca" in words or "can't" in words
