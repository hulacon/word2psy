"""Offline tests for GPT-2 surprisal word/token alignment (no model needed)."""

import math

import numpy as np

from word2psy.models.gpt2_surprisal import (
    aggregate_word_logprobs,
    find_word_spans,
)


class TestFindWordSpans:
    def test_simple_sentence(self):
        text = "The dog barked."
        spans = find_word_spans(text, ["The", "dog", "barked"])
        assert spans == [(0, 3), (4, 7), (8, 14)]

    def test_repeated_words_advance(self):
        text = "the cat and the dog"
        spans = find_word_spans(text, ["the", "cat", "and", "the", "dog"])
        assert spans[0] == (0, 3)
        assert spans[3] == (12, 15)  # second "the", not the first

    def test_contraction_split(self):
        # nltk splits "don't" into "do" + "n't"
        text = "I don't know."
        spans = find_word_spans(text, ["I", "do", "n't", "know"])
        assert spans == [(0, 1), (2, 4), (4, 7), (8, 12)]

    def test_unfindable_token_is_none(self):
        # nltk turns straight quotes into `` and '' which aren't in the text
        text = '"Hello," she said.'
        spans = find_word_spans(text, ["``", "Hello", "she", "said"])
        assert spans[0] is None
        assert spans[1] == (1, 6)

    def test_case_insensitive_fallback(self):
        spans = find_word_spans("THE DOG", ["the", "dog"])
        assert spans == [(0, 3), (4, 7)]


class TestAggregateWordLogprobs:
    def test_one_token_per_word(self):
        text = "the dog"
        spans = [(0, 3), (4, 7)]
        offsets = [(0, 3), (3, 7)]  # GPT-2 style: " dog" includes the space
        lps = np.array([-1.0, -2.0])
        result = aggregate_word_logprobs(text, spans, offsets, lps)
        assert result == [-1.0, -2.0]

    def test_multi_token_word_sums(self):
        text = "unbelievable"
        spans = [(0, 12)]
        offsets = [(0, 2), (2, 9), (9, 12)]
        lps = np.array([-1.0, -2.0, -0.5])
        result = aggregate_word_logprobs(text, spans, offsets, lps)
        assert result == [-3.5]

    def test_none_span_gives_nan(self):
        text = "word"
        result = aggregate_word_logprobs(
            text, [None], [(0, 4)], np.array([-1.0])
        )
        assert math.isnan(result[0])

    def test_word_with_nan_token_gives_nan(self):
        text = "the dog"
        spans = [(0, 3), (4, 7)]
        offsets = [(0, 3), (3, 7)]
        lps = np.array([np.nan, -2.0])
        result = aggregate_word_logprobs(text, spans, offsets, lps)
        assert math.isnan(result[0])
        assert result[1] == -2.0

    def test_punctuation_token_not_assigned(self):
        # Dropped punctuation between words must not pollute word sums
        text = "dog, cat"
        spans = [(0, 3), (5, 8)]  # words only; "," was filtered out
        offsets = [(0, 3), (3, 4), (4, 8)]
        lps = np.array([-1.0, -5.0, -2.0])
        result = aggregate_word_logprobs(text, spans, offsets, lps)
        assert result == [-1.0, -2.0]
