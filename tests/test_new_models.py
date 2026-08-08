"""Integration tests for fasttext, word2vec, and gpt2_surprisal models.

Require model downloads (fastText ~7 GB cached, word2vec ~1.7 GB,
GPT-2 ~550 MB).
"""

import math

import numpy as np
import pytest

from word2psy.pipeline import score_text


class TestFastTextModel:
    @pytest.fixture(scope="class")
    def model(self):
        from word2psy.models.fasttext_embed import FastTextModel

        m = FastTextModel(device="cpu")
        m.load()
        return m

    def test_shape_and_naming(self, model):
        scores = model.predict("dog")
        assert len(scores) == 300
        assert "fasttext_000" in scores and "fasttext_299" in scores

    def test_no_oov(self, model):
        scores = model.predict("floofdoggo")  # not a real word
        vec = np.array(list(scores.values()))
        assert np.isfinite(vec).all()
        assert np.linalg.norm(vec) > 0

    def test_similar_words_closer(self, model):
        def vec(w):
            v = np.array(list(model.predict(w).values()))
            return v / np.linalg.norm(v)

        assert vec("dog") @ vec("cat") > vec("dog") @ vec("algebra")


class TestWord2VecModel:
    @pytest.fixture(scope="class")
    def model(self):
        from word2psy.models.word2vec import Word2VecModel

        m = Word2VecModel(device="cpu")
        m.load()
        return m

    def test_shape_and_naming(self, model):
        scores = model.predict("dog")
        assert len(scores) == 300
        assert "word2vec_000" in scores

    def test_oov_gives_nan(self, model):
        scores = model.predict("floofdoggo")
        assert all(math.isnan(v) for v in scores.values())

    def test_similar_words_closer(self, model):
        def vec(w):
            v = np.array(list(model.predict(w).values()))
            return v / np.linalg.norm(v)

        assert vec("dog") @ vec("cat") > vec("dog") @ vec("algebra")


class TestGPT2Surprisal:
    @pytest.fixture(scope="class")
    def model(self):
        from word2psy.models.gpt2_surprisal import GPT2SurprisalModel

        m = GPT2SurprisalModel(device="cpu")
        m.load()
        return m

    def test_predictable_word_lower_surprisal(self, model):
        frame = "The capital of France is {}."
        words = ["The", "capital", "of", "France", "is"]
        paris = model.predict_context(frame.format("Paris"), words + ["Paris"])
        tokyo = model.predict_context(frame.format("Tokyo"), words + ["Tokyo"])
        assert paris[-1]["gpt2_surprisal"] < tokyo[-1]["gpt2_surprisal"]

    def test_all_words_scored(self, model):
        words = ["The", "dog", "chased", "the", "cat"]
        scores = model.predict_context("The dog chased the cat.", words)
        assert len(scores) == 5
        assert all(math.isfinite(s["gpt2_surprisal"]) for s in scores)

    def test_same_word_differs_by_context(self, model):
        words = ["the", "dog", "bit", "the", "postman"]
        scores = model.predict_context("the dog bit the postman", words)
        # Two instances of "the" in different positions: different surprisal
        assert scores[0]["gpt2_surprisal"] != scores[3]["gpt2_surprisal"]

    def test_pipeline_integration(self, model):
        words_df, chunks_df = score_text(
            "The dog chased the cat.", [model], quiet=True
        )
        assert "gpt2_surprisal" in words_df.columns
        assert words_df["gpt2_surprisal"].notna().all()
        assert "gpt2_surprisal" not in chunks_df.columns
