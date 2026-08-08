"""Tests for readability, wordform, emotion, sentiment, and minilm models.

readability and wordform are near-offline (small nltk/cmudict data);
emotion, sentiment, and minilm download transformer weights (~1 GB total).
"""

import math

import numpy as np
import pytest


SIMPLE = "The cat sat on the mat. The dog ran fast."
COMPLEX = (
    "Notwithstanding the epistemological ramifications inherent in "
    "phenomenological hermeneutics, contemporary methodological "
    "paradigms necessitate interdisciplinary reconceptualization."
)


class TestReadability:
    @pytest.fixture(scope="class")
    def model(self):
        from word2psy.models.readability import ReadabilityModel

        m = ReadabilityModel(device="cpu")
        m.load()
        return m

    def test_feature_names(self, model):
        scores = model.predict(SIMPLE)
        assert len(scores) == 7
        assert all(k.startswith("readability_") for k in scores)

    def test_complex_text_higher_grade(self, model):
        simple = model.predict(SIMPLE)
        complex_ = model.predict(COMPLEX)
        assert (
            complex_["readability_flesch_kincaid_grade"]
            > simple["readability_flesch_kincaid_grade"]
        )
        assert (
            complex_["readability_flesch_ease"] < simple["readability_flesch_ease"]
        )


class TestWordform:
    @pytest.fixture(scope="class")
    def model(self):
        from word2psy.models.wordform import WordformModel

        m = WordformModel(device="cpu")
        m.load()
        return m

    def test_basic_counts(self, model):
        scores = model.predict("cat")
        assert scores["length"] == 3
        assert scores["n_syllables"] == 1
        assert scores["n_phonemes"] == 3  # K AE T

    def test_multisyllabic(self, model):
        scores = model.predict("banana")
        assert scores["n_syllables"] == 3

    def test_oov_phonemes_nan_syllables_estimated(self, model):
        scores = model.predict("floofdoggo")
        assert math.isnan(scores["n_phonemes"])
        assert scores["n_syllables"] >= 2

    def test_old20_dense_vs_sparse(self, model):
        # "cat" lives in a dense orthographic neighborhood; "xylophone" doesn't
        cat = model.predict("cat")["old20"]
        xylo = model.predict("xylophone")["old20"]
        assert cat < xylo
        assert cat >= 1.0  # self excluded


class TestEmotion:
    @pytest.fixture(scope="class")
    def model(self):
        from word2psy.models.emotion import EmotionModel

        m = EmotionModel(device="cpu")
        m.load()
        return m

    def test_28_labels(self, model):
        scores = model.predict("Hello there.")
        assert len(scores) == 28
        assert "emotion_joy" in scores and "emotion_neutral" in scores

    def test_valenced_text(self, model):
        happy = model.predict("I am overjoyed, this is the best day of my life!")
        angry = model.predict("I am furious, this is a disgusting betrayal.")
        assert happy["emotion_joy"] > angry["emotion_joy"]
        assert angry["emotion_anger"] > happy["emotion_anger"]


class TestSentiment:
    @pytest.fixture(scope="class")
    def model(self):
        from word2psy.models.sentiment import SentimentModel

        m = SentimentModel(device="cpu")
        m.load()
        return m

    def test_three_labels_sum_to_one(self, model):
        scores = model.predict("The weather is nice today.")
        assert set(scores) == {
            "sentiment_negative",
            "sentiment_neutral",
            "sentiment_positive",
        }
        assert abs(sum(scores.values()) - 1.0) < 1e-4

    def test_polarity(self, model):
        pos = model.predict("I absolutely love this, it's wonderful!")
        neg = model.predict("This is terrible, I hate it.")
        assert pos["sentiment_positive"] > 0.5
        assert neg["sentiment_negative"] > 0.5


class TestSentenceEmbed:
    @pytest.fixture(scope="class")
    def model(self):
        from word2psy.models.sentence_embed import SentenceEmbedModel

        m = SentenceEmbedModel(device="cpu")
        m.load()
        return m

    def test_shape_and_norm(self, model):
        scores = model.predict("A sentence about dogs.")
        assert len(scores) == 384
        vec = np.array(list(scores.values()))
        assert abs(np.linalg.norm(vec) - 1.0) < 0.01

    def test_similarity_structure(self, model):
        def vec(t):
            return np.array(list(model.predict(t).values()))

        dog1 = vec("The dog chased the ball.")
        dog2 = vec("A puppy played fetch.")
        tax = vec("Quarterly tax filings are due in April.")
        assert dog1 @ dog2 > dog1 @ tax


class TestExtendedNorms:
    def test_new_dimensions_present(self):
        from word2psy.models.lexical_norms import LexicalNormsModel

        m = LexicalNormsModel(device="cpu")
        m.load()
        scores = m.predict("conversation")
        for feat in (
            "familiarity",
            "semantic_size",
            "gender_association",
            "socialness",
            "body_object_interaction",
        ):
            assert feat in scores
        assert len(scores) == 23
        # "conversation" is social; "gravel" is not
        gravel = m.predict("gravel")
        assert scores["socialness"] > gravel["socialness"]
        # "hammer" affords more bodily interaction than "cloud"
        assert (
            m.predict("hammer")["body_object_interaction"]
            > m.predict("cloud")["body_object_interaction"]
        )
        m.unload()
