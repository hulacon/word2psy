"""Tests for the lexical norms model.

These tests require norm databases and the fastText model to be downloaded
(~7GB on first run). Mark as slow/integration tests if needed.
"""

import pytest

from word2psy.models.lexical_norms import LexicalNormsModel

EXPECTED_FEATURES = [
    "concreteness",
    "valence",
    "arousal",
    "dominance",
    "age_of_acquisition",
    "imageability",
    "sensorimotor_touch",
    "sensorimotor_hearing",
    "sensorimotor_smell",
    "sensorimotor_taste",
    "sensorimotor_vision",
    "sensorimotor_interoception",
    "sensorimotor_mouth",
    "sensorimotor_hand",
    "sensorimotor_foot",
    "sensorimotor_head",
    "sensorimotor_torso",
    "zipf_frequency",
]


@pytest.fixture(scope="module")
def norms_model():
    """Load the lexical norms model once for all tests in this module."""
    model = LexicalNormsModel(device="cpu")
    model.load()
    return model


class TestLexicalNormsModel:
    def test_attributes(self):
        model = LexicalNormsModel()
        assert model.name == "lexical_norms"
        assert model.level == "word"

    def test_predict_returns_all_features(self, norms_model):
        scores = norms_model.predict("dog")
        for feat in EXPECTED_FEATURES:
            assert feat in scores, f"Missing feature: {feat}"

    def test_predict_values_are_numeric(self, norms_model):
        scores = norms_model.predict("cat")
        for name, val in scores.items():
            assert isinstance(val, float), f"{name} is not float: {type(val)}"

    def test_concrete_word_has_high_concreteness(self, norms_model):
        scores = norms_model.predict("table")
        # Table is highly concrete; score should be above midpoint (2.5 on 1-5 scale)
        assert scores["concreteness"] > 2.5

    def test_abstract_word_has_lower_concreteness(self, norms_model):
        concrete = norms_model.predict("hammer")["concreteness"]
        abstract = norms_model.predict("freedom")["concreteness"]
        assert concrete > abstract

    def test_frequency_common_vs_rare(self, norms_model):
        common = norms_model.predict("the")["zipf_frequency"]
        rare = norms_model.predict("defenestrate")["zipf_frequency"]
        assert common > rare

    def test_predict_batch(self, norms_model):
        words = ["dog", "cat", "liberty", "chair"]
        results = norms_model.predict_batch(words)
        assert len(results) == 4
        for r in results:
            assert len(r) == len(EXPECTED_FEATURES)

    def test_batch_deduplication(self, norms_model):
        words = ["dog", "dog", "cat", "dog"]
        results = norms_model.predict_batch(words)
        assert len(results) == 4
        # Same word should get same scores
        assert results[0] == results[1]
        assert results[0] == results[3]

    def test_oov_word_still_works(self, norms_model):
        # Nonsense word — fastText should still produce an embedding via subword
        scores = norms_model.predict("xyzzyplugh")
        assert all(isinstance(v, float) for v in scores.values())
