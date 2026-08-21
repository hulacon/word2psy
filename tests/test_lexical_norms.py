"""Tests for the lexical norms model.

These tests require norm databases and the fastText model to be downloaded
(~7GB on first run). Mark as slow/integration tests if needed.
"""

import pytest

from word2psy.models.lexical_norms import LexicalNormsModel

EXPECTED_FEATURES = [
    "lexical_norms_concreteness",
    "lexical_norms_valence",
    "lexical_norms_arousal",
    "lexical_norms_dominance",
    "lexical_norms_age_of_acquisition",
    "lexical_norms_imageability",
    "lexical_norms_sensorimotor_touch",
    "lexical_norms_sensorimotor_hearing",
    "lexical_norms_sensorimotor_smell",
    "lexical_norms_sensorimotor_taste",
    "lexical_norms_sensorimotor_vision",
    "lexical_norms_sensorimotor_interoception",
    "lexical_norms_sensorimotor_mouth",
    "lexical_norms_sensorimotor_hand",
    "lexical_norms_sensorimotor_foot",
    "lexical_norms_sensorimotor_head",
    "lexical_norms_sensorimotor_torso",
    "lexical_norms_familiarity",
    "lexical_norms_semantic_size",
    "lexical_norms_gender_association",
    "lexical_norms_socialness",
    "lexical_norms_body_object_interaction",
    "lexical_norms_zipf_frequency",
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
        assert scores["lexical_norms_concreteness"] > 2.5

    def test_abstract_word_has_lower_concreteness(self, norms_model):
        concrete = norms_model.predict("hammer")["lexical_norms_concreteness"]
        abstract = norms_model.predict("freedom")["lexical_norms_concreteness"]
        assert concrete > abstract

    def test_frequency_common_vs_rare(self, norms_model):
        common = norms_model.predict("the")["lexical_norms_zipf_frequency"]
        rare = norms_model.predict("defenestrate")["lexical_norms_zipf_frequency"]
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
