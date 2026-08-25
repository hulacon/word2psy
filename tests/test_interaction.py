"""Interaction model tests — analytic closed-list rates, fully offline
(needs only the small nltk punkt data, the readability-tier footprint)."""

import math

import pytest


DIALOGUE = "Hey, are you okay? I saw your mother yesterday."
# tokens: hey are you okay i saw your mother yesterday  -> 9 word tokens
# sentences: "Hey, are you okay?" (question), "I saw your mother yesterday."


@pytest.fixture(scope="module")
def model():
    from word2psy.models.interaction import InteractionModel

    m = InteractionModel(device="cpu")
    m.load()
    return m


class TestInteraction:
    def test_feature_names(self, model):
        scores = model.predict(DIALOGUE)
        assert len(scores) == 7
        assert all(k.startswith("interaction_") for k in scores)

    def test_hand_computed_rates(self, model):
        s = model.predict(DIALOGUE)
        assert s["interaction_second_person"] == pytest.approx(2 / 9)  # you, your
        assert s["interaction_first_person_sing"] == pytest.approx(1 / 9)  # i
        assert s["interaction_first_person_plur"] == 0.0
        assert s["interaction_third_person"] == 0.0
        assert s["interaction_person_noun"] == pytest.approx(1 / 9)  # mother
        assert s["interaction_discourse_marker"] == pytest.approx(2 / 9)  # hey, okay
        assert s["interaction_question"] == pytest.approx(1 / 2)

    def test_third_person_excludes_it(self, model):
        s = model.predict("She gave him the box and it broke.")
        # she, him -> 2 of 8 tokens; "it" is not a person reference
        assert s["interaction_third_person"] == pytest.approx(2 / 8)

    def test_non_social_text_is_all_zero(self, model):
        s = model.predict("The red car drove down the empty road.")
        zeros = [k for k in s if k != "interaction_question"]
        assert all(s[k] == 0.0 for k in zeros)
        assert s["interaction_question"] == 0.0

    def test_empty_chunk_is_nan_not_error(self, model):
        s = model.predict("")
        assert all(math.isnan(v) for v in s.values())

    def test_punctuation_only_chunk_is_nan(self, model):
        s = model.predict("...")
        assert all(math.isnan(v) for v in s.values())

    def test_single_word_chunk_is_degenerate_but_defined(self, model):
        # the twp1000/word case: a bare pronoun scores rate 1.0
        s = model.predict("you")
        assert s["interaction_second_person"] == 1.0
        assert s["interaction_first_person_sing"] == 0.0

    def test_case_insensitive(self, model):
        assert (
            model.predict("YOU")["interaction_second_person"]
            == model.predict("you")["interaction_second_person"]
            == 1.0
        )

    def test_registered_and_analytic(self):
        from word2psy.cli import MODEL_REGISTRY
        from word2psy.metadata import get_model_checkpoint

        assert "interaction" in MODEL_REGISTRY
        assert get_model_checkpoint("interaction") is None
