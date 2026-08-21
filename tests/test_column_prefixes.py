"""Contract B §4.1: every feature column starts with its model's prefix.

The rule exists so a consumer can attribute a column to the model that made
it -- psytwill maps columns to models by prefix and nothing else. Two models
(`wordform`, `lexical_norms`) shipped through 0.3.1 emitting bare names
(`length`, `valence`, `zipf_frequency`), which psytwill could not attribute
to anything. It went unnoticed because every extraction run before 2026-08-20
used only embedding models, whose columns are prefixed by construction.

`stimfeat_preflight.py` did not catch it either: it checks that the *declared*
prefix namespace is collision-free, never that an emitted column lands in it.
These tests close that gap on the producer side, with stubs rather than the
~7 GB of weights the real models want.
"""

import pytest

from word2psy.cli import MODEL_REGISTRY

# psytwill's reserved non-feature columns (§4.1). Never prefixed.
RESERVED = {
    "stimulus_id", "filename", "filepath", "image_idx", "time", "onset",
    "offset", "chunk_idx", "chunk_label", "n_words", "word", "word_idx",
    "sentence_idx", "voice", "speaker", "turn_idx",
}


def assert_prefixed(model_name: str, keys) -> None:
    """Every non-reserved key equals the model name or starts with `name_`."""
    bad = [
        k for k in keys
        if k not in RESERVED
        and k != model_name
        and not k.startswith(f"{model_name}_")
    ]
    assert not bad, (
        f"{model_name} emits unprefixed feature columns {bad}. Contract B §4.1 "
        f"requires every feature column to start with the model's declared "
        f"prefix. Fix: rename them to {model_name}_<name> in the model's "
        f"predict/predict_batch return dict."
    )


class TestLexicalNormsPrefixes:
    """Stub the regressors and fastText so no weights are downloaded."""

    @pytest.fixture
    def stubbed(self):
        import numpy as np
        from word2psy.models.lexical_norms import LexicalNormsModel
        from word2psy.norms.train import NORM_DIMENSIONS

        class FakeRegressor:
            def predict(self, vecs):
                return np.zeros(len(vecs), dtype=np.float32)

        class FakeFastText:
            def get_word_vector(self, w):
                return np.zeros(300, dtype=np.float32)

        m = LexicalNormsModel(device="cpu")
        m._regressors = {k: FakeRegressor() for k in NORM_DIMENSIONS}
        m._ft_model = FakeFastText()
        return m

    def test_predict_is_prefixed(self, stubbed):
        assert_prefixed("lexical_norms", stubbed.predict("hammer"))

    def test_predict_batch_is_prefixed(self, stubbed):
        rows = stubbed.predict_batch(["hammer", "freedom"])
        assert len(rows) == 2
        for row in rows:
            assert_prefixed("lexical_norms", row)

    def test_every_norm_dimension_survives(self, stubbed):
        """The prefix is added, not substituted for, the dimension name."""
        from word2psy.norms.train import NORM_DIMENSIONS

        keys = set(stubbed.predict("hammer"))
        missing = [d for d in NORM_DIMENSIONS
                   if f"lexical_norms_{d}" not in keys]
        assert not missing, f"dimensions lost in the rename: {missing}"
        assert "lexical_norms_zipf_frequency" in keys

    def test_mutating_the_prefix_off_fails(self, stubbed):
        """The assertion has teeth: a bare column must be caught."""
        row = dict(stubbed.predict("hammer"))
        row["valence"] = 0.0
        with pytest.raises(AssertionError, match="unprefixed feature columns"):
            assert_prefixed("lexical_norms", row)


class TestWordformPrefixes:
    def test_predict_is_prefixed(self):
        pytest.importorskip("nltk")
        from word2psy.models.wordform import WordformModel

        m = WordformModel(device="cpu")
        try:
            m.load()
        except Exception as exc:  # cmudict not cached in this environment
            pytest.skip(f"wordform weights unavailable: {exc}")
        assert_prefixed("wordform", m.predict("cat"))
        assert_prefixed("wordform", m.predict_batch(["cat", "banana"])[0])


def test_registry_names_are_valid_prefixes():
    """A prefix has to be a legal §4.1 identifier to be usable as one."""
    import re

    bad = [n for n in MODEL_REGISTRY if not re.fullmatch(r"[a-z][a-z0-9_]*", n)]
    assert not bad, f"registry names unusable as column prefixes: {bad}"
