"""lexical_norms_experiential: the probe fit, the weights file, and the model.

Offline: a synthetic ratings spreadsheet (test_binder's layout) and a stub
fastText; the real ratings and the 7 GB vectors are never needed.
"""

import zlib

import numpy as np
import pytest

from word2psy.exceptions import NormDataError
from word2psy.metadata import label_problems, model_labels
from word2psy.norms import binder, experiential as ex

from .test_binder import _sheet


class StubFastText:
    """Deterministic 300-d vectors per word."""

    def get_word_vector(self, word):
        rng = np.random.default_rng(zlib.crc32(word.encode()))
        return rng.standard_normal(ex.EMBED_DIM).astype(np.float32)


@pytest.fixture
def ratings_file(tmp_path):
    rows = [(f"word{chr(97 + i // 26)}{chr(97 + i % 26)}", 1 + i % 3,
             ("Complexity",) if i % 3 else ()) for i in range(60)]
    p = tmp_path / "WordSet1_Ratings.xlsx"
    _sheet(rows).to_excel(p, index=False)
    return p


def test_fit_recovers_a_planted_map_and_folds_the_standardisation():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 20)) * rng.uniform(0.5, 5, 20) + rng.uniform(-3, 3, 20)
    w = rng.standard_normal((20, 2))
    Y = X @ w + 0.05 * rng.standard_normal((300, 2))
    fit = ex.fit_probes(X, Y)
    assert fit["W"].shape == (20, 2) and fit["b"].shape == (2,)
    pred = X @ fit["W"] + fit["b"]
    assert np.corrcoef(pred[:, 0], Y[:, 0])[0, 1] > 0.99
    # the folded map equals ridge on standardised X at the chosen alpha
    from sklearn.linear_model import Ridge
    mu, sd = X.mean(0), X.std(0)
    r = Ridge(alpha=fit["alpha"][1]).fit((X - mu) / sd, Y[:, 1])
    np.testing.assert_allclose(pred[:, 1], r.predict((X - mu) / sd), rtol=1e-6, atol=1e-8)


def test_each_column_trains_on_its_rated_rows():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((100, 5))
    Y = np.column_stack([X[:, 0], X[:, 1]])
    Y[:40, 1] = np.nan
    fit = ex.fit_probes(X, Y)
    assert fit["n_train"].tolist() == [100, 60]
    assert np.isfinite(fit["W"]).all()


def test_grouped_alpha_selection_runs_on_groups():
    rng = np.random.default_rng(2)
    g = np.repeat(np.arange(30), 4)
    X = rng.standard_normal((120, 6))
    Y = X[:, :1] + rng.standard_normal((120, 1))
    a = ex.fit_probes(X, Y, groups=g)
    assert a["n_train"][0] == 120 and a["alpha"][0] in ex.ALPHAS


def test_train_writes_valid_weights_for_a_subset(tmp_path, ratings_file):
    out = tmp_path / "w.weights"  # any name: never rewritten to .npz
    path = ex.train(ratings_file, attributes=["Pain", "lexical_norms_experiential_vision", "# skip", ""],
                    out=out, ft_model=StubFastText(), quiet=True)
    assert path == out and out.is_file()
    w = ex.read_weights(out)
    # spreadsheet order, whatever order they were named in
    assert w["columns"] == ("lexical_norms_experiential_vision", "lexical_norms_experiential_pain")
    assert w["W"].shape == (300, 2)
    import json
    meta = json.loads(str(w["trained_on"]))
    assert meta["ratings_sha256"] == binder.file_sha256(ratings_file)
    assert meta["n_words"] == 60


def test_complexity_trains_on_the_words_that_rated_it(tmp_path, ratings_file):
    w = ex.read_weights(ex.train(ratings_file, attributes=["complexity", "vision"], out=tmp_path / "w.npz",
                                 ft_model=StubFastText(), quiet=True))
    assert dict(zip(w["columns"], w["n_train"].tolist())) == {
        "lexical_norms_experiential_complexity": 20, "lexical_norms_experiential_vision": 60}


def test_unknown_attribute_is_refused(tmp_path, ratings_file):
    with pytest.raises(NormDataError, match="unknown Binder attribute"):
        ex.train(ratings_file, attributes=["telepathy"], out=tmp_path / "w.npz",
                 ft_model=StubFastText(), quiet=True)


def test_missing_weights_fail_loudly_naming_how_to_build(tmp_path, monkeypatch):
    monkeypatch.setenv(ex.WEIGHTS_ENV, str(tmp_path / "absent.npz"))
    with pytest.raises(NormDataError) as e:
        ex.read_weights(ex.weights_path())
    msg = str(e.value)
    assert "train_experiential.py" in msg and ex.WEIGHTS_ENV in msg and "Binder" in msg


def test_model_emits_exactly_the_weights_columns(tmp_path, ratings_file, monkeypatch):
    from word2psy.models.lexical_norms_experiential import LexicalNormsExperientialModel
    from word2psy.norms import train as train_mod

    out = ex.train(ratings_file, attributes=["pain", "vision", "fearful"], out=tmp_path / "w.npz",
                   ft_model=StubFastText(), quiet=True)
    monkeypatch.setenv(ex.WEIGHTS_ENV, str(out))
    monkeypatch.setattr(train_mod, "load_fasttext", lambda: StubFastText())
    m = LexicalNormsExperientialModel(device="cpu")
    m.load()
    rows = m.predict_batch(["Cat", "dog", "cat"])
    assert list(rows[0]) == ["lexical_norms_experiential_vision", "lexical_norms_experiential_pain",
                             "lexical_norms_experiential_fearful"]
    assert rows[0] == rows[2]  # case-folded, deduplicated
    w = ex.read_weights(out)
    vec = StubFastText().get_word_vector("dog").astype(np.float64)
    np.testing.assert_allclose(list(rows[1].values()), vec @ w["W"] + w["b"])
    assert m.predict("dog") == rows[1]  # exactly: a word's value does not depend on its batch
    many = m.predict_batch(["dog"] + [f"filler{i}" for i in range(50)])
    assert many[0] == rows[1]


def test_model_load_fails_before_fasttext_without_weights(tmp_path, monkeypatch):
    from word2psy.models.lexical_norms_experiential import LexicalNormsExperientialModel
    from word2psy.norms import train as train_mod

    monkeypatch.setenv(ex.WEIGHTS_ENV, str(tmp_path / "absent.npz"))
    monkeypatch.setattr(train_mod, "load_fasttext", lambda: pytest.fail("loaded fastText first"))
    with pytest.raises(NormDataError):
        LexicalNormsExperientialModel(device="cpu").load()


def test_sidecar_labels_a_subset_run():
    cols = ["lexical_norms_experiential_vision", "lexical_norms_experiential_pain"]
    aggs = [f"{c}_mean" for c in cols]
    got = model_labels("lexical_norms_experiential", cols, aggs)
    assert set(got["columns"]) == set(cols + aggs)
    assert label_problems(got, cols + aggs) == []
