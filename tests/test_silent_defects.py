"""Regression tests for defects that a green default suite did not catch.

Each test pins one defect from the bug-hunting pass: something that either
produced wrong values without saying so, or swallowed a failure. All are
default tier: fake models, small fixtures, and a fake of the OpenCLIP
tokenizer contract (no weights).
"""

import sys
import time
import warnings

import numpy as np
import pandas as pd
import pytest

from word2psy.models.base import BaseModel


# --- CSV input: pandas NA parsing ------------------------------------------

def test_csv_na_like_words_are_kept_verbatim(tmp_path):
    """pandas' default NA parsing turned the stimulus words "null", "None",
    "NA", "nan" and "N/A" into NaN, which ``astype(str)`` then scored as the
    word "nan". A blank cell did the same."""
    from word2psy.cli import read_inputs

    src = tmp_path / "words.csv"
    src.write_text("word\nnull\nNone\nNA\nnan\nN/A\n\"\"\ncat\n")
    text, _, _ = read_inputs([src], text_column="word")
    assert text == ["null", "None", "NA", "nan", "N/A", "", "cat"]


def test_csv_id_labels_are_kept_verbatim(tmp_path):
    """IDs were parsed as numbers before ``astype(str)``: "01" and "1"
    collapsed onto the same label, and one blank ID turned every other
    label into a float ("2" -> "2.0")."""
    from word2psy.cli import read_inputs

    src = tmp_path / "stim.csv"
    src.write_text("id,word\n01,cat\n1,dog\n,owl\n2,emu\n")
    _, labels, _ = read_inputs([src], text_column="word", id_column="id")
    assert labels == ["01", "1", "", "2"]


# --- Multi-file text input: stimulus_id ------------------------------------

class _FakeWord(BaseModel):
    name = "fake_word"
    level = "word"

    def load(self):
        self.model = object()

    def predict(self, text):
        return {"fake_word_length": float(len(text))}


class _SlowWord(_FakeWord):
    name = "slow_word"

    def predict_batch(self, texts):
        time.sleep(1.0)
        return [{"slow_word_length": float(len(t))} for t in texts]


def _run_cli(monkeypatch, argv, classes):
    import word2psy.cli as cli

    for cls in classes.values():
        monkeypatch.setitem(cli.MODEL_REGISTRY, cls.name,
                            ("tests.test_silent_defects", cls.__name__, "fake"))
    monkeypatch.setattr(cli, "_load_model_class", lambda name: classes[name])
    monkeypatch.setattr(sys, "argv", ["word2psy", *argv])
    cli.main()


@pytest.mark.parametrize("extra", [[], ["--by-sentence"]])
def test_each_text_file_gets_its_own_stimulus_id(tmp_path, monkeypatch, extra):
    """With several .txt inputs every chunk was labelled with the FIRST
    file's stem, so distinct stimuli shared one stimulus_id."""
    a = tmp_path / "stim-01.txt"
    b = tmp_path / "stim-02.txt"
    a.write_text("One cat. Two dogs.")
    b.write_text("Three owls.")
    out = tmp_path / "scores.csv"
    _run_cli(monkeypatch,
             ["fake_word", str(a), str(b), "-o", str(out), "--quiet", *extra],
             {"fake_word": _FakeWord})
    chunks = pd.read_csv(tmp_path / "scores_chunks.csv")
    words = pd.read_csv(tmp_path / "scores_words.csv")
    assert chunks.columns[0] == "stimulus_id"
    expected = ["stim-01", "stim-01", "stim-02"] if extra else ["stim-01", "stim-02"]
    assert chunks["stimulus_id"].tolist() == expected
    by_chunk = dict(zip(chunks["chunk_idx"], chunks["stimulus_id"]))
    assert (words["stimulus_id"] == words["chunk_idx"].map(by_chunk)).all()


def test_constant_stimulus_id_overrides_per_file_ids(tmp_path, monkeypatch):
    a = tmp_path / "stim-01.txt"
    b = tmp_path / "stim-02.txt"
    a.write_text("One cat.")
    b.write_text("Three owls.")
    out = tmp_path / "scores.csv"
    _run_cli(monkeypatch,
             ["fake_word", str(a), str(b), "-o", str(out), "--quiet",
              "--stimulus-id", "run-01"],
             {"fake_word": _FakeWord})
    chunks = pd.read_csv(tmp_path / "scores_chunks.csv")
    assert chunks["stimulus_id"].tolist() == ["run-01", "run-01"]


# --- Sidecar runtime_sec ----------------------------------------------------

def test_sidecar_runtime_is_per_model(tmp_path, monkeypatch):
    """Every model's runtime_sec was total_time / n_models -- the same
    number for all, whatever each actually took."""
    import json

    src = tmp_path / "in.txt"
    src.write_text("One cat.")
    out = tmp_path / "scores.csv"
    _run_cli(monkeypatch,
             ["fake_word", "slow_word", str(src), "-o", str(out), "--quiet"],
             {"fake_word": _FakeWord, "slow_word": _SlowWord})
    meta = json.loads((tmp_path / "scores.meta.json").read_text())
    fast = meta["models"]["fake_word"]["runtime_sec"]
    slow = meta["models"]["slow_word"]["runtime_sec"]
    # The old code gave both the same value (total / 2).
    assert slow >= 1.0
    assert slow - fast >= 0.5


# --- by_sentence drops empty chunks ----------------------------------------

def test_by_sentence_keeps_a_chunk_with_no_sentences():
    """A chunk with no sentences (empty text) vanished from the output under
    --by-sentence, taking its passthrough row with it."""
    from word2psy.pipeline import score_text

    pt = pd.DataFrame({"stimulus_id": ["stim-01", "stim-02"]})
    _, chunks = score_text(["One cat. Two dogs.", ""], [_FakeWord()],
                           passthrough=pt, by_sentence=True, quiet=True)
    assert chunks["stimulus_id"].tolist() == ["stim-01", "stim-01", "stim-02"]
    assert chunks["chunk_label"].tolist() == ["chunk_0/s0", "chunk_0/s1", "chunk_1/s0"]
    assert chunks["n_words"].tolist() == [2, 2, 0]


# --- readability on a chunk with no words ----------------------------------

@pytest.mark.parametrize("text", ["", "...", "   "])
def test_readability_is_nan_without_words(text):
    """textstat returns 0.0 for every metric on text with no words, which
    read as a real (and extreme) grade level."""
    from word2psy.models.readability import ReadabilityModel

    m = ReadabilityModel()
    m.load()
    scores = m.predict(text)
    assert all(np.isnan(v) for v in scores.values())
    assert set(scores) <= set(ReadabilityModel.nulls)


def test_readability_unchanged_on_real_text():
    from word2psy.models.readability import ReadabilityModel

    m = ReadabilityModel()
    m.load()
    assert all(np.isfinite(v) for v in m.predict("The cat sat on the mat.").values())


# --- CLIP truncation warning was unreachable ---------------------------------

class _FakeClip:
    def encode_text(self, tokens):
        import torch
        return torch.ones(tokens.shape[0], 4)


class _FakeClipTokenizer:
    """Mimics open_clip's tokenizer contract: ``__call__`` always returns
    exactly ``context_length`` columns (pad or truncate), ``encode`` returns
    the raw BPE ids. Faked because importing open_clip pulls in torchvision,
    which the CPU-torch CI install cannot import; the real tokenizer's
    shape was checked by hand (``(n, 77)`` for a 200-word text)."""

    context_length = 77

    def encode(self, text):
        return list(range(len(text.split())))

    def __call__(self, texts):
        import torch
        return torch.zeros(len(texts), self.context_length, dtype=torch.long)


def _clip_model():
    from word2psy.models.clip_text import CLIPTextModel

    m = CLIPTextModel(device="cpu")
    m._tokenizer = _FakeClipTokenizer()
    m.model = _FakeClip()
    return m


def test_clip_warns_when_a_chunk_is_truncated(caplog):
    """open_clip pads/truncates to exactly 77 tokens, so the old
    ``tokens.shape[1] > 77`` check could never fire."""
    m = _clip_model()
    with caplog.at_level("WARNING", logger="word2psy.models.clip_text"):
        m.predict_batch(["word " * 200, "a cat"])
    assert any("77" in r.getMessage() for r in caplog.records)


def test_clip_does_not_warn_on_short_chunks(caplog):
    m = _clip_model()
    with caplog.at_level("WARNING", logger="word2psy.models.clip_text"):
        m.predict_batch(["a cat", "word " * 70])
    assert not caplog.records


def test_clip_tokenizer_without_encode_still_scores():
    """HF/SigLIP OpenCLIP tokenizers have no encode(); skip the check."""

    class NoEncode:
        context_length = 77

        def __call__(self, texts):
            import torch
            return torch.zeros(len(texts), 77, dtype=torch.long)

    m = _clip_model()
    m._tokenizer = NoEncode()
    assert len(m.predict_batch(["a cat", "word " * 200])) == 2


# --- dashboard swallowed every projection error ------------------------------

def _payload(monkeypatch, exc):
    from word2psy.viz import dashboard

    def boom(*a, **k):
        raise exc

    monkeypatch.setattr(dashboard, "compute_projection", boom)
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(10, 3)),
                      columns=["a", "b", "c"])
    return dashboard._projection_payload(df, ["a", "b", "c"], "mds", 2)


def test_dashboard_projection_code_errors_propagate(monkeypatch):
    """``except Exception`` turned a broken projection (e.g. a sklearn API
    change) into a warning and a dashboard silently missing its views."""
    with pytest.raises(TypeError):
        _payload(monkeypatch, TypeError("unexpected keyword argument"))


def test_dashboard_degenerate_input_still_degrades(monkeypatch):
    with pytest.warns(UserWarning, match="projection failed"):
        assert _payload(monkeypatch, ValueError("degenerate")) is None


# --- compute_projection mutated its input ------------------------------------

def test_compute_projection_does_not_mutate_input():
    """NaN imputation wrote column means into the caller's array."""
    from word2psy.viz.projection import compute_projection

    X = np.random.default_rng(0).normal(size=(10, 3))
    X[0, 1] = np.nan
    before = X.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        compute_projection(X, method="pca")
    np.testing.assert_array_equal(X, before)


# --- crossmodal: duplicate image labels ---------------------------------------

def test_crossmodal_duplicate_filenames_fall_back_to_a_unique_id():
    """Two images with the same basename in different folders collapsed onto
    one similarity-matrix column name."""
    from word2psy.crossmodal import cross_modal_similarity

    dim = 4
    text = pd.DataFrame({"chunk_label": ["cat"],
                         **{f"clip_text_{i:03d}": [1.0] for i in range(dim)}})
    image = pd.DataFrame({
        "filename": ["img.png", "img.png"],
        "filepath": ["sub-01/img.png", "sub-02/img.png"],
        **{f"clip_{i:03d}": [1.0, 0.5] for i in range(dim)},
    })
    sim = cross_modal_similarity(text, image)
    assert list(sim.columns) == ["sub-01/img.png", "sub-02/img.png"]


def test_crossmodal_unique_filenames_unchanged():
    from word2psy.crossmodal import cross_modal_similarity

    dim = 4
    text = pd.DataFrame({"chunk_label": ["cat"],
                         **{f"clip_text_{i:03d}": [1.0] for i in range(dim)}})
    image = pd.DataFrame({
        "filename": ["a.png", "b.png"],
        "filepath": ["x/a.png", "x/b.png"],
        **{f"clip_{i:03d}": [1.0, 0.5] for i in range(dim)},
    })
    assert list(cross_modal_similarity(text, image).columns) == ["a.png", "b.png"]


# --- feature_config: count column inside an embedding group -------------------

def test_pooled_count_column_is_not_an_embedding_dimension():
    """``word2vec_*`` also matched ``word2vec_n_pooled`` (a word count) in a
    chunks table, mixing it into the embedding group."""
    from word2psy.viz.feature_config import get_model_columns

    cols = ["chunk_idx", "word2vec_n_pooled", "word2vec_000", "word2vec_001",
            "fasttext_n_pooled", "fasttext_000", "fasttext_001"]
    assert get_model_columns(cols, "word2vec") == ["word2vec_000", "word2vec_001"]
    assert get_model_columns(cols, "fasttext") == ["fasttext_000", "fasttext_001"]
