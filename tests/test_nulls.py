"""Contract B §4.1 (schema 1.1) producer duty: every NaN word2psy can emit is declared.

Offline fixtures (nltk punkt/cmudict only) cover the word, chunk and
aggregate conditions: a CMUdict miss, a one-word chunk, a chunk with no word
tokens, and a chunk whose every word is null. Each asserts both directions:
NaN columns are a subset of the sidecar's `nulls`, and every key names an
inventoried column. The model-level rules for weighted models (word2vec OOV,
gpt2 alignment) are checked on real sidecars by `word2psy sidecar refresh`.
"""

import json

import numpy as np
import pandas as pd
import pytest

from word2psy.cli import MODEL_REGISTRY, _load_model_class
from word2psy.metadata import SCHEMA_VERSION, MetadataBuilder, declared_nulls, model_nulls
from word2psy.pipeline import score_text
from word2psy.sidecar import refresh_sidecar

KINDS = {"undefined", "undefinable", "missing"}

# chunk 0: ordinary; 1: one out-of-CMUdict word; 2: punctuation only (no word
# tokens); 3: one ordinary word
CHUNKS = ["Hello there, do you see me?", "Zyxqvwpt", "...", "Cat"]


@pytest.mark.parametrize("model", sorted(MODEL_REGISTRY))
def test_every_model_declares_well_formed_nulls(model):
    for col, entry in declared_nulls(model).items():
        assert set(entry) == {"means", "when"}, col
        assert entry["means"] in KINDS, col
        assert entry["when"].strip(), col


@pytest.fixture(scope="module")
def scored():
    models = [_load_model_class(n)() for n in ("wordform", "interaction")]
    words, chunks = score_text(CHUNKS, models, quiet=True)
    b = MetadataBuilder()
    for m in models:
        b.add_model(m.name, m.feature_names_, 1.0, level=m.level,
                    aggregate_features=getattr(m, "aggregate_features_", None))
    return words, chunks, b.build()


def _nan_cols(df):
    num = df.select_dtypes(include="number")
    return {c for c in num.columns if num[c].isna().any()}


def test_all_nans_declared_and_every_key_inventoried(scored):
    words, chunks, meta = scored
    assert meta["schema_version"] == SCHEMA_VERSION == "1.1"
    declared, inventoried = set(), set()
    for entry in meta["models"].values():
        declared |= set(entry["nulls"])
        inventoried |= set(entry["features"]["columns"])
        inventoried |= set(entry.get("chunk_aggregates", {}).get("columns", []))
    reserved = {"onset", "offset"}  # untimed text input
    assert _nan_cols(words) - reserved <= declared
    assert _nan_cols(chunks) - reserved <= declared
    assert declared <= inventoried


def test_aggregate_kinds_follow_the_decided_rule(scored):
    words, chunks, meta = scored
    nulls = meta["models"]["wordform"]["nulls"]
    # one-word chunks (1 and 3): sd is undefined
    assert chunks.loc[[1, 3], "wordform_length_sd"].isna().all()
    assert nulls["wordform_length_sd"]["means"] == "undefined"
    # the CMUdict miss: word-level missing, and the chunk mean inherits it
    assert np.isnan(words.loc[words.word == "zyxqvwpt", "wordform_n_phonemes"]).all()
    assert np.isnan(chunks.loc[1, "wordform_n_phonemes_mean"])
    assert nulls["wordform_n_phonemes"]["means"] == "missing"
    assert nulls["wordform_n_phonemes_mean"]["means"] == "missing"
    # a never-null word feature: its mean is null only on an empty chunk
    assert nulls["wordform_length_mean"]["means"] == "undefined"
    # the chunk with no word tokens: interaction is undefined throughout
    assert chunks.loc[2, [c for c in chunks if c.startswith("interaction_")]].isna().all()
    assert meta["models"]["interaction"]["nulls"]["interaction_question"]["means"] == "undefined"


def test_pooled_vector_on_an_empty_chunk_is_undefined():
    # fastText never NaNs per word; its pooled chunk vector does on a wordless chunk
    got = model_nulls("fasttext", ["fasttext_000"], pooled_columns=["fasttext_000"])
    assert got["fasttext_000"]["means"] == "undefined"
    # word2vec's own OOV entry (missing) dominates, per one-column-one-kind
    assert model_nulls("word2vec", ["word2vec_000"],
                       pooled_columns=["word2vec_000"])["word2vec_000"]["means"] == "missing"


def test_model_nulls_filters_to_emitted_columns():
    assert model_nulls("wordform", ["wordform_length"]) == {}
    assert set(model_nulls("wordform", ["wordform_n_phonemes"], ["wordform_n_phonemes_sd"])) == {
        "wordform_n_phonemes", "wordform_n_phonemes_sd"}


class TestRefresh:
    def _family_1_0(self, tmp_path, scored):
        words, chunks, meta = scored
        meta = json.loads(json.dumps(meta))
        for kind, df in (("words", words), ("chunks", chunks)):
            path = tmp_path / f"fam_{kind}.csv"
            df.to_csv(path, index=False)
            meta["output"][kind] = {"path": str(path), "rows": len(df), "columns": df.shape[1]}
        meta["schema_version"] = "1.0"
        for e in meta["models"].values():  # a 1.0 sidecar: no nulls, no aggregate inventory
            e.pop("nulls")
            e.pop("chunk_aggregates", None)
        side = tmp_path / "fam.meta.json"
        side.write_text(json.dumps(meta, indent=2))
        return side

    def test_refresh_rebuilds_inventory_and_nulls(self, tmp_path, scored):
        side = self._family_1_0(tmp_path, scored)
        csvs = {p: p.read_bytes() for p in tmp_path.glob("*.csv")}
        assert refresh_sidecar(side).status == "refreshed"
        meta = json.loads(side.read_text())
        wf = meta["models"]["wordform"]
        assert "wordform_n_phonemes_sd" in wf["chunk_aggregates"]["columns"]
        assert wf["nulls"] == scored[2]["models"]["wordform"]["nulls"]  # same as a fresh run
        assert all(p.read_bytes() == b for p, b in csvs.items())
        assert refresh_sidecar(side).status == "unchanged"

    def test_refresh_refuses_an_undeclared_nan(self, tmp_path, scored):
        side = self._family_1_0(tmp_path, scored)
        chunks_csv = tmp_path / "fam_chunks.csv"
        df = pd.read_csv(chunks_csv)
        df.loc[0, "wordform_length_max"] = np.nan  # fine: declared (empty chunk)
        df.loc[0, "wordform_length_mean"] = np.nan
        df.to_csv(chunks_csv, index=False)
        words_csv = tmp_path / "fam_words.csv"
        w = pd.read_csv(words_csv)
        w.loc[0, "wordform_length"] = np.nan  # a producer defect: length is never null
        w.to_csv(words_csv, index=False)
        before = side.read_text()
        r = refresh_sidecar(side)
        assert r.status == "refused" and r.undeclared == {"wordform": ["wordform_length"]}
        assert side.read_text() == before
