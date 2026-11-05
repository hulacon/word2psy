"""Contract B §4.1 (schema 1.2) producer duty: every column word2psy emits is labelled.

For every model: each column in its inventory (chunk aggregates and the
pooling count column included) resolves to `interpretability` and
`derivation`; every `columns` key is a real column (embedding patterns
expanded); values come from the vocabularies; and `derived_from` is present
exactly when the derivation is formula-on-model, probe or zero-shot, and then
equals the parent model's recorded `checkpoint`. Offline: class attributes
only, plus an nltk/textstat scoring run for the sidecar and refresh checks.
"""

import json

import pandas as pd
import pytest

from word2psy.cli import MODEL_REGISTRY, _load_model_class
from word2psy.metadata import (
    AGGREGATE_STATS,
    SCHEMA_VERSION,
    MetadataBuilder,
    declared_labels,
    get_model_checkpoint,
    label_problems,
    model_labels,
    resolve_label,
)
from word2psy.norms.train import NORM_DIMENSIONS
from word2psy.pipeline import score_text
from word2psy.sidecar import _inventory, refresh_sidecar
from word2psy.viz.feature_config import FEATURE_CONFIGS

CHUNKS = ["Hello there, do you see me?", "Zyxqvwpt", "...", "Cat"]


def static_inventory(name: str) -> list[str] | None:
    """Every column the model can emit, from the static feature configs: the
    expanded embedding pattern (+ the pooling count for word-level spaces), or
    the named columns (+ their chunk aggregates for word/context models).
    None when the names are only known at load time (emotion's label set)."""
    level = _load_model_class(name).level
    cfg = FEATURE_CONFIGS[name]
    if cfg.feature_type == "embedding":
        width = max(3, len(str(cfg.n_dims - 1)))
        cols = [f"{name}_{i:0{width}d}" for i in range(cfg.n_dims)]
        return cols + [f"{name}_n_pooled"] if level == "word" else cols
    if not cfg.scalar_features:
        return None
    cols = list(cfg.scalar_features)
    if level in ("word", "context"):
        cols += [f"{c}_{s}" for c in cfg.scalar_features for s in AGGREGATE_STATS]
    return cols


def sidecar_labels(name: str, inventory: list[str]) -> dict:
    """What the sidecar writer emits for a run that produced ``inventory``."""
    is_agg = tuple(f"_{s}" for s in AGGREGATE_STATS)
    count = f"{name}_n_pooled" if f"{name}_n_pooled" in inventory else None
    aggregates = [c for c in inventory
                  if c.endswith(is_agg) and c.rsplit("_", 1)[0] in inventory]
    cols = [c for c in inventory if c not in aggregates and c != count]
    pooled = cols if count else None
    return model_labels(name, cols, aggregates, pooled, count_column=count)


@pytest.mark.parametrize("model", sorted(MODEL_REGISTRY))
def test_every_column_resolves(model):
    inventory = static_inventory(model)
    if inventory is None:
        # columns unknown until load: the default alone must label any of them
        assert declared_labels(model)["columns"] == {}, model
        inventory = [f"{model}_any"]
    assert label_problems(sidecar_labels(model, inventory), inventory) == []


@pytest.mark.parametrize("model", sorted(MODEL_REGISTRY))
def test_every_declared_column_key_is_real(model):
    inventory = static_inventory(model) or []
    for col in declared_labels(model)["columns"]:
        assert col in inventory, (model, col)


def test_every_registered_model_has_a_static_config():
    assert set(MODEL_REGISTRY) <= set(FEATURE_CONFIGS)


@pytest.mark.parametrize("model", sorted(MODEL_REGISTRY))
def test_derived_from_is_the_parent_checkpoint(model):
    known = {get_model_checkpoint(m) for m in MODEL_REGISTRY} - {None}
    labels = declared_labels(model)
    for entry in [labels["default"], *labels["columns"].values()]:
        if "derived_from" in entry:
            assert entry["derived_from"] in known, (model, entry["derived_from"])


class TestSeededChoices:
    """The column audit the labels were seeded from."""

    def test_lexical_norms_probes_on_fasttext_and_zipf_formula(self):
        labels = declared_labels("lexical_norms")
        inv = static_inventory("lexical_norms")
        dims = [f"lexical_norms_{d}" for d in NORM_DIMENSIONS]
        assert set(dims) | {"lexical_norms_zipf_frequency"} == {
            c for c in inv if not c.endswith(tuple(f"_{s}" for s in AGGREGATE_STATS))}
        assert len(dims) == 23
        for col in dims:
            lab = resolve_label(labels, col)
            assert lab["derivation"] == "probe"
            assert lab["derived_from"] == get_model_checkpoint("fasttext") == "crawl-300d-2M-subword"
            assert lab["interpretability"] == "nameable"
        zipf = resolve_label(labels, "lexical_norms_zipf_frequency")
        assert zipf["derivation"] == "formula" and "derived_from" not in zipf

    @pytest.mark.parametrize("model", ["fasttext", "word2vec"])
    def test_pooling_count_is_bookkeeping(self, model):
        lab = resolve_label(declared_labels(model), f"{model}_n_pooled")
        assert lab["derivation"] == "bookkeeping"
        assert resolve_label(declared_labels(model), f"{model}_000")["derivation"] == "embedding"

    @pytest.mark.parametrize("model", ["fasttext", "word2vec", "minilm", "clip_text",
                                       "clap_text", "ebind_text"])
    def test_embeddings_are_opaque(self, model):
        assert {k: declared_labels(model)["default"][k]
                for k in ("interpretability", "derivation")} == {
            "interpretability": "opaque", "derivation": "embedding"}

    def test_surprisal_is_formula_on_gpt2_in_bits(self):
        lab = resolve_label(declared_labels("gpt2_surprisal"), "gpt2_surprisal")
        assert lab["derivation"] == "formula-on-model"
        assert lab["derived_from"] == get_model_checkpoint("gpt2_surprisal")
        assert lab["Units"] == "bits"

    @pytest.mark.parametrize("model,derivation", [
        ("emotion", "trained-on-labels"), ("sentiment", "trained-on-labels"),
        ("wordform", "formula"), ("readability", "formula"), ("interaction", "formula"),
    ])
    def test_named_models(self, model, derivation):
        default = declared_labels(model)["default"]
        assert default["interpretability"] == "nameable"
        assert default["derivation"] == derivation


class TestLabelProblems:
    INV = ["m_a", "m_b"]

    def _lab(self, default, columns=None):
        return {"default": default, "columns": columns or {}}

    def test_clean(self):
        lab = self._lab({"interpretability": "opaque", "derivation": "embedding"})
        assert label_problems(lab, self.INV) == []

    @pytest.mark.parametrize("labels", [
        {"default": {"interpretability": "opaque"}, "columns": {}},  # no derivation
        {"default": {"interpretability": "legible", "derivation": "formula"}, "columns": {}},
        {"default": {"interpretability": "opaque", "derivation": "probe"}, "columns": {}},
        {"default": {"interpretability": "opaque", "derivation": "formula",
                     "derived_from": "x"}, "columns": {}},
        {"default": {"interpretability": "opaque", "derivation": "formula"},
         "columns": {"m_zzz": {"Units": "s"}}},  # not a column
        {"default": {"interpretability": "opaque", "derivation": "formula", "units": "s"},
         "columns": {}},  # not a key
        {"default": {"interpretability": "opaque", "derivation": "formula",
                     "source_validation": [{"citation": "x"}]}, "columns": {}},
    ])
    def test_caught(self, labels):
        assert label_problems(labels, self.INV)

    def test_column_entry_merges_keywise(self):
        lab = self._lab({"interpretability": "nameable", "derivation": "formula", "Units": "s"},
                        {"m_b": {"derivation": "probe", "derived_from": "parent"}})
        assert resolve_label(lab, "m_b") == {"interpretability": "nameable",
                                             "derivation": "probe", "derived_from": "parent",
                                             "Units": "s"}
        assert label_problems(lab, self.INV) == []


@pytest.fixture(scope="module")
def scored():
    models = [_load_model_class(n)() for n in ("wordform", "interaction", "readability")]
    words, chunks = score_text(CHUNKS, models, quiet=True)
    b = MetadataBuilder()
    for m in models:
        b.add_model(m.name, m.feature_names_, 1.0, level=m.level,
                    aggregate_features=getattr(m, "aggregate_features_", None))
    return words, chunks, b.build()


def _sidecar_inventory(name, entry):
    cols = _inventory(name, entry["features"])
    cols += entry.get("chunk_aggregates", {}).get("columns", [])
    pooling = entry.get("chunk_pooling")
    if pooling:
        cols += _inventory(name, pooling["features"]) + [pooling["count_column"]]
    return cols


def test_sidecar_labels_cover_every_inventoried_column(scored):
    _, chunks, meta = scored
    assert meta["schema_version"] == SCHEMA_VERSION == "1.2"
    for name, entry in meta["models"].items():
        inv = _sidecar_inventory(name, entry)
        assert label_problems(entry["labels"], inv) == [], name
    # the emitted chunk columns are all inventoried, so all labelled
    feats = [c for c in chunks if c.split("_")[0] in {"wordform", "interaction", "readability"}]
    inv = {c for n, e in meta["models"].items() for c in _sidecar_inventory(n, e)}
    assert set(feats) <= inv


def test_aggregates_inherit_the_word_column(scored):
    labels = scored[2]["models"]["wordform"]["labels"]
    word = resolve_label(labels, "wordform_n_phonemes")
    for stat in AGGREGATE_STATS:
        assert resolve_label(labels, f"wordform_n_phonemes_{stat}") == word
    assert word["Units"] == "phonemes"


def test_pooled_count_labelled_only_when_emitted():
    dims = [f"fasttext_{i:03d}" for i in range(300)]
    with_pool = model_labels("fasttext", dims, pooled_columns=dims,
                             count_column="fasttext_n_pooled")
    assert with_pool["columns"]["fasttext_n_pooled"]["derivation"] == "bookkeeping"
    assert label_problems(with_pool, dims + ["fasttext_n_pooled"]) == []
    words_only = model_labels("fasttext", dims)
    assert words_only["columns"] == {}
    assert label_problems(words_only, dims) == []


def test_lexical_norms_sidecar_drops_unemitted_columns():
    got = model_labels("lexical_norms", ["lexical_norms_valence"], ["lexical_norms_valence_mean"])
    assert set(got["columns"]) == {"lexical_norms_valence", "lexical_norms_valence_mean"}
    assert label_problems(got, ["lexical_norms_valence", "lexical_norms_valence_mean"]) == []


class TestRefresh:
    def _family_1_1(self, tmp_path, scored):
        words, chunks, meta = scored
        meta = json.loads(json.dumps(meta))
        for kind, df in (("words", words), ("chunks", chunks)):
            path = tmp_path / f"fam_{kind}.csv"
            df.to_csv(path, index=False)
            meta["output"][kind] = {"path": str(path), "rows": len(df), "columns": df.shape[1]}
        meta["schema_version"] = "1.1"
        for e in meta["models"].values():  # a 1.1 sidecar: nulls, no labels
            e.pop("labels")
        side = tmp_path / "fam.meta.json"
        side.write_text(json.dumps(meta, indent=2))
        return side

    def test_refresh_writes_labels_json_only(self, tmp_path, scored):
        side = self._family_1_1(tmp_path, scored)
        csvs = {p: p.read_bytes() for p in tmp_path.glob("*.csv")}
        r = refresh_sidecar(side)
        assert r.status == "refreshed"
        meta = json.loads(side.read_text())
        assert meta["schema_version"] == "1.2"
        for name, entry in meta["models"].items():
            assert entry["labels"] == scored[2]["models"][name]["labels"]  # as a fresh run
        assert "models.*.labels" in meta["refreshed"][-1]["fields"]
        assert all(p.read_bytes() == b for p, b in csvs.items())
        assert refresh_sidecar(side).status == "unchanged"

    def test_refresh_labels_the_pooling_count(self, tmp_path):
        dims = [f"fasttext_{i:03d}" for i in range(300)]
        chunks = pd.DataFrame({"chunk_idx": [0], "fasttext_n_pooled": [1],
                               **{d: [0.1] for d in dims}})
        words = pd.DataFrame({"chunk_idx": [0], "word": ["cat"], **{d: [0.1] for d in dims}})
        b = MetadataBuilder()
        b.add_model("fasttext", dims, 1.0, level="word",
                    pooled_features=["fasttext_n_pooled", *dims])
        meta = b.build()
        for kind, df in (("words", words), ("chunks", chunks)):
            path = tmp_path / f"fam_{kind}.csv"
            df.to_csv(path, index=False)
            meta["output"][kind] = {"path": str(path), "rows": 1, "columns": df.shape[1]}
        fresh = meta["models"]["fasttext"].pop("labels")
        assert "fasttext_n_pooled" in fresh["columns"]
        meta["schema_version"] = "1.1"
        side = tmp_path / "fam.meta.json"
        side.write_text(json.dumps(meta))
        assert refresh_sidecar(side).status == "refreshed"
        assert json.loads(side.read_text())["models"]["fasttext"]["labels"] == fresh
