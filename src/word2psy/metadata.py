"""Metadata generation for word2psy output files.

Generates a sidecar JSON file describing input, output, models, and features.
"""

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def get_version() -> str:
    """Get word2psy package version."""
    try:
        from word2psy import __version__

        return __version__
    except Exception:
        pass
    try:
        from importlib.metadata import version

        return version("word2psy")
    except Exception:
        return "unknown"


def get_model_version(model_name: str) -> str:
    """Get version of the underlying model package."""
    version_map = {
        "clip_text": ("open_clip_torch", None),
        "ebind_text": ("ebind", None),
        "lexical_norms": ("scikit-learn", "fasttext+ridge"),
        "lexical_norms_experiential": ("numpy", "fasttext+ridge"),
        "gpt2_surprisal": ("transformers", None),
        "fasttext": ("fasttext-wheel", None),
        "word2vec": ("gensim", None),
        "emotion": ("transformers", None),
        "sentiment": ("transformers", None),
        "readability": ("textstat", None),
        "interaction": ("nltk", None),
        "wordform": ("rapidfuzz", None),
        "minilm": ("sentence-transformers", None),
    }
    pkg, fallback = version_map.get(model_name, (None, "unknown"))
    if pkg:
        try:
            from importlib.metadata import version

            return version(pkg)
        except Exception:
            pass
    return fallback or "unknown"


def get_model_checkpoint(model_name: str) -> str | None:
    """Return the model class's ``checkpoint`` attribute (Contract B §4.1).

    Class-level defaults only — accurate for CLI runs, which never override
    checkpoint arguments. API users constructing models with non-default
    checkpoints should amend the sidecar themselves.
    """
    try:
        from word2psy.cli import _load_model_class

        cls = _load_model_class(model_name)
        return getattr(cls, "checkpoint", None)
    except Exception:
        return None


SCHEMA_VERSION = "1.2"  # Contract B §4.1 extractor output convention

AGGREGATE_STATS = ("mean", "sd", "min", "max")


def declared_nulls(model_name: str) -> dict[str, dict[str, str]]:
    """The model class's own ``nulls`` map. Raises on a lookup error: a silent
    ``{}`` would be a false claim that the model cannot emit NaN."""
    from word2psy.cli import _load_model_class

    return {c: dict(e) for c, e in _load_model_class(model_name).nulls.items()}


def model_nulls(
    model_name: str,
    columns: list[str],
    aggregate_columns: list[str] | None = None,
    pooled_columns: list[str] | None = None,
) -> dict[str, dict[str, str]]:
    """The sidecar ``nulls`` map: declared entries for emitted columns, plus
    derived entries for the chunk aggregates (DECIDED 2026-09-23):

    - ``{f}_sd`` is ``undefined``: fewer than two words with a value (ddof=1),
      e.g. every one-word chunk, or an empty chunk.
    - ``{f}_mean/_min/_max`` inherit the word column's kind when ``f`` has an
      entry (an all-null chunk), else ``undefined`` (the chunk has no words).
    - a pooled embedding dimension without its own entry is ``undefined`` on
      a chunk with no words (a mean over zero vectors); one with an entry
      (word2vec OOV, ``missing``) keeps it, and its ``when`` names both.
    """
    declared = declared_nulls(model_name)
    emitted = set(columns)
    out = {c: e for c, e in declared.items() if c in emitted or c in set(pooled_columns or ())}
    for col in pooled_columns or ():
        out.setdefault(col, {"means": "undefined",
                             "when": "on the chunks table, the chunk has no words "
                                     "(the pooled vector is a mean over zero words)"})
    for col in aggregate_columns or ():
        feat, stat = col.rsplit("_", 1)
        if stat == "sd":
            out[col] = {"means": "undefined",
                        "when": "fewer than two words in the chunk have a value (sd uses ddof=1): "
                                "every one-word chunk, or a chunk with no words"}
        elif feat in declared:
            out[col] = {"means": declared[feat]["means"],
                        "when": f"every word in the chunk is null ({declared[feat]['when']}); "
                                "or the chunk has no words"}
        else:
            out[col] = {"means": "undefined", "when": "the chunk has no words"}
    return out


# Contract B §4.1 (schema 1.2) label vocabularies.
INTERPRETABILITY = ("nameable", "opaque")
DERIVATIONS = ("formula", "formula-on-model", "trained-on-labels", "probe", "zero-shot",
               "embedding", "bookkeeping")
NEEDS_DERIVED_FROM = frozenset({"formula-on-model", "probe", "zero-shot"})
LABEL_KEYS = frozenset({"interpretability", "derivation", "derived_from", "source_validation",
                        "LongName", "Description", "Units", "Levels", "TermURL"})
SOURCE_VALIDATION_KEYS = frozenset({"citation", "eval_dataset", "metric", "value"})


def declared_labels(model_name: str) -> dict[str, dict]:
    """The model class's own ``labels`` (a deep copy). Raises on a lookup error,
    like ``declared_nulls``: a silent empty map would leave columns unlabelled."""
    from word2psy.cli import _load_model_class

    lab = _load_model_class(model_name).labels
    return {"default": copy.deepcopy(lab.get("default", {})),
            "columns": copy.deepcopy(lab.get("columns", {}))}


def resolve_label(labels: dict[str, dict], column: str) -> dict[str, Any]:
    """One column's labels: the default, with its ``columns`` entry merged key-wise."""
    return {**labels.get("default", {}), **labels.get("columns", {}).get(column, {})}


def model_labels(
    model_name: str,
    columns: list[str],
    aggregate_columns: list[str] | None = None,
    pooled_columns: list[str] | None = None,
    count_column: str | None = None,
) -> dict[str, dict]:
    """The sidecar ``labels`` map: the declared default, the declared ``columns``
    entries for emitted columns (word/chunk features, pooled dimensions, the
    pooling count column), and one entry per chunk aggregate ``{f}_{stat}``
    copying the word column ``f``'s entry. Aggregates of a column without an
    entry need none: the default already covers them."""
    declared = declared_labels(model_name)
    emitted = set(columns) | set(pooled_columns or ())
    if count_column:
        emitted.add(count_column)
    out = {c: e for c, e in declared["columns"].items() if c in emitted}
    for col in aggregate_columns or ():
        feat = col.rsplit("_", 1)[0]
        if feat in declared["columns"]:
            out[col] = copy.deepcopy(declared["columns"][feat])
    return {"default": declared["default"], "columns": out}


def label_problems(labels: dict[str, dict], inventory: list[str]) -> list[str]:
    """What is wrong with a ``labels`` map against a column inventory (empty if
    nothing): an unresolved required key, a ``columns`` key outside the
    inventory, a value outside the vocabularies, an unknown key, or
    ``derived_from`` present where it is forbidden or absent where required."""
    problems: list[str] = []
    if set(labels) != {"default", "columns"}:
        problems.append(f"labels keys are {sorted(labels)}, not ['columns', 'default']")
    inv = set(inventory)
    for col in sorted(set(labels.get("columns", {})) - inv):
        problems.append(f"{col}: labelled but not in the inventory")
    for key in set(labels.get("default", {})) - LABEL_KEYS:
        problems.append(f"default: unknown key {key!r}")
    for col, entry in labels.get("columns", {}).items():
        for key in set(entry) - LABEL_KEYS:
            problems.append(f"{col}: unknown key {key!r}")
    for col in inventory:
        lab = resolve_label(labels, col)
        if lab.get("interpretability") not in INTERPRETABILITY:
            problems.append(f"{col}: interpretability {lab.get('interpretability')!r}")
        deriv = lab.get("derivation")
        if deriv not in DERIVATIONS:
            problems.append(f"{col}: derivation {deriv!r}")
        has_parent = bool(lab.get("derived_from"))
        if has_parent != (deriv in NEEDS_DERIVED_FROM):
            problems.append(f"{col}: derived_from {'present' if has_parent else 'absent'} "
                            f"with derivation {deriv!r}")
        if "derived_from" in lab and not isinstance(lab["derived_from"], str):
            problems.append(f"{col}: derived_from is not a string")
        for item in lab.get("source_validation", []):
            if set(item) != SOURCE_VALIDATION_KEYS:
                problems.append(f"{col}: source_validation item keys {sorted(item)}")
    return problems


def get_feature_info(
    model_name: str, feature_names: list[str], level: str | None = None
) -> dict[str, Any]:
    """Get feature pattern/definition info for a model.

    Embedding-style feature sets (all names ``{model}_{NNN}``) are
    compacted to a pattern instead of listing every column.
    """
    count = len(feature_names)
    prefix = f"{model_name}_"

    info: dict[str, Any]
    if feature_names and all(
        f.startswith(prefix) and f[len(prefix) :].isdigit() for f in feature_names
    ):
        info = {
            "pattern": f"{model_name}_{{NNN}}",
            "range": [0, count - 1],
            "count": count,
        }
    else:
        info = {"columns": feature_names, "count": count}

    if level:
        info["level"] = level
    return info


class MetadataBuilder:
    """Builder for output metadata."""

    def __init__(self):
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.input_info: dict[str, Any] = {}
        self.output_info: dict[str, Any] = {}
        self.device: str | None = None
        self.models: dict[str, Any] = {}
        self.model_features: dict[str, list[str]] = {}
        self.total_runtime_sec = 0.0

    def set_input_text(
        self,
        path: Path | None,
        n_words: int,
        n_chunks: int,
        n_sentences: int,
    ) -> None:
        """Set input info for text processing."""
        self.input_info = {
            "type": "text",
            "n_words": n_words,
            "n_chunks": n_chunks,
            "n_sentences": n_sentences,
        }
        if path is not None:
            self.input_info["path"] = str(Path(path).resolve())

    def set_output(self, kind: str, path: Path, rows: int, columns: int) -> None:
        """Record an output file under ``kind`` (e.g. "words", "chunks")."""
        self.output_info[kind] = {
            "path": str(Path(path).resolve()),
            "rows": rows,
            "columns": columns,
        }

    def set_device(self, device: str) -> None:
        """Set device used for inference."""
        self.device = str(device)

    def add_model(
        self,
        model_name: str,
        feature_names: list[str],
        runtime_sec: float,
        level: str | None = None,
        pooled_features: list[str] | None = None,
        aggregate_features: list[str] | None = None,
    ) -> None:
        """Add model info after it completes.

        ``pooled_features`` names the chunk-level columns produced by
        mean-pooling a word-level embedding (see
        ``pipeline.pool_word_embeddings``). Recorded so a consumer can
        tell a pooled chunk vector from one the model emitted directly --
        the column names are identical either way.
        """
        package_version = get_model_version(model_name)
        entry = {
            "version": package_version,  # legacy key, one deprecation cycle
            "package_version": package_version,
            "checkpoint": get_model_checkpoint(model_name),
            "runtime_sec": round(runtime_sec, 3),
            "features": get_feature_info(model_name, feature_names, level=level),
        }
        if pooled_features:
            dims = [c for c in pooled_features if not c.endswith("_n_pooled")]
            entry["chunk_pooling"] = {
                "stat": "mean",
                "nan_policy": "omit",
                "count_column": f"{model_name}_n_pooled",
                "features": get_feature_info(model_name, dims, level="chunk"),
            }
        if aggregate_features:
            entry["chunk_aggregates"] = {
                "stats": list(AGGREGATE_STATS),
                "nan_policy": "omit",
                "sd_ddof": 1,
                "columns": list(aggregate_features),
            }
        pooled_dims = [c for c in (pooled_features or []) if not c.endswith("_n_pooled")]
        entry["nulls"] = model_nulls(model_name, feature_names, aggregate_features, pooled_dims)
        entry["labels"] = model_labels(
            model_name, feature_names, aggregate_features, pooled_dims,
            count_column=f"{model_name}_n_pooled" if pooled_features else None)
        self.models[model_name] = entry
        self.model_features[model_name] = feature_names
        self.total_runtime_sec += runtime_sec

    def build(self) -> dict[str, Any]:
        """Build the final metadata dict."""
        return {
            "schema_version": SCHEMA_VERSION,
            "extractor": "word2psy",
            "extractor_version": get_version(),
            "word2psy_version": get_version(),  # legacy key, one deprecation cycle
            "created_at": self.created_at,
            "input": self.input_info,
            "output": self.output_info,
            "device": self.device,
            "total_runtime_sec": round(self.total_runtime_sec, 3),
            "models": self.models,
        }

    def save(self, output_path: Path) -> Path:
        """Save metadata to sidecar JSON file."""
        meta_path = Path(output_path).with_suffix(".meta.json")
        metadata = self.build()
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        return meta_path
