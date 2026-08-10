"""Metadata generation for word2psy output files.

Generates a sidecar JSON file describing input, output, models, and features.
"""

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
        "lexical_norms": ("scikit-learn", "fasttext+ridge"),
        "gpt2_surprisal": ("transformers", None),
        "fasttext": ("fasttext-wheel", None),
        "word2vec": ("gensim", None),
        "emotion": ("transformers", None),
        "sentiment": ("transformers", None),
        "readability": ("textstat", None),
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
    ) -> None:
        """Add model info after it completes."""
        package_version = get_model_version(model_name)
        self.models[model_name] = {
            "version": package_version,  # legacy key, one deprecation cycle
            "package_version": package_version,
            "checkpoint": get_model_checkpoint(model_name),
            "runtime_sec": round(runtime_sec, 3),
            "features": get_feature_info(model_name, feature_names, level=level),
        }
        self.model_features[model_name] = feature_names
        self.total_runtime_sec += runtime_sec

    def build(self) -> dict[str, Any]:
        """Build the final metadata dict."""
        return {
            "schema_version": "1.0",
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
