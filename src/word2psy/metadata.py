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
    }
    pkg, fallback = version_map.get(model_name, (None, "unknown"))
    if pkg:
        try:
            from importlib.metadata import version

            return version(pkg)
        except Exception:
            pass
    return fallback or "unknown"


def get_feature_info(model_name: str, feature_names: list[str]) -> dict[str, Any]:
    """Get feature pattern/definition info for a model."""
    count = len(feature_names)

    if model_name == "clip_text" and feature_names and feature_names[0].startswith(
        "clip_text_"
    ):
        return {
            "pattern": "clip_text_{NNN}",
            "range": [0, count - 1],
            "count": count,
            "level": "chunk",
        }

    if model_name == "lexical_norms":
        return {
            "columns": feature_names,
            "count": count,
            "level": "word",
        }

    return {"columns": feature_names, "count": count}


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
        self, model_name: str, feature_names: list[str], runtime_sec: float
    ) -> None:
        """Add model info after it completes."""
        self.models[model_name] = {
            "version": get_model_version(model_name),
            "runtime_sec": round(runtime_sec, 3),
            "features": get_feature_info(model_name, feature_names),
        }
        self.model_features[model_name] = feature_names
        self.total_runtime_sec += runtime_sec

    def build(self) -> dict[str, Any]:
        """Build the final metadata dict."""
        return {
            "word2psy_version": get_version(),
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
