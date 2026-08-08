"""Feature-to-visualization mapping configuration.

Defines which visualization types are appropriate for each model's output,
enabling smart defaults in the CLI and automatic visualization suggestions.

Each config also records the output ``level`` ("word" or "chunk"), i.e.
which of the two output CSVs the model's columns live in. Context-level
models (gpt2_surprisal) produce word-table columns, so they are "word"
here even though they are position-dependent.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class FeatureConfig:
    """Configuration for a model's feature visualization options."""

    # Model name (matches MODEL_REGISTRY key)
    name: str

    # Human-readable description
    description: str

    # Feature type classification
    feature_type: Literal["scalar", "named_distribution", "embedding"]

    # Number of output dimensions
    n_dims: int

    # Which output table the columns live in
    level: Literal["word", "chunk"] = "word"

    # Visualization appropriateness
    timeseries: bool = False
    mds_clustering: bool = False

    # Special handling options
    timeseries_mode: Literal["all", "top_k", "none"] = "none"
    top_k: int = 5

    # Column patterns for matching
    column_patterns: list[str] = field(default_factory=list)

    # Scalar feature names (for models with named outputs)
    scalar_features: list[str] = field(default_factory=list)


_LEXICAL_NORM_FEATURES = [
    "concreteness",
    "valence",
    "arousal",
    "dominance",
    "age_of_acquisition",
    "imageability",
    "familiarity",
    "semantic_size",
    "gender_association",
    "socialness",
    "body_object_interaction",
    "sensorimotor_touch",
    "sensorimotor_hearing",
    "sensorimotor_smell",
    "sensorimotor_taste",
    "sensorimotor_vision",
    "sensorimotor_interoception",
    "sensorimotor_mouth",
    "sensorimotor_hand",
    "sensorimotor_foot",
    "sensorimotor_head",
    "sensorimotor_torso",
    "zipf_frequency",
]

_WORDFORM_FEATURES = ["length", "n_syllables", "n_phonemes", "old20"]

_READABILITY_FEATURES = [
    "readability_flesch_ease",
    "readability_flesch_kincaid_grade",
    "readability_gunning_fog",
    "readability_smog",
    "readability_coleman_liau",
    "readability_ari",
    "readability_dale_chall",
]

_SENTIMENT_FEATURES = [
    "sentiment_negative",
    "sentiment_neutral",
    "sentiment_positive",
]


# Feature configurations for word2psy models
FEATURE_CONFIGS: dict[str, FeatureConfig] = {
    # --- word-level ---
    "lexical_norms": FeatureConfig(
        name="lexical_norms",
        description="22 psycholinguistic word norms + Zipf frequency",
        feature_type="named_distribution",
        n_dims=23,
        level="word",
        timeseries=True,
        mds_clustering=True,
        timeseries_mode="all",
        column_patterns=list(_LEXICAL_NORM_FEATURES[:11])
        + ["sensorimotor_*", "zipf_frequency"],
        scalar_features=list(_LEXICAL_NORM_FEATURES),
    ),
    "wordform": FeatureConfig(
        name="wordform",
        description="Length, syllables, phonemes, orthographic OLD20",
        feature_type="named_distribution",
        n_dims=4,
        level="word",
        timeseries=True,
        mds_clustering=False,
        timeseries_mode="all",
        column_patterns=list(_WORDFORM_FEATURES),
        scalar_features=list(_WORDFORM_FEATURES),
    ),
    "fasttext": FeatureConfig(
        name="fasttext",
        description="300-dim fastText subword embeddings",
        feature_type="embedding",
        n_dims=300,
        level="word",
        timeseries=False,
        mds_clustering=True,
        timeseries_mode="none",
        column_patterns=["fasttext_*"],
    ),
    "word2vec": FeatureConfig(
        name="word2vec",
        description="300-dim GoogleNews word2vec embeddings",
        feature_type="embedding",
        n_dims=300,
        level="word",
        timeseries=False,
        mds_clustering=True,
        timeseries_mode="none",
        column_patterns=["word2vec_*"],
    ),
    "gpt2_surprisal": FeatureConfig(
        name="gpt2_surprisal",
        description="Word surprisal in context (bits) from GPT-2",
        feature_type="scalar",
        n_dims=1,
        level="word",
        timeseries=True,
        mds_clustering=False,
        timeseries_mode="all",
        column_patterns=["gpt2_surprisal"],
        scalar_features=["gpt2_surprisal"],
    ),
    # --- chunk-level ---
    "sentiment": FeatureConfig(
        name="sentiment",
        description="Negative/neutral/positive probabilities (RoBERTa)",
        feature_type="named_distribution",
        n_dims=3,
        level="chunk",
        timeseries=True,
        mds_clustering=False,
        timeseries_mode="all",
        column_patterns=["sentiment_*"],
        scalar_features=list(_SENTIMENT_FEATURES),
    ),
    "emotion": FeatureConfig(
        name="emotion",
        description="28 GoEmotions probabilities (RoBERTa)",
        feature_type="named_distribution",
        n_dims=28,
        level="chunk",
        timeseries=True,
        mds_clustering=True,
        timeseries_mode="top_k",
        top_k=8,
        column_patterns=["emotion_*"],
    ),
    "readability": FeatureConfig(
        name="readability",
        description="7 classic readability metrics",
        feature_type="named_distribution",
        n_dims=7,
        level="chunk",
        timeseries=True,
        mds_clustering=False,
        timeseries_mode="all",
        column_patterns=["readability_*"],
        scalar_features=list(_READABILITY_FEATURES),
    ),
    "minilm": FeatureConfig(
        name="minilm",
        description="384-dim MiniLM sentence embeddings",
        feature_type="embedding",
        n_dims=384,
        level="chunk",
        timeseries=False,
        mds_clustering=True,
        timeseries_mode="none",
        column_patterns=["minilm_*"],
    ),
    "clip_text": FeatureConfig(
        name="clip_text",
        description="512-dim CLIP ViT-B-32 text embeddings",
        feature_type="embedding",
        n_dims=512,
        level="chunk",
        timeseries=False,
        mds_clustering=True,
        timeseries_mode="none",
        column_patterns=["clip_text_*"],
    ),
    # Pseudo-model: pipeline-generated per-chunk aggregates of word-level
    # features ({feature}_{mean,sd,min,max}; see pipeline.aggregate_word_features)
    "word_aggregates": FeatureConfig(
        name="word_aggregates",
        description="Per-chunk mean/sd/min/max of word-level features",
        feature_type="named_distribution",
        n_dims=len(_LEXICAL_NORM_FEATURES + _WORDFORM_FEATURES + ["gpt2_surprisal"]) * 4,
        level="chunk",
        timeseries=True,
        mds_clustering=True,
        timeseries_mode="top_k",
        top_k=8,
        column_patterns=[
            f"{feat}_{stat}"
            for feat in _LEXICAL_NORM_FEATURES + _WORDFORM_FEATURES + ["gpt2_surprisal"]
            for stat in ("mean", "sd", "min", "max")
        ],
    ),
}


def get_model_columns(df_columns: list[str], model_name: str) -> list[str]:
    """Get the columns in ``df_columns`` produced by ``model_name``."""
    config = FEATURE_CONFIGS.get(model_name)
    if not config:
        return []

    cols = []
    for pattern in config.column_patterns:
        for col in df_columns:
            if fnmatch.fnmatch(col, pattern) and col not in cols:
                cols.append(col)
    return cols


def get_timeseries_features(df_columns: list[str]) -> list[str]:
    """Get columns appropriate for timeseries visualization.

    Returns scalar features from named distributions.
    Excludes high-dimensional embeddings.
    """
    timeseries_cols = []

    for config in FEATURE_CONFIGS.values():
        if not config.timeseries:
            continue

        if config.timeseries_mode in ("all", "top_k"):
            for pattern in config.column_patterns:
                for col in df_columns:
                    if fnmatch.fnmatch(col, pattern) and col not in timeseries_cols:
                        timeseries_cols.append(col)

    return timeseries_cols


def get_mds_features(df_columns: list[str]) -> dict[str, list[str]]:
    """Get column groups appropriate for MDS/clustering visualization.

    Returns dict mapping model name to list of columns.
    """
    mds_groups = {}

    for config in FEATURE_CONFIGS.values():
        if not config.mds_clustering:
            continue

        matching_cols = get_model_columns(df_columns, config.name)
        if len(matching_cols) >= 2:
            mds_groups[config.name] = matching_cols

    return mds_groups


def detect_models_in_dataframe(
    df_columns: list[str],
    level: Literal["word", "chunk"] | None = None,
) -> list[str]:
    """Detect which word2psy models produced columns in a DataFrame.

    Parameters
    ----------
    df_columns : list of str
        Column names to inspect.
    level : "word" or "chunk", optional
        If given, only report models whose output lives at that level.
    """
    detected = []

    for config in FEATURE_CONFIGS.values():
        if level is not None and config.level != level:
            continue
        for pattern in config.column_patterns:
            for col in df_columns:
                if fnmatch.fnmatch(col, pattern):
                    if config.name not in detected:
                        detected.append(config.name)
                    break

    return detected


def get_visualization_recommendations(df_columns: list[str]) -> dict:
    """Get visualization recommendations for a DataFrame.

    Returns a dict with recommended visualizations and the features to use.
    """
    detected_models = detect_models_in_dataframe(df_columns)
    timeseries_features = get_timeseries_features(df_columns)
    mds_groups = get_mds_features(df_columns)

    recommendations = {
        "detected_models": detected_models,
        "timeseries": {
            "available": len(timeseries_features) > 0,
            "features": timeseries_features,
            "description": "Scalar features over word/sentence/chunk sequence",
        },
        "heatmap": {
            "available": len(timeseries_features) >= 2,
            "features": timeseries_features,
            "description": "Correlation matrix of scalar features",
        },
        "scatter": {
            "available": len(mds_groups) > 0,
            "groups": mds_groups,
            "description": "2D projection to visualize similarity structure",
        },
        "browse": {
            "available": True,
            "description": "Interactive HTML dashboard with per-word detail view",
        },
    }

    return recommendations


# Summary table for documentation/CLI help
VISUALIZATION_MATRIX = """
Model          | Level | Timeseries  | Scatter/MDS | Notes
---------------|-------|-------------|-------------|----------------------------------
lexical_norms  | word  | Yes (all)   | Yes         | 23 psycholinguistic features
wordform       | word  | Yes (all)   | No          | 4 orthographic/phonological stats
fasttext       | word  | No          | Yes         | 300-dim subword embeddings
word2vec       | word  | No          | Yes         | 300-dim embeddings (NaN for OOV)
gpt2_surprisal | word  | Yes         | No          | 1 contextual surprisal value
sentiment      | chunk | Yes (all)   | No          | 3 sentiment probabilities
emotion        | chunk | Yes (top-8) | Yes         | 28 GoEmotions probabilities
readability    | chunk | Yes (all)   | No          | 7 readability metrics
minilm         | chunk | No          | Yes         | 384-dim sentence embeddings
clip_text      | chunk | No          | Yes         | 512-dim text embeddings
word_aggregates| chunk | Yes (top-8) | Yes         | per-chunk mean/sd/min/max of word features
"""
