"""Feature-to-visualization mapping configuration.

Defines which visualization types are appropriate for each model's output,
enabling smart defaults in the CLI and automatic visualization suggestions.
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


# Feature configurations for word2psy models
FEATURE_CONFIGS: dict[str, FeatureConfig] = {
    "lexical_norms": FeatureConfig(
        name="lexical_norms",
        description="18 psycholinguistic word properties",
        feature_type="named_distribution",
        n_dims=18,
        timeseries=True,
        mds_clustering=True,
        timeseries_mode="all",
        column_patterns=[
            "concreteness", "valence", "arousal", "dominance",
            "age_of_acquisition", "imageability", "zipf_frequency",
            "sensorimotor_*",
        ],
        scalar_features=[
            "concreteness",
            "valence",
            "arousal",
            "dominance",
            "age_of_acquisition",
            "imageability",
            "zipf_frequency",
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
        ],
    ),
    "clip_text": FeatureConfig(
        name="clip_text",
        description="512-dim CLIP ViT-B-32 text embeddings",
        feature_type="embedding",
        n_dims=512,
        timeseries=False,
        mds_clustering=True,
        timeseries_mode="none",
        column_patterns=["clip_text_*"],
        scalar_features=[],
    ),
}


def get_timeseries_features(df_columns: list[str]) -> list[str]:
    """Get columns appropriate for timeseries visualization.

    Returns scalar features from named distributions.
    Excludes high-dimensional embeddings.
    """
    timeseries_cols = []

    for config in FEATURE_CONFIGS.values():
        if not config.timeseries:
            continue

        if config.timeseries_mode == "all":
            for pattern in config.column_patterns:
                for col in df_columns:
                    if fnmatch.fnmatch(col, pattern) and col not in timeseries_cols:
                        timeseries_cols.append(col)

        elif config.timeseries_mode == "top_k":
            for feat in config.scalar_features[:config.top_k]:
                if feat in df_columns and feat not in timeseries_cols:
                    timeseries_cols.append(feat)

    return timeseries_cols


def get_mds_features(df_columns: list[str]) -> dict[str, list[str]]:
    """Get column groups appropriate for MDS/clustering visualization.

    Returns dict mapping model name to list of columns.
    """
    mds_groups = {}

    for config in FEATURE_CONFIGS.values():
        if not config.mds_clustering:
            continue

        matching_cols = []
        for pattern in config.column_patterns:
            for col in df_columns:
                if fnmatch.fnmatch(col, pattern):
                    matching_cols.append(col)

        if len(matching_cols) >= 2:
            mds_groups[config.name] = matching_cols

    return mds_groups


def detect_models_in_dataframe(df_columns: list[str]) -> list[str]:
    """Detect which word2psy models produced columns in a DataFrame."""
    detected = []

    for config in FEATURE_CONFIGS.values():
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
    }

    return recommendations


# Summary table for documentation/CLI help
VISUALIZATION_MATRIX = """
Model         | Timeseries | Scatter/MDS | Notes
--------------|------------|-------------|--------------------------------------
lexical_norms | Yes (all)  | Yes         | 18 interpretable psycholinguistic features
clip_text     | No         | Yes         | 512-dim semantic text embeddings
"""
