"""2D scatter projection visualization."""

import fnmatch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .projection import compute_projection, PROJECTION_METHODS


# Columns to exclude from auto-detected feature lists
_INDEX_COLUMNS = ["word_idx", "sentence_idx", "chunk_idx", "onset", "offset"]


def get_feature_columns(
    df: pd.DataFrame,
    patterns: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[str]:
    """Get feature columns matching patterns."""
    exclude = exclude or _INDEX_COLUMNS

    if patterns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return [c for c in numeric_cols if c not in exclude]

    all_cols = df.columns.tolist()
    matched = set()
    for pattern in patterns:
        for col in all_cols:
            if fnmatch.fnmatch(col, pattern):
                matched.add(col)

    return sorted([c for c in matched if c not in exclude])


def plot_scatter(
    df: pd.DataFrame,
    features: list[str] | None = None,
    method: str = "pca",
    color_by: str | None = None,
    figsize: tuple[int, int] = (8, 6),
    title: str | None = None,
    point_size: int = 20,
    alpha: float = 0.7,
) -> plt.Figure:
    """Plot 2D scatter projection of features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with feature columns.
    features : list of str, optional
        Features to project. Supports glob patterns (e.g., "clip_text_*").
        If None, uses all numeric features.
    method : str
        Projection method: "pca", "umap", "tsne", "mds", or "mds_nonmetric".
    color_by : str, optional
        Column to use for coloring points.
    figsize : tuple
        Figure size (width, height) in inches.
    title : str, optional
        Plot title.
    point_size : int
        Size of scatter points.
    alpha : float
        Transparency of points.

    Returns
    -------
    matplotlib.figure.Figure
    """
    feature_cols = get_feature_columns(df, features)

    if len(feature_cols) < 2:
        raise ValueError("Need at least 2 features for projection.")

    # Extract feature matrix
    X = df[feature_cols].values.copy()

    # Project to 2D using unified projection module
    X_2d, info = compute_projection(X, method=method, n_components=2)
    xlabel = info["xlabel"]
    ylabel = info["ylabel"]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Color mapping
    if color_by and color_by in df.columns:
        colors = df[color_by].values
        scatter = ax.scatter(
            X_2d[:, 0], X_2d[:, 1],
            c=colors,
            s=point_size,
            alpha=alpha,
            cmap="viridis",
        )
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(color_by)
    else:
        ax.scatter(
            X_2d[:, 0], X_2d[:, 1],
            s=point_size,
            alpha=alpha,
            c="steelblue",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"{method.upper()} Projection ({len(feature_cols)} features)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
