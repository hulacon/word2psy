"""Correlation heatmap visualization."""

import fnmatch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Columns to exclude from auto-detected feature lists
_INDEX_COLUMNS = ["word_idx", "sentence_idx", "chunk_idx", "onset", "offset"]

# High-dimensional embedding prefixes to exclude
_EMBEDDING_PREFIXES = ["clip_text_"]


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

    return [c for c in matched if c not in exclude]


def plot_heatmap(
    df: pd.DataFrame,
    features: list[str] | None = None,
    method: str = "pearson",
    figsize: tuple[int, int] | None = None,
    cmap: str = "RdBu_r",
    title: str | None = None,
) -> plt.Figure:
    """Plot correlation heatmap of features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with feature columns.
    features : list of str, optional
        Features to include. Supports glob patterns.
        If None, uses scalar features (excludes high-dim embeddings).
    method : str
        Correlation method: "pearson" or "spearman".
    figsize : tuple, optional
        Figure size (width, height) in inches.
    cmap : str
        Colormap for the heatmap.
    title : str, optional
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    feature_cols = get_feature_columns(df, features)

    # For default, exclude high-dimensional embeddings
    if features is None:
        feature_cols = [
            c for c in feature_cols
            if not any(c.startswith(p) for p in _EMBEDDING_PREFIXES)
        ]

    if len(feature_cols) < 2:
        raise ValueError("Need at least 2 features for correlation heatmap.")

    # Compute correlation matrix
    corr = df[feature_cols].corr(method=method)

    # Determine figure size
    if figsize is None:
        n = len(feature_cols)
        size = max(6, min(20, n * 0.5))
        figsize = (size, size)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5, "label": f"{method.title()} correlation"},
        ax=ax,
        annot=len(feature_cols) <= 15,
        fmt=".2f",
        annot_kws={"size": 8},
    )

    ax.set_title(title or f"Feature Correlation ({method.title()})")

    fig.tight_layout()
    return fig
