"""Time series visualization for text feature scores."""

from __future__ import annotations

import fnmatch
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .index_utils import detect_index_column, prepare_index_values


# Columns to exclude from auto-detected feature lists
_INDEX_COLUMNS = ["word_idx", "sentence_idx", "chunk_idx", "onset", "offset"]

# High-dimensional embedding prefixes to exclude from timeseries
_EMBEDDING_PREFIXES = ["clip_text_"]


def get_feature_columns(
    df: pd.DataFrame,
    patterns: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[str]:
    """Get feature columns matching patterns.

    Parameters
    ----------
    df : pd.DataFrame
    patterns : list of str, optional
        Glob patterns to match (e.g., ["lexical_norms_concreteness", "lexical_norms_sensorimotor_*"]).
        If None, returns all numeric columns except common ID columns.
    exclude : list of str, optional
        Columns to exclude.

    Returns
    -------
    list of str
    """
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


def plot_timeseries(
    df: pd.DataFrame,
    features: list[str] | None = None,
    index_col: str | None = None,
    figsize: tuple[int, int] | None = None,
    title: str | None = None,
    show_diff: bool = False,
    rolling_window: int | None = None,
) -> plt.Figure:
    """Plot feature values over word/sentence/chunk sequence.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with an index column and feature columns.
    features : list of str, optional
        Features to plot. Supports glob patterns (e.g., "lexical_norms_sensorimotor_*").
        If None, plots all scalar features (not high-dim embeddings).
    index_col : str, optional
        Name of the index column. If None, auto-detects from DataFrame
        (word_idx, chunk_idx, sentence_idx, chunk_label, onset).
    figsize : tuple, optional
        Figure size (width, height) in inches.
    title : str, optional
        Plot title.
    show_diff : bool
        If True, plot first-order differences between consecutive points.
    rolling_window : int, optional
        If provided, overlay rolling average with this window size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Auto-detect index column
    detected_col, index_type = detect_index_column(df)
    if index_col is None:
        index_col = detected_col

    if index_col is None:
        # Fall back to row numbers
        index_col = "_row_index"
        df = df.copy()
        df[index_col] = np.arange(len(df))
        index_type = "integer"

    if index_col not in df.columns and index_col != "_row_index":
        raise ValueError(f"Index column '{index_col}' not found in DataFrame.")

    # Validate rolling window
    if rolling_window is not None:
        if rolling_window >= len(df):
            new_window = max(1, len(df) // 3)
            warnings.warn(
                f"rolling_window ({rolling_window}) >= data length ({len(df)}), "
                f"reducing to {new_window}."
            )
            rolling_window = new_window

    # Get feature columns
    feature_cols = get_feature_columns(df, features)

    # For default (no patterns), exclude high-dimensional embeddings
    if features is None:
        feature_cols = [
            c for c in feature_cols
            if not any(c.startswith(p) for p in _EMBEDDING_PREFIXES)
        ]

    if not feature_cols:
        raise ValueError("No features found to plot.")

    # Prepare index values and formatting
    x_values, format_info = prepare_index_values(df, index_col, index_type)

    # Determine layout
    n_features = len(feature_cols)
    if n_features == 1:
        nrows, ncols = 1, 1
    elif n_features <= 3:
        nrows, ncols = n_features, 1
    elif n_features <= 6:
        nrows, ncols = (n_features + 1) // 2, 2
    else:
        nrows, ncols = (n_features + 2) // 3, 3

    if figsize is None:
        figsize = (4 * ncols, 2.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        values = df[col].values

        # Plot main line
        ax.plot(x_values, values, linewidth=1, label=col, alpha=0.8)

        # Plot differences if requested
        if show_diff:
            diff_values = np.diff(values)
            diff_x = x_values[1:]
            ax.plot(diff_x, diff_values, linewidth=0.8, label=f"{col} (diff)",
                    alpha=0.6, linestyle="--")

        # Plot rolling average if requested
        if rolling_window is not None:
            rolling_avg = pd.Series(values).rolling(window=rolling_window, center=True).mean()
            ax.plot(x_values, rolling_avg, linewidth=1.5, label=f"rolling ({rolling_window})",
                    alpha=0.9, color="red")

        ax.set_xlabel(format_info["xlabel"])
        ax.set_ylabel(col)
        ax.set_title(col)
        ax.grid(True, alpha=0.3)

        # Handle ordinal x-axis labels
        if index_type == "ordinal" and "tickvals" in format_info:
            ax.set_xticks(format_info["tickvals"])
            ax.set_xticklabels(format_info["ticktext"], rotation=45, ha="right")

        # Show legend if multiple lines
        if show_diff or rolling_window is not None:
            ax.legend(fontsize=8, loc="upper right")

    # Hide unused axes
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=12)

    fig.tight_layout()
    return fig
