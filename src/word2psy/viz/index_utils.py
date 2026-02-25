"""Index column detection and axis formatting for text data.

Handles detection of index columns (word_idx, sentence_idx, chunk_idx,
chunk_label, onset) and provides formatting utilities for x-axis labels
across visualizations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_index_column(
    df: pd.DataFrame,
) -> tuple[str | None, str]:
    """Detect the index column and its type from a text DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The data frame to analyze.

    Returns
    -------
    column_name : str or None
        Name of the detected index column, or None if not found.
    index_type : str
        Type of index: "time", "integer", or "ordinal".
        - "time": Continuous time values (onset/offset in seconds)
        - "integer": Integer indices (word_idx, sentence_idx, chunk_idx)
        - "ordinal": Categorical labels (chunk_label, word)
    """
    candidates = [
        ("word_idx", "integer"),
        ("chunk_idx", "integer"),
        ("sentence_idx", "integer"),
        ("chunk_label", "ordinal"),
        ("onset", "time"),
    ]

    for col, default_type in candidates:
        if col in df.columns:
            idx_type = _classify_index_type(df, col, default_type)
            return col, idx_type

    return None, "integer"


def _classify_index_type(
    df: pd.DataFrame,
    col: str,
    default: str = "integer",
) -> str:
    """Classify the type of an index column."""
    if col in ("onset", "offset"):
        return "time"
    elif col in ("word_idx", "sentence_idx", "chunk_idx"):
        return "integer"
    elif col in ("chunk_label", "word"):
        return "ordinal"

    # Infer from data type
    dtype = df[col].dtype
    if pd.api.types.is_float_dtype(dtype):
        return "time"
    elif pd.api.types.is_integer_dtype(dtype):
        return "integer"
    else:
        return "ordinal"


def prepare_index_values(
    df: pd.DataFrame,
    index_col: str | None,
    index_type: str,
) -> tuple[np.ndarray, dict]:
    """Prepare x-axis values and formatting info.

    Parameters
    ----------
    df : pd.DataFrame
        The data frame.
    index_col : str or None
        The index column name.
    index_type : str
        The index type ("time", "integer", or "ordinal").

    Returns
    -------
    x_values : np.ndarray
        Values for x-axis.
    format_info : dict
        Formatting information with keys:
        - "xlabel": X-axis label string
        - "tickmode": Tick mode ("linear", "array")
        - "tickvals": Tick positions (for ordinal)
        - "ticktext": Tick labels (for ordinal)
    """
    format_info = {}

    if index_col is None:
        # Use row numbers
        x_values = np.arange(len(df))
        format_info["xlabel"] = "Index"
        format_info["tickmode"] = "linear"
        format_info["hoverformat"] = "d"
        return x_values, format_info

    x_values = df[index_col].values

    if index_type == "time":
        format_info["xlabel"] = "Time (s)"
        format_info["tickmode"] = "linear"
        format_info["tickformat"] = ".1f"
        format_info["hoverformat"] = ".2f"

    elif index_type == "integer":
        format_info["xlabel"] = index_col.replace("_", " ").title()
        format_info["tickmode"] = "linear"
        format_info["hoverformat"] = "d"

    elif index_type == "ordinal":
        # For ordinal data (chunk labels, words), use numeric x with text labels
        x_values = np.arange(len(df))
        labels = df[index_col].astype(str).values

        # Truncate long labels for display
        max_label_len = 25
        truncated = [
            (s[:max_label_len] + "...") if len(s) > max_label_len else s
            for s in labels
        ]

        format_info["xlabel"] = index_col.replace("_", " ").title()
        format_info["tickmode"] = "array"
        format_info["tickvals"] = x_values
        format_info["ticktext"] = truncated
        format_info["original_labels"] = labels
        format_info["hoverformat"] = ""

        # For many points, subsample tick labels
        if len(labels) > 20:
            step = max(1, len(labels) // 10)
            format_info["tickvals"] = x_values[::step]
            format_info["ticktext"] = [truncated[i] for i in range(0, len(truncated), step)]

    return x_values, format_info
