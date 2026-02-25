"""Visualization module for word2psy.

Provides static matplotlib/seaborn visualizations for text feature data:

- ``plot_timeseries``: Feature values over word/sentence/chunk sequence
- ``plot_heatmap``: Correlation heatmap of scalar features
- ``plot_scatter``: 2D projection scatter (PCA, UMAP, t-SNE, MDS)
- ``compute_projection``: Dimensionality reduction (PCA, PPCA, UMAP, t-SNE, MDS)
- ``get_visualization_recommendations``: Auto-suggest visualizations for a DataFrame

Imports are deferred until access to avoid loading heavy dependencies at startup.
"""

_LAZY_IMPORTS = {
    "plot_timeseries": ".timeseries",
    "plot_heatmap": ".heatmap",
    "plot_scatter": ".scatter",
    "compute_projection": ".projection",
    "PROJECTION_METHODS": ".projection",
    "get_visualization_recommendations": ".feature_config",
    "detect_models_in_dataframe": ".feature_config",
    "get_timeseries_features": ".feature_config",
    "get_mds_features": ".feature_config",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(_LAZY_IMPORTS.keys())
