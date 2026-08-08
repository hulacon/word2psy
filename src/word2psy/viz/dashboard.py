"""Interactive HTML dashboard for exploring word2psy scores.

Mirrors viz2psy's ``--browse`` dashboard: a single self-contained HTML file
with a model selector, overview visualizations (timeseries, 2D/3D clustering,
trajectory) where every point is a word (or chunk), and a click-to-open
detail viewer. Where viz2psy shows the image, the detail view renders the
word itself in large type alongside its sentence context; a slider and
prev/next buttons browse the full set, preserving the viz2psy interface.

All heavy computation (projections, feature detection) happens here in
Python; the HTML embeds pre-computed data as JSON and renders with Plotly.js
(loaded from CDN, same version as viz2psy).
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from .feature_config import (
    FEATURE_CONFIGS,
    detect_models_in_dataframe,
    get_model_columns,
)
from .projection import compute_projection

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.27.0.min.js"

# Word-table columns that are structural rather than features
_WORD_INDEX_COLS = [
    "word_idx", "word", "sentence_idx", "chunk_idx", "chunk_label",
    "onset", "offset",
]
_CHUNK_INDEX_COLS = ["chunk_idx", "chunk_label", "n_words"]

# Core (non-sensorimotor) lexical norms shown in the detail "Norms" panel
_CORE_NORMS = [
    "concreteness", "valence", "arousal", "dominance", "age_of_acquisition",
    "imageability", "familiarity", "semantic_size", "gender_association",
    "socialness", "body_object_interaction", "zipf_frequency",
]
_SENSORIMOTOR = [
    "sensorimotor_touch", "sensorimotor_hearing", "sensorimotor_smell",
    "sensorimotor_taste", "sensorimotor_vision", "sensorimotor_interoception",
    "sensorimotor_mouth", "sensorimotor_hand", "sensorimotor_foot",
    "sensorimotor_head", "sensorimotor_torso",
]
_WORDFORM = ["length", "n_syllables", "n_phonemes", "old20", "gpt2_surprisal"]


def _values(series: pd.Series) -> list:
    """Series -> JSON-safe list (NaN -> None, numpy -> python)."""
    out = []
    for v in series:
        if isinstance(v, (float, np.floating)):
            out.append(None if np.isnan(v) else round(float(v), 5))
        elif isinstance(v, (int, np.integer)):
            out.append(int(v))
        elif v is None or (isinstance(v, float) and np.isnan(v)):
            out.append(None)
        else:
            out.append(str(v))
    return out


def _feature_ranges(df: pd.DataFrame, cols: list[str]) -> dict:
    """Dataset min/max per column, for normalized detail bars."""
    ranges = {}
    for col in cols:
        vals = pd.to_numeric(df[col], errors="coerce")
        lo, hi = vals.min(), vals.max()
        if pd.notna(lo) and pd.notna(hi):
            ranges[col] = [round(float(lo), 5), round(float(hi), 5)]
    return ranges


def _projection_payload(
    df: pd.DataFrame,
    cols: list[str],
    method: str,
    n_components: int,
) -> dict | None:
    """Compute a projection, dropping rows with any NaN (e.g. word2vec OOV).

    Returns dict with coordinate arrays, kept row indices, and axis labels,
    or None if the projection is not possible.
    """
    X = df[cols].to_numpy(dtype=float)
    valid = ~np.isnan(X).any(axis=1)
    indices = np.flatnonzero(valid)
    if len(indices) < 3:
        return None

    try:
        X_proj, info = compute_projection(
            X[valid], method=method, n_components=n_components
        )
    except Exception as e:  # pragma: no cover - degenerate inputs
        warnings.warn(f"{method} projection failed: {e}")
        return None

    payload = {
        "x": [round(float(v), 4) for v in X_proj[:, 0]],
        "y": [round(float(v), 4) for v in X_proj[:, 1]],
        "indices": indices.tolist(),
        "xlabel": info.get("xlabel", "dim 1"),
        "ylabel": info.get("ylabel", "dim 2"),
        "method": method,
    }
    if n_components == 3:
        payload["z"] = [round(float(v), 4) for v in X_proj[:, 2]]
        payload["zlabel"] = info.get("zlabel", "dim 3")
    return payload


def _timeseries_features(df: pd.DataFrame, cols: list[str], config) -> list[str]:
    """Feature order for the timeseries view (top-k by variance if configured)."""
    if config.timeseries_mode == "top_k":
        variances = [(c, float(pd.to_numeric(df[c], errors="coerce").var())) for c in cols]
        variances.sort(key=lambda t: (np.isnan(t[1]), -t[1] if not np.isnan(t[1]) else 0))
        return [c for c, _ in variances[: config.top_k]]
    return cols


def _build_model_entry(df: pd.DataFrame, name: str, mds_max: int) -> dict | None:
    """Build the payload entry for one detected model."""
    config = FEATURE_CONFIGS[name]
    cols = get_model_columns(df.columns.tolist(), name)
    if not cols:
        return None

    n = len(df)
    entry = {
        "level": config.level,
        "description": config.description,
        "nDims": len(cols),
        "featureType": config.feature_type,
        "tsFeatures": _timeseries_features(df, cols, config) if config.timeseries else [],
        "clustering": {},
        "trajectory": None,
    }

    if config.mds_clustering and len(cols) >= 2 and n >= 3:
        cluster_method = "mds" if n <= mds_max else "pca"
        proj2d = _projection_payload(df, cols, cluster_method, 2)
        if proj2d:
            entry["clustering"]["2d"] = proj2d
        if len(cols) >= 3 and n >= 4:
            proj3d = _projection_payload(df, cols, cluster_method, 3)
            if proj3d:
                entry["clustering"]["3d"] = proj3d
        # Trajectory always uses PCA (stable, fast, ordered path is the point)
        traj = _projection_payload(df, cols, "pca", 2)
        if traj:
            entry["trajectory"] = traj

    entry["available"] = {
        "timeseries": bool(entry["tsFeatures"]),
        "clustering2d": "2d" in entry["clustering"],
        "clustering3d": "3d" in entry["clustering"],
        "trajectory": entry["trajectory"] is not None,
    }
    if not any(entry["available"].values()):
        return None
    return entry


def _reconstruct_chunk_texts(words_df: pd.DataFrame | None, chunks_df: pd.DataFrame) -> list[str]:
    """Rebuild display text per chunk by joining its words."""
    if words_df is None or "chunk_idx" not in words_df.columns:
        return [""] * len(chunks_df)
    grouped = words_df.groupby("chunk_idx")["word"].apply(
        lambda s: " ".join(str(w) for w in s)
    )
    return [grouped.get(ci, "") for ci in chunks_df["chunk_idx"]]


def _detail_panels(word_cols: list[str], chunk_cols: list[str]) -> dict:
    """Configure detail-view feature panels from available columns."""
    panels = {"word": [], "chunk": []}

    core = [c for c in _CORE_NORMS if c in word_cols]
    if core:
        panels["word"].append(
            {"id": "norms", "label": "Lexical norms", "kind": "bars_norm", "features": core}
        )
    senso = [c for c in _SENSORIMOTOR if c in word_cols]
    if len(senso) >= 3:
        panels["word"].append(
            {"id": "sensorimotor", "label": "Sensorimotor (radar)", "kind": "radar", "features": senso}
        )
    wf = [c for c in _WORDFORM if c in word_cols]
    if wf:
        panels["word"].append(
            {"id": "wordform", "label": "Wordform & surprisal", "kind": "bars_norm", "features": wf}
        )

    emotions = sorted(c for c in chunk_cols if c.startswith("emotion_"))
    if emotions:
        panels["chunk"].append(
            {"id": "emotions", "label": "Emotions (GoEmotions)", "kind": "bars_sorted", "features": emotions}
        )
    sentiments = [c for c in chunk_cols if c.startswith("sentiment_")]
    if sentiments:
        panels["chunk"].append(
            {"id": "sentiment", "label": "Sentiment", "kind": "bars_prob", "features": sentiments}
        )
    readability = [c for c in chunk_cols if c.startswith("readability_")]
    if readability:
        panels["chunk"].append(
            {"id": "readability", "label": "Readability", "kind": "bars_norm", "features": readability}
        )
    agg_means = [c for c in chunk_cols if c.endswith("_mean")]
    if agg_means:
        panels["chunk"].append(
            {"id": "word_means", "label": "Word-feature means", "kind": "bars_norm", "features": agg_means}
        )
    return panels


def create_dashboard(
    words_df: pd.DataFrame | None,
    chunks_df: pd.DataFrame | None = None,
    title: str = "word2psy Dashboard",
    max_points: int = 2000,
    mds_max: int = 500,
) -> str:
    """Create the interactive dashboard as a self-contained HTML string.

    Parameters
    ----------
    words_df : pd.DataFrame or None
        Word-level scores (``*_words.csv``).
    chunks_df : pd.DataFrame or None
        Chunk-level scores (``*_chunks.csv``).
    title : str
        Page title.
    max_points : int
        Truncate tables beyond this many rows (protects file size and
        projection cost).
    mds_max : int
        Use MDS for clustering up to this many rows, PCA beyond.
    """
    if words_df is None and chunks_df is None:
        raise ValueError("Need at least one of words_df / chunks_df.")

    truncated = {}
    if words_df is not None and len(words_df) > max_points:
        truncated["words"] = len(words_df)
        words_df = words_df.iloc[:max_points].reset_index(drop=True)
    if chunks_df is not None and len(chunks_df) > max_points:
        truncated["chunks"] = len(chunks_df)
        chunks_df = chunks_df.iloc[:max_points].reset_index(drop=True)

    models = {}
    word_feature_cols: list[str] = []
    chunk_feature_cols: list[str] = []

    payload: dict = {"title": title, "models": models, "truncated": truncated}

    # --- word table ---
    if words_df is not None and len(words_df):
        detected = detect_models_in_dataframe(words_df.columns.tolist(), level="word")
        for name in detected:
            entry = _build_model_entry(words_df, name, mds_max)
            if entry:
                models[name] = entry
        for name in detected:
            for c in get_model_columns(words_df.columns.tolist(), name):
                if c not in word_feature_cols:
                    word_feature_cols.append(c)

        scalar_cols = [c for c in word_feature_cols if _is_scalar_col(c)]
        payload["words"] = {
            "word": _values(words_df["word"]),
            "sentence_idx": _values(words_df["sentence_idx"]),
            "chunk_idx": _values(words_df["chunk_idx"]),
            "chunk_label": _values(words_df["chunk_label"]),
            "features": {c: _values(words_df[c]) for c in scalar_cols},
        }
        payload.setdefault("ranges", {}).update(
            _feature_ranges(words_df, scalar_cols)
        )

    # --- chunk table ---
    if chunks_df is not None and len(chunks_df):
        detected = detect_models_in_dataframe(chunks_df.columns.tolist(), level="chunk")
        for name in detected:
            entry = _build_model_entry(chunks_df, name, mds_max)
            if entry:
                models[name] = entry
        for name in detected:
            for c in get_model_columns(chunks_df.columns.tolist(), name):
                if c not in chunk_feature_cols:
                    chunk_feature_cols.append(c)

        scalar_cols = [c for c in chunk_feature_cols if _is_scalar_col(c)]
        extra_cols = [
            c for c in chunks_df.columns
            if c not in _CHUNK_INDEX_COLS and c not in chunk_feature_cols
        ]
        payload["chunks"] = {
            "chunk_idx": _values(chunks_df["chunk_idx"]),
            "chunk_label": _values(chunks_df["chunk_label"]),
            "n_words": _values(chunks_df["n_words"]) if "n_words" in chunks_df.columns else [],
            "text": _reconstruct_chunk_texts(words_df, chunks_df),
            "features": {c: _values(chunks_df[c]) for c in scalar_cols},
            "extra": {c: _values(chunks_df[c]) for c in extra_cols},
        }
        payload.setdefault("ranges", {}).update(
            _feature_ranges(chunks_df, scalar_cols)
        )

    if not models:
        raise ValueError(
            "No word2psy model outputs detected in the provided CSV(s). "
            "Expected columns like 'concreteness', 'clip_text_000', ..."
        )

    payload["panels"] = _detail_panels(word_feature_cols, chunk_feature_cols)

    payload_json = json.dumps(payload, separators=(",", ":"))
    html = (
        _HTML_TEMPLATE
        .replace("__TITLE__", title)
        .replace("__PLOTLY_CDN__", PLOTLY_CDN)
        .replace("__PAYLOAD__", payload_json)
    )
    return html


def _is_scalar_col(col: str) -> bool:
    """True if a feature column is scalar (not an embedding dimension)."""
    tail = col.rsplit("_", 1)[-1]
    return not (len(tail) == 3 and tail.isdigit())


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<script src="__PLOTLY_CDN__"></script>
<style>
* { box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
body { margin: 0; padding: 20px; background: #f5f5f5; }
.container { max-width: 1100px; margin: 0 auto; }
h1 { color: #333; margin-bottom: 5px; }
.subtitle { color: #666; margin-bottom: 20px; }
.controls { display: flex; gap: 20px; margin-bottom: 20px; flex-wrap: wrap; }
.control-group { background: white; padding: 15px 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
.control-group > label { display: block; font-weight: 600; margin-bottom: 8px; color: #333; }
select { padding: 8px 12px; font-size: 14px; border: 1px solid #ddd; border-radius: 4px; min-width: 200px; }
.viz-buttons { display: flex; gap: 8px; }
.viz-btn { padding: 8px 16px; font-size: 14px; border: 2px solid #ddd; border-radius: 4px; background: white; cursor: pointer; transition: all 0.2s; }
.viz-btn:hover { border-color: #007bff; }
.viz-btn.active { background: #007bff; color: white; border-color: #007bff; }
.viz-btn.disabled { opacity: 0.4; cursor: not-allowed; }
.browse-btn { padding: 8px 16px; font-size: 14px; border: 2px solid #4caf50; border-radius: 4px; background: #e8f4e8; cursor: pointer; transition: all 0.2s; display: block; width: 100%; margin-bottom: 6px; }
.browse-btn:hover { background: #c8e6c9; border-color: #388e3c; }
.sub-options { display: none; margin-top: 10px; padding-top: 10px; border-top: 1px solid #eee; }
.sub-options.visible { display: block; }
.sub-toggle { display: inline-flex; background: #f0f0f0; border-radius: 4px; overflow: hidden; margin-right: 12px; }
.sub-toggle button { padding: 6px 12px; font-size: 12px; border: none; background: transparent; cursor: pointer; }
.sub-toggle button.active { background: #007bff; color: white; }
.toggle-label { display: inline-flex; align-items: center; gap: 6px; cursor: pointer; font-size: 13px; }
.model-info { font-size: 12px; color: #666; margin-top: 5px; max-width: 240px; }
.plot-container { background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); padding: 20px; }
#plot { width: 100%; height: 600px; }
.warning { display: none; text-align: center; padding: 60px 20px; color: #666; }
.warning .emoji { font-size: 64px; margin-bottom: 20px; }
.warning .message { font-size: 18px; margin-bottom: 10px; }
.warning .detail { font-size: 14px; color: #999; }
.footnote { color: #999; font-size: 12px; margin-top: 12px; }

/* Detail overlay (the "single image viewer" analog) */
#detail-overlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.45); z-index: 100; }
#detail-overlay.open { display: flex; align-items: center; justify-content: center; }
.detail-box { background: #f5f5f5; border-radius: 10px; width: min(1060px, 94vw); max-height: 92vh; overflow: auto; padding: 20px 24px; position: relative; }
.detail-close { position: absolute; top: 12px; right: 14px; border: none; background: transparent; font-size: 22px; cursor: pointer; color: #666; }
.detail-close:hover { color: #000; }
.detail-content { display: flex; gap: 20px; flex-wrap: wrap; }
.stimulus-panel { background: white; border-radius: 8px; padding: 24px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); flex: 1; min-width: 300px; display: flex; flex-direction: column; }
.stimulus-word { font-size: 56px; font-weight: 700; color: #1a1a2e; text-align: center; margin: 30px 0 10px 0; word-break: break-word; }
.stimulus-label { font-size: 26px; font-weight: 700; color: #1a1a2e; text-align: center; margin: 20px 0 10px 0; }
.stimulus-meta { text-align: center; color: #888; font-size: 13px; margin-bottom: 18px; }
.stimulus-context { background: #f8f9fb; border-radius: 6px; padding: 14px 16px; color: #444; font-size: 15px; line-height: 1.6; margin-top: auto; max-height: 220px; overflow: auto; }
.stimulus-context mark { background: #ffe58a; border-radius: 3px; padding: 0 3px; font-weight: 600; }
.features-panel { background: white; border-radius: 8px; padding: 16px 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); flex: 1.3; min-width: 380px; }
.features-panel select { margin-bottom: 6px; }
#detail-plot { width: 100%; height: 420px; }
.detail-nav { display: flex; align-items: center; gap: 12px; margin-top: 16px; background: white; border-radius: 8px; padding: 12px 16px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
.detail-nav button { padding: 6px 14px; font-size: 15px; border: 1px solid #ccc; border-radius: 4px; background: white; cursor: pointer; }
.detail-nav button:hover { border-color: #007bff; }
.detail-nav input[type="range"] { flex: 1; }
.detail-nav .pos { font-size: 13px; color: #555; min-width: 110px; text-align: right; }
</style>
</head>
<body>
<div class="container">
  <h1>__TITLE__</h1>
  <p class="subtitle" id="subtitle"></p>

  <div class="controls">
    <div class="control-group">
      <label>Model Output</label>
      <select id="model-select" onchange="updatePlot()"></select>
      <div id="model-info" class="model-info"></div>
    </div>

    <div class="control-group">
      <label>Visualization</label>
      <div class="viz-buttons">
        <button class="viz-btn active" data-viz="timeseries" onclick="selectViz('timeseries')">&#128200; Time Series</button>
        <button class="viz-btn" data-viz="clustering" onclick="selectViz('clustering')">&#128309; Clustering</button>
        <button class="viz-btn" data-viz="trajectory" onclick="selectViz('trajectory')">&#128640; Trajectory</button>
      </div>
      <div id="sub-timeseries" class="sub-options">
        <label class="toggle-label"><input type="checkbox" id="rolling-toggle" onchange="updatePlot()"><span>Smooth (rolling avg)</span></label>
      </div>
      <div id="sub-clustering" class="sub-options">
        <div class="sub-toggle">
          <button class="active" data-sub="2d" onclick="selectSubOption('clustering','2d')">2D</button>
          <button data-sub="3d" onclick="selectSubOption('clustering','3d')">3D</button>
        </div>
        <label class="toggle-label"><input type="checkbox" id="labels-toggle" onchange="updatePlot()"><span>Show labels</span></label>
      </div>
      <div id="sub-trajectory" class="sub-options"></div>
    </div>

    <div class="control-group" id="browse-group">
      <label>Browser</label>
    </div>
  </div>

  <div class="plot-container">
    <div id="plot"></div>
    <div id="warning" class="warning">
      <div class="emoji">&#128533;</div>
      <div class="message" id="warning-message">Not available</div>
      <div class="detail" id="warning-detail"></div>
    </div>
  </div>
  <div class="footnote" id="footnote"></div>
</div>

<div id="detail-overlay" onclick="if(event.target===this)closeDetail()">
  <div class="detail-box">
    <button class="detail-close" onclick="closeDetail()">&#10005;</button>
    <div class="detail-content">
      <div class="stimulus-panel" id="stimulus-panel"></div>
      <div class="features-panel">
        <select id="panel-select" onchange="renderDetailPanel()"></select>
        <div id="detail-plot"></div>
      </div>
    </div>
    <div class="detail-nav">
      <button onclick="stepDetail(-1)">&#9664;</button>
      <input type="range" id="detail-slider" min="0" max="0" value="0" oninput="jumpDetail(parseInt(this.value))">
      <button onclick="stepDetail(1)">&#9654;</button>
      <span class="pos" id="detail-pos"></span>
    </div>
  </div>
</div>

<script>
const DATA = __PAYLOAD__;

const COLORS = { bar: '#636EFA', prob: '#00CC96', sorted: '#EF553B', radar: 'rgba(99,110,250,0.35)' };
let currentViz = 'timeseries';
let subOptions = { clustering: '2d' };
let detail = { level: null, idx: 0, panel: null };

const WORDS = DATA.words || null;
const CHUNKS = DATA.chunks || null;
const nWords = WORDS ? WORDS.word.length : 0;
const nChunks = CHUNKS ? CHUNKS.chunk_label.length : 0;

function prettify(name) {
  return name.replace(/^(sensorimotor|emotion|sentiment|readability)_/, '').replace(/_/g, ' ');
}
function rowLabel(level, i) {
  return level === 'word' ? WORDS.word[i] : String(CHUNKS.chunk_label[i]);
}
function nRows(level) { return level === 'word' ? nWords : nChunks; }
function tableFeatures(level) { return level === 'word' ? WORDS.features : CHUNKS.features; }

// ---------- init ----------
function init() {
  const sel = document.getElementById('model-select');
  const groups = { word: [], chunk: [] };
  for (const [name, m] of Object.entries(DATA.models)) groups[m.level].push(name);
  for (const [level, names] of Object.entries(groups)) {
    if (!names.length) continue;
    const og = document.createElement('optgroup');
    og.label = level === 'word' ? 'Word-level' : 'Chunk-level';
    for (const name of names) {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name + ' (' + DATA.models[name].nDims + 'd)';
      og.appendChild(opt);
    }
    sel.appendChild(og);
  }

  const parts = [];
  if (nWords) parts.push(nWords + ' words');
  if (nChunks) parts.push(nChunks + ' chunks');
  parts.push(Object.keys(DATA.models).length + ' model outputs');
  document.getElementById('subtitle').textContent =
    parts.join(' · ') + ' — click any point to inspect it.';

  const bg = document.getElementById('browse-group');
  if (nWords && DATA.panels.word.length) {
    const b = document.createElement('button');
    b.className = 'browse-btn';
    b.innerHTML = '&#128292; Browse words';
    b.onclick = () => openDetail('word', 0);
    bg.appendChild(b);
  }
  if (nChunks && DATA.panels.chunk.length) {
    const b = document.createElement('button');
    b.className = 'browse-btn';
    b.innerHTML = '&#128196; Browse chunks';
    b.onclick = () => openDetail('chunk', 0);
    bg.appendChild(b);
  }

  const trunc = [];
  for (const [tbl, total] of Object.entries(DATA.truncated || {}))
    trunc.push('showing first ' + (tbl === 'words' ? nWords : nChunks) + ' of ' + total + ' ' + tbl);
  document.getElementById('footnote').textContent = trunc.join('; ');

  document.getElementById('sub-timeseries').classList.add('visible');
  const defaultLabels = document.getElementById('labels-toggle');
  defaultLabels.checked = true;
  updatePlot();
  document.addEventListener('keydown', e => {
    if (!document.getElementById('detail-overlay').classList.contains('open')) return;
    if (e.key === 'Escape') closeDetail();
    if (e.key === 'ArrowLeft') stepDetail(-1);
    if (e.key === 'ArrowRight') stepDetail(1);
  });
}

// ---------- overview plots ----------
function selectViz(viz) {
  currentViz = viz;
  document.querySelectorAll('.viz-btn').forEach(b => b.classList.toggle('active', b.dataset.viz === viz));
  for (const v of ['timeseries', 'clustering', 'trajectory'])
    document.getElementById('sub-' + v).classList.toggle('visible', v === viz);
  updatePlot();
}
function selectSubOption(viz, sub) {
  subOptions[viz] = sub;
  document.querySelectorAll('#sub-' + viz + ' .sub-toggle button')
    .forEach(b => b.classList.toggle('active', b.dataset.sub === sub));
  updatePlot();
}

const VIZ_WARNINGS = {
  timeseries: ['Time series not available for this model',
               'High-dimensional embeddings are not interpretable as individual time series.'],
  clustering: ['Clustering not available for this model',
               'This model’s output cannot be projected to 2D/3D.'],
  trajectory: ['Trajectory not available for this model',
               'This model’s output cannot show state-space evolution.'],
};

function rollingMean(values, w) {
  const half = Math.floor(w / 2), out = [];
  for (let i = 0; i < values.length; i++) {
    let s = 0, n = 0;
    for (let j = Math.max(0, i - half); j <= Math.min(values.length - 1, i + half); j++) {
      const v = values[j];
      if (v !== null && v !== undefined) { s += v; n++; }
    }
    out.push(n ? s / n : null);
  }
  return out;
}

function hoverText(level, i) {
  if (level === 'word')
    return WORDS.word[i] + '<br>word ' + i + ' · sentence ' + WORDS.sentence_idx[i] +
           ' · ' + WORDS.chunk_label[i];
  let t = String(CHUNKS.chunk_label[i]);
  for (const [k, vals] of Object.entries(CHUNKS.extra || {})) t += '<br>' + k + ': ' + vals[i];
  return t;
}

function updatePlot() {
  const modelName = document.getElementById('model-select').value;
  const model = DATA.models[modelName];
  const level = model.level;
  document.getElementById('model-info').textContent = model.nDims + ' dimensions — ' + model.description;

  document.querySelectorAll('.viz-btn').forEach(btn => {
    const v = btn.dataset.viz;
    const ok = v === 'timeseries' ? model.available.timeseries
             : v === 'clustering' ? (model.available.clustering2d || model.available.clustering3d)
             : model.available.trajectory;
    btn.classList.toggle('disabled', !ok);
  });

  const plotDiv = document.getElementById('plot');
  const warnDiv = document.getElementById('warning');
  let traces = null, layout = null;

  if (currentViz === 'timeseries' && model.available.timeseries) {
    [traces, layout] = buildTimeseries(model, level);
  } else if (currentViz === 'clustering') {
    const key = subOptions.clustering;
    const proj = model.clustering[key];
    if (proj) [traces, layout] = buildScatter(proj, level, key === '3d', false, modelName);
  } else if (currentViz === 'trajectory' && model.available.trajectory) {
    [traces, layout] = buildScatter(model.trajectory, level, false, true, modelName);
  }

  if (!traces) {
    plotDiv.style.display = 'none';
    warnDiv.style.display = 'block';
    const [msg, det] = VIZ_WARNINGS[currentViz];
    document.getElementById('warning-message').textContent = msg;
    document.getElementById('warning-detail').textContent = det;
    return;
  }
  plotDiv.style.display = 'block';
  warnDiv.style.display = 'none';
  Plotly.react('plot', traces, layout, { responsive: true }).then(attachClickHandler);
}

function buildTimeseries(model, level) {
  const feats = model.tsFeatures;
  const featureTable = tableFeatures(level);
  const n = nRows(level);
  const x = [...Array(n).keys()];
  const smooth = document.getElementById('rolling-toggle').checked;
  const w = Math.max(2, Math.min(10, Math.floor(n / 3)));
  const customdata = x;

  const traces = feats.map(f => {
    let y = featureTable[f];
    if (smooth) y = rollingMean(y, w);
    return {
      type: 'scatter',
      mode: smooth ? 'lines' : 'lines+markers',
      name: prettify(f),
      x: x, y: y,
      customdata: customdata,
      marker: { size: 5, opacity: 0.75 },
      line: smooth ? { width: 3 } : { width: 1.5 },
      hovertemplate: prettify(f) + '<br>%{text}<br>value: %{y:.4g}<extra></extra>',
      text: x.map(i => rowLabel(level, i)),
    };
  });

  const layout = {
    title: (level === 'word' ? 'Word' : 'Chunk') + ' sequence — ' + prettifyModelTitle(),
    xaxis: { title: level === 'word' ? 'Word index' : 'Chunk index' },
    yaxis: { title: 'Value' },
    hovermode: 'closest',
    margin: { t: 50, r: 30, b: 60, l: 60 },
  };
  if (level === 'chunk' && n <= 30) {
    layout.xaxis.tickmode = 'array';
    layout.xaxis.tickvals = x;
    layout.xaxis.ticktext = x.map(i => truncate(String(CHUNKS.chunk_label[i]), 14));
  }
  return [traces, layout];
}

function prettifyModelTitle() {
  return document.getElementById('model-select').value;
}
function truncate(s, n) { return s.length > n ? s.slice(0, n) + '…' : s; }

function buildScatter(proj, level, is3d, asTrajectory, modelName) {
  const idx = proj.indices;
  const colors = [...Array(idx.length).keys()];
  const showLabels = !asTrajectory && document.getElementById('labels-toggle').checked
                     && idx.length <= 300;
  const labels = idx.map(i => truncate(rowLabel(level, i), 14));

  const traces = [];
  if (asTrajectory) {
    traces.push({
      type: 'scatter', mode: 'lines',
      x: proj.x, y: proj.y,
      line: { color: 'rgba(100,100,100,0.45)', width: 1 },
      hoverinfo: 'skip', showlegend: false,
    });
  }
  const marker = {
    size: is3d ? 5 : (showLabels ? 8 : 10),
    color: colors,
    colorscale: 'Viridis',
    colorbar: { title: 'Position' },
    opacity: 0.85,
  };
  const main = {
    type: is3d ? 'scatter3d' : 'scatter',
    mode: showLabels ? 'markers+text' : 'markers',
    x: proj.x, y: proj.y,
    marker: marker,
    customdata: idx,
    text: labels,
    textposition: 'top center',
    textfont: { size: 11, color: '#333' },
    hovertemplate: idx.map(i => hoverText(level, i) + '<extra></extra>'),
    showlegend: false,
  };
  if (is3d) main.z = proj.z;
  traces.push(main);

  if (asTrajectory && idx.length > 1) {
    traces.push({ type: 'scatter', mode: 'markers', x: [proj.x[0]], y: [proj.y[0]],
      marker: { size: 16, color: 'green', symbol: 'circle-open', line: { width: 3 } },
      name: 'Start', hovertemplate: 'Start<extra></extra>' });
    traces.push({ type: 'scatter', mode: 'markers',
      x: [proj.x[proj.x.length - 1]], y: [proj.y[proj.y.length - 1]],
      marker: { size: 16, color: 'red', symbol: 'square-open', line: { width: 3 } },
      name: 'End', hovertemplate: 'End<extra></extra>' });
  }

  const methodLabel = proj.method.toUpperCase();
  const title = modelName + ' — ' + (asTrajectory
    ? 'state-space trajectory (' + methodLabel + ')'
    : methodLabel + ' ' + (is3d ? '3D' : '2D') + ' projection');
  const layout = { title: title, hovermode: 'closest', margin: { t: 50, r: 30, b: 60, l: 60 },
                   showlegend: asTrajectory };
  if (is3d) {
    layout.scene = { xaxis: { title: proj.xlabel }, yaxis: { title: proj.ylabel },
                     zaxis: { title: proj.zlabel } };
  } else {
    layout.xaxis = { title: proj.xlabel };
    layout.yaxis = { title: proj.ylabel };
  }
  return [traces, layout];
}

function attachClickHandler() {
  const plotDiv = document.getElementById('plot');
  plotDiv.on('plotly_click', data => {
    if (!data.points || !data.points.length) return;
    const p = data.points[0];
    let idx = (p.customdata !== undefined && p.customdata !== null) ? p.customdata : p.pointIndex;
    if (Array.isArray(idx)) idx = idx[0];
    const model = DATA.models[document.getElementById('model-select').value];
    if (typeof idx === 'number') openDetail(model.level, idx);
  });
}

// ---------- detail viewer ----------
function openDetail(level, idx) {
  const panels = DATA.panels[level];
  if (!panels.length) return;
  const keepPanel = detail.level === level && detail.panel;
  detail.level = level;
  detail.idx = idx;
  detail.panel = keepPanel || panels[0].id;

  const sel = document.getElementById('panel-select');
  sel.innerHTML = '';
  for (const p of panels) {
    const opt = document.createElement('option');
    opt.value = p.id;
    opt.textContent = p.label;
    if (p.id === detail.panel) opt.selected = true;
    sel.appendChild(opt);
  }
  const slider = document.getElementById('detail-slider');
  slider.max = nRows(level) - 1;
  slider.value = idx;

  document.getElementById('detail-overlay').classList.add('open');
  renderDetail();
}
function closeDetail() { document.getElementById('detail-overlay').classList.remove('open'); }
function stepDetail(d) {
  jumpDetail(Math.min(nRows(detail.level) - 1, Math.max(0, detail.idx + d)));
}
function jumpDetail(idx) {
  detail.idx = idx;
  document.getElementById('detail-slider').value = idx;
  renderDetail();
}

function escapeHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function renderDetail() {
  const { level, idx } = detail;
  const panel = document.getElementById('stimulus-panel');

  if (level === 'word') {
    const w = WORDS.word[idx];
    let context = '';
    const si = WORDS.sentence_idx[idx], ci = WORDS.chunk_idx[idx];
    const parts = [];
    for (let j = 0; j < nWords; j++) {
      if (WORDS.sentence_idx[j] === si && WORDS.chunk_idx[j] === ci) {
        parts.push(j === idx ? '<mark>' + escapeHtml(WORDS.word[j]) + '</mark>' : escapeHtml(WORDS.word[j]));
      }
    }
    context = parts.join(' ');
    panel.innerHTML =
      '<div class="stimulus-word">' + escapeHtml(w) + '</div>' +
      '<div class="stimulus-meta">word ' + (idx + 1) + ' of ' + nWords +
      ' · sentence ' + si + ' · ' + escapeHtml(String(WORDS.chunk_label[idx])) + '</div>' +
      '<div class="stimulus-context">' + context + '</div>';
  } else {
    const meta = ['chunk ' + (idx + 1) + ' of ' + nChunks];
    if (CHUNKS.n_words.length) meta.push(CHUNKS.n_words[idx] + ' words');
    for (const [k, vals] of Object.entries(CHUNKS.extra || {})) meta.push(k + ': ' + vals[idx]);
    panel.innerHTML =
      '<div class="stimulus-label">' + escapeHtml(String(CHUNKS.chunk_label[idx])) + '</div>' +
      '<div class="stimulus-meta">' + escapeHtml(meta.join(' · ')) + '</div>' +
      '<div class="stimulus-context">' + escapeHtml(CHUNKS.text[idx] || '(text unavailable)') + '</div>';
  }

  document.getElementById('detail-pos').textContent =
    (level === 'word' ? 'word ' : 'chunk ') + (idx + 1) + ' / ' + nRows(level);
  renderDetailPanel();
}

function normValue(feature, v) {
  const r = DATA.ranges[feature];
  if (v === null || v === undefined || !r || r[1] <= r[0]) return null;
  return Math.max(0, Math.min(1, (v - r[0]) / (r[1] - r[0])));
}

function renderDetailPanel() {
  detail.panel = document.getElementById('panel-select').value;
  const { level, idx } = detail;
  const cfg = DATA.panels[level].find(p => p.id === detail.panel);
  const featureTable = tableFeatures(level);
  const feats = cfg.features.filter(f => featureTable[f] !== undefined);
  const raw = feats.map(f => featureTable[f][idx]);
  let traces, layout;

  if (cfg.kind === 'radar') {
    const labels = feats.map(prettify);
    traces = [{
      type: 'scatterpolar',
      r: [...raw, raw[0]],
      theta: [...labels, labels[0]],
      fill: 'toself', fillcolor: COLORS.radar,
      line: { color: '#636EFA' },
      hovertemplate: '%{theta}: %{r:.3f}<extra></extra>',
    }];
    layout = { polar: { radialaxis: { visible: true } }, margin: { t: 30, b: 30, l: 60, r: 60 } };
  } else if (cfg.kind === 'bars_sorted' || cfg.kind === 'bars_prob') {
    let pairs = feats.map((f, i) => [prettify(f), raw[i]]);
    if (cfg.kind === 'bars_sorted') pairs.sort((a, b) => (b[1] ?? -1) - (a[1] ?? -1));
    traces = [{
      type: 'bar', orientation: 'h',
      y: pairs.map(p => p[0]), x: pairs.map(p => p[1]),
      marker: { color: cfg.kind === 'bars_prob' ? COLORS.prob : COLORS.sorted },
      hovertemplate: '%{y}: %{x:.3f}<extra></extra>',
    }];
    layout = {
      xaxis: { title: 'Probability', range: [0, 1] },
      yaxis: { autorange: 'reversed', automargin: true },
      margin: { t: 20, r: 20, b: 50, l: 10 },
    };
  } else {
    // bars_norm: dataset-normalized values, raw in hover
    const hover = feats.map((f, i) => {
      const r = DATA.ranges[f] || [null, null];
      const v = raw[i];
      return prettify(f) + ': ' + (v === null || v === undefined ? 'n/a' : v.toPrecision(4)) +
             (r[0] !== null ? ' (dataset range ' + r[0] + ' – ' + r[1] + ')' : '');
    });
    traces = [{
      type: 'bar', orientation: 'h',
      y: feats.map(prettify),
      x: feats.map((f, i) => normValue(f, raw[i])),
      marker: { color: COLORS.bar },
      customdata: hover,
      hovertemplate: '%{customdata}<extra></extra>',
    }];
    layout = {
      xaxis: { title: 'Value (normalized to dataset range)', range: [0, 1] },
      yaxis: { autorange: 'reversed', automargin: true },
      margin: { t: 20, r: 20, b: 50, l: 10 },
    };
  }
  layout.height = Math.max(360, feats.length * 22 + 120);
  Plotly.react('detail-plot', traces, layout, { displayModeBar: false, responsive: true });
}

document.addEventListener('DOMContentLoaded', init);
</script>
</body>
</html>
"""
