"""Binder et al. (2016) experiential-attribute probes on fastText: training and weights.

One ridge probe per attribute, from the word's fastText ``crawl-300d-2M-subword``
vector (``get_word_vector``, exactly as ``lexical_norms`` reads it) to the
attribute's 0-6 rating. Features are standardised; alpha is chosen per
attribute by 5-fold CV (pooled squared error) on a log grid; the
standardisation is folded into ``W`` and ``b``. Each attribute trains on the
words that rated it (the authors did not rate verbs and adjectives on
Complexity, nor adjectives on Practice and Caused).

The ratings are user-supplied (``word2psy.norms.binder``), so the weights are
not package data. Build them once::

    python scripts/train_experiential.py --ratings WordSet1_Ratings.xlsx \\
        [--attributes keep.txt] [--out experiential_v1.npz]

and point ``$WORD2PSY_EXPERIENTIAL_WEIGHTS`` at the file, or leave it at
``<word2psy cache>/models/experiential_v1.npz``. ``--attributes`` restricts the
probe to a subset (one attribute per line); the model emits exactly the
attributes its weights hold. The weights record the ratings file's sha256.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from word2psy.exceptions import NormDataError
from word2psy.norms import binder
from word2psy.norms.train import MODELS_DIR

BACKBONE = "crawl-300d-2M-subword"  # FastTextModel.checkpoint
EMBED_DIM = 300
PREFIX = "lexical_norms_experiential_"
#: Every column the probe can emit, in the spreadsheet's attribute order.
COLUMNS = tuple(PREFIX + binder.column_name(a) for a in binder.ATTRIBUTES)

WEIGHTS_ENV = "WORD2PSY_EXPERIENTIAL_WEIGHTS"
WEIGHTS_NAME = "experiential_v1.npz"
WEIGHTS_FORMAT = "1"

ALPHAS = np.logspace(-2, 6, 17)
N_INNER = 5
SEED = 0


def weights_path() -> Path:
    """``$WORD2PSY_EXPERIENTIAL_WEIGHTS``, else ``<word2psy cache>/models/experiential_v1.npz``."""
    env = os.environ.get(WEIGHTS_ENV)
    return Path(env) if env else MODELS_DIR / WEIGHTS_NAME


def _how_to_build() -> str:
    return (f"Build them with scripts/train_experiential.py --ratings <WordSet1_Ratings.xlsx> "
            f"and set ${WEIGHTS_ENV} to the file, or place it at "
            f"<word2psy cache>/models/{WEIGHTS_NAME}. The ratings are user-supplied: "
            f"{binder._where_to_get_it()}")


def read_weights(path: str | os.PathLike) -> dict:
    """Load and validate a weights file written by :func:`train`.

    Raises ``NormDataError`` naming the path and how to build it: the model
    must fail loudly without its weights, never emit a silent default.
    """
    path = Path(path)
    if not path.is_file():
        raise NormDataError(binder.NORM_NAME, f"experiential weights not found at {path}. "
                                              f"{_how_to_build()}")
    with np.load(path, allow_pickle=False) as z:
        w = {k: z[k] for k in z.files}
    fmt = str(w.get("format_version", ""))
    if fmt != WEIGHTS_FORMAT:
        raise NormDataError(binder.NORM_NAME, f"{path}: weights format {fmt!r}, expected {WEIGHTS_FORMAT!r}")
    if str(w["backbone"]) != BACKBONE:
        raise NormDataError(binder.NORM_NAME, f"{path}: trained on {w['backbone']!s}, this model reads {BACKBONE}")
    cols = tuple(str(c) for c in w["columns"])
    unknown = [c for c in cols if c not in COLUMNS]
    if not cols or unknown or len(set(cols)) != len(cols):
        raise NormDataError(binder.NORM_NAME, f"{path}: bad column list {list(cols)[:5]} (unknown {unknown[:5]})")
    if w["W"].shape != (EMBED_DIM, len(cols)) or w["b"].shape != (len(cols),):
        raise NormDataError(binder.NORM_NAME, f"{path}: W {w['W'].shape} / b {w['b'].shape} do not fit "
                                              f"{EMBED_DIM} -> {len(cols)}")
    w["columns"] = cols
    return w


# ---------------------------------------------------------------------------
# the probe fit (shared form with viz2psy's experiential trainer)
# ---------------------------------------------------------------------------

def group_folds(n_groups: int, n_splits: int, seed: int) -> np.ndarray:
    """A shuffled, balanced fold index per group."""
    f = np.arange(n_groups) % n_splits
    np.random.default_rng(seed).shuffle(f)
    return f


def _standardise(X):
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    return mu, sd


def _ridge_all_alphas(Xtr, Ytr, Xte, alphas):
    """Predictions on ``Xte`` for every alpha: (n_alpha, n_te, n_cols). Standardises on Xtr."""
    mu, sd = _standardise(Xtr)
    U, s, Vt = np.linalg.svd((Xtr - mu) / sd, full_matrices=False)
    A = ((Xte - mu) / sd) @ Vt.T
    ym = Ytr.mean(0)
    UtY = U.T @ (Ytr - ym)
    return np.stack([A @ ((s / (s ** 2 + a))[:, None] * UtY) + ym for a in alphas])


def fit_probes(X: np.ndarray, Y: np.ndarray, groups: np.ndarray | None = None, *,
               alphas: np.ndarray = ALPHAS, n_inner: int = N_INNER, seed: int = SEED) -> dict:
    """Ridge per column of ``Y`` (NaN = not rated: that column skips the row).

    Alpha per column by ``n_inner``-fold CV grouped by ``groups`` (default: one
    group per row), pooled squared error, on standardised ``X``. Returns
    ``W`` (d x c), ``b`` (c,) with the standardisation folded in, so a
    prediction is ``X @ W + b``; ``alpha`` (c,), ``alpha_at_edge`` (c,) and
    ``n_train`` (c,).
    """
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    n, d = X.shape
    c = Y.shape[1]
    groups = np.arange(n) if groups is None else np.asarray(groups)
    W, b = np.zeros((d, c)), np.zeros(c)
    alpha, n_train = np.zeros(c), np.zeros(c, int)
    # columns sharing a missingness pattern share their SVDs
    patterns: dict[bytes, list[int]] = {}
    for j in range(c):
        patterns.setdefault(np.isfinite(Y[:, j]).tobytes(), []).append(j)
    for key, cols in patterns.items():
        rows = np.flatnonzero(np.frombuffer(key, dtype=bool))
        Xr, Yr = X[rows], Y[np.ix_(rows, cols)]
        gu, gi = np.unique(groups[rows], return_inverse=True)
        fold = group_folds(len(gu), n_inner, seed)[gi]
        sse = np.zeros((len(alphas), len(cols)))
        for k in range(n_inner):
            tr, te = fold != k, fold == k
            P = _ridge_all_alphas(Xr[tr], Yr[tr], Xr[te], alphas)
            sse += ((P - Yr[te][None]) ** 2).sum(1)
        a = alphas[sse.argmin(0)]
        mu, sd = _standardise(Xr)
        U, s, Vt = np.linalg.svd((Xr - mu) / sd, full_matrices=False)
        ym = Yr.mean(0)
        UtY = U.T @ (Yr - ym)
        beta = Vt.T @ ((s[:, None] / (s[:, None] ** 2 + a[None, :])) * UtY)  # d x cols, standardised space
        Wc = beta / sd[:, None]
        W[:, cols] = Wc
        b[cols] = ym - mu @ Wc
        alpha[cols] = a
        n_train[cols] = len(rows)
    edge = (alpha == alphas[0]) | (alpha == alphas[-1])
    return {"W": W, "b": b, "alpha": alpha, "alpha_at_edge": edge, "n_train": n_train}


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def _resolve_attributes(attributes) -> list[str]:
    """Attribute names (``vision``, ``Vision`` or the full column name) -> returned-table names."""
    known = [binder.column_name(a) for a in binder.ATTRIBUTES]
    if attributes is None:
        return known
    out = []
    for a in attributes:
        a = str(a).strip()
        if not a or a.startswith("#"):
            continue
        a = a.removeprefix(PREFIX).lower()
        if a not in known:
            raise NormDataError(binder.NORM_NAME, f"unknown Binder attribute {a!r}; known: {known}")
        if a not in out:
            out.append(a)
    if not out:
        raise NormDataError(binder.NORM_NAME, "empty attribute list")
    return [a for a in known if a in out]  # spreadsheet order


def train(ratings: str | os.PathLike | None = None, *, attributes=None,
          out: str | os.PathLike | None = None, ft_model=None, quiet: bool = False) -> Path:
    """Fit the probes on the Binder ratings and write the weights file; returns its path."""
    import word2psy
    from word2psy.norms.train import load_fasttext

    df = binder.load_binder(ratings)
    attrs = _resolve_attributes(attributes)
    ft = ft_model if ft_model is not None else load_fasttext()
    words = df["word"].tolist()
    X = np.array([ft.get_word_vector(w) for w in words], dtype=np.float64)
    ok = np.linalg.norm(X, axis=1) > 0
    X, Y = X[ok], df.loc[ok, attrs].to_numpy(np.float64)
    fit = fit_probes(X, Y)
    out = Path(out) if out is not None else weights_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    trained_on = {
        "ratings": "Binder et al. (2016) WordSet1_Ratings.xlsx", "citation": binder.CITATION,
        "ratings_sha256": df.attrs["sha256"], "n_words": int(ok.sum()),
        "dropped_zero_vectors": [w for w, k in zip(words, ok) if not k],
        "duplicate_rule": "a word listed twice keeps the row with the most rated attributes",
        "features": f"fastText {BACKBONE} get_word_vector, standardised (folded into W, b)",
        "alpha_grid": ALPHAS.tolist(), "alpha_selection": f"{N_INNER}-fold CV, seed {SEED}, pooled SSE",
        "word2psy": word2psy.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(out, "wb") as f:  # a file handle: np.savez would append ".npz" to any other name
        np.savez(f, format_version=np.array(WEIGHTS_FORMAT), backbone=np.array(BACKBONE),
                 columns=np.array([PREFIX + a for a in attrs]), W=fit["W"], b=fit["b"],
                 alpha=fit["alpha"], n_train=fit["n_train"], trained_on=np.array(json.dumps(trained_on)))
    if not quiet:
        edge = [a for a, e in zip(attrs, fit["alpha_at_edge"]) if e]
        print(f"wrote {out}: {len(attrs)} attributes, {int(ok.sum())} words"
              + (f"; alpha at a grid edge for {edge}" if edge else ""))
    return out
