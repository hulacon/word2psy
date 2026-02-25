"""Train Ridge regressors from fastText embeddings to predict lexical norms."""

import os
from pathlib import Path

import joblib
import numpy as np

from word2psy.exceptions import NormDataError

CACHE_DIR = Path(os.environ.get("WORD2PSY_CACHE", Path.home() / ".cache" / "word2psy"))
MODELS_DIR = CACHE_DIR / "models"
FASTTEXT_DIR = CACHE_DIR / "fasttext"

FASTTEXT_MODEL_NAME = "crawl-300d-2M-subword.bin"
FASTTEXT_URL = (
    "https://dl.fbaipublicfiles.com/fasttext/vectors-english/"
    "crawl-300d-2M-subword.zip"
)

# All norm dimensions to train regressors for.
# Maps standardised feature name -> (norm_database_name, column_in_parquet).
NORM_DIMENSIONS = {
    "concreteness": ("brysbaert_concreteness", "concreteness"),
    "valence": ("nrc_vad", "valence"),
    "arousal": ("nrc_vad", "arousal"),
    "dominance": ("nrc_vad", "dominance"),
    "age_of_acquisition": ("kuperman_aoa", "age_of_acquisition"),
    "imageability": ("glasgow", "imageability"),
    "sensorimotor_touch": ("lancaster_sensorimotor", "sensorimotor_touch"),
    "sensorimotor_hearing": ("lancaster_sensorimotor", "sensorimotor_hearing"),
    "sensorimotor_smell": ("lancaster_sensorimotor", "sensorimotor_smell"),
    "sensorimotor_taste": ("lancaster_sensorimotor", "sensorimotor_taste"),
    "sensorimotor_vision": ("lancaster_sensorimotor", "sensorimotor_vision"),
    "sensorimotor_interoception": (
        "lancaster_sensorimotor",
        "sensorimotor_interoception",
    ),
    "sensorimotor_mouth": ("lancaster_sensorimotor", "sensorimotor_mouth"),
    "sensorimotor_hand": ("lancaster_sensorimotor", "sensorimotor_hand"),
    "sensorimotor_foot": ("lancaster_sensorimotor", "sensorimotor_foot"),
    "sensorimotor_head": ("lancaster_sensorimotor", "sensorimotor_head"),
    "sensorimotor_torso": ("lancaster_sensorimotor", "sensorimotor_torso"),
}


def _download_fasttext_model() -> Path:
    """Download the fastText crawl-300d-2M-subword model if not cached."""
    FASTTEXT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = FASTTEXT_DIR / FASTTEXT_MODEL_NAME

    if model_path.exists():
        return model_path

    import io
    import urllib.request
    import zipfile

    zip_path = FASTTEXT_DIR / "crawl-300d-2M-subword.zip"

    if not zip_path.exists():
        print("Downloading fastText model (~2GB compressed, ~7GB uncompressed)...")
        print("  This is a one-time download.")
        req = urllib.request.Request(
            FASTTEXT_URL, headers={"User-Agent": "word2psy/0.1"}
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            with open(zip_path, "wb") as f:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 1024 * 1024  # 1MB
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded / total * 100
                        print(
                            f"\r  {downloaded / 1e9:.1f} / {total / 1e9:.1f} GB "
                            f"({pct:.0f}%)",
                            end="",
                            flush=True,
                        )
        print()

    print("Extracting fastText model...")
    import zipfile

    with zipfile.ZipFile(zip_path) as zf:
        # Find the .bin file in the archive
        bin_files = [n for n in zf.namelist() if n.endswith(".bin")]
        if not bin_files:
            raise NormDataError(
                "fasttext", "No .bin file found in fastText archive"
            )
        with zf.open(bin_files[0]) as src, open(model_path, "wb") as dst:
            import shutil

            shutil.copyfileobj(src, dst)

    # Remove the zip to save disk space
    zip_path.unlink()
    print(f"  fastText model ready at {model_path}")

    return model_path


def load_fasttext():
    """Load the fastText model, downloading if necessary."""
    import fasttext

    model_path = _download_fasttext_model()
    # Suppress fastText warnings about deprecated model format
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fasttext.load_model(str(model_path))


def _build_training_data(
    norm_df, ft_model
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build X (embeddings) and y (scores) arrays from a norm DataFrame.

    Returns (X, y, words) where rows with zero vectors are excluded.
    """
    words = norm_df["word"].tolist()
    scores = norm_df.iloc[:, 1].values.astype(np.float64)

    X = np.array([ft_model.get_word_vector(w) for w in words], dtype=np.float32)

    # Exclude rows where fastText returned a zero vector (very rare)
    norms = np.linalg.norm(X, axis=1)
    valid = norms > 0
    return X[valid], scores[valid], [w for w, v in zip(words, valid) if v]


def train_single_norm(
    feature_name: str,
    *,
    ft_model=None,
    force: bool = False,
    quiet: bool = False,
) -> Path:
    """Train a Ridge regressor for a single norm dimension.

    Returns the path to the saved joblib file.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_score

    from word2psy.norms.download import load_norm

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / f"{feature_name}.joblib"

    if out_path.exists() and not force:
        return out_path

    if feature_name not in NORM_DIMENSIONS:
        raise NormDataError(
            feature_name,
            f"Unknown dimension. Available: {list(NORM_DIMENSIONS)}",
        )

    norm_db_name, col_name = NORM_DIMENSIONS[feature_name]

    # Load norm data
    norm_df = load_norm(norm_db_name)[["word", col_name]].dropna()

    # Load fastText if not provided
    if ft_model is None:
        ft_model = load_fasttext()

    # Build training data
    X, y, words = _build_training_data(norm_df, ft_model)

    if not quiet:
        print(f"Training {feature_name}: {len(X)} words...", end=" ")

    # Train with built-in cross-validation for alpha selection
    model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
    model.fit(X, y)

    # Report cross-validated performance
    if not quiet:
        scores = cross_val_score(model, X, y, cv=5, scoring="r2")
        print(f"CV r2 = {scores.mean():.3f} (+/- {scores.std():.3f})")

    joblib.dump(model, out_path)
    return out_path


def train_all_norms(*, force: bool = False, quiet: bool = False) -> dict[str, Path]:
    """Train Ridge regressors for all norm dimensions.

    Loads fastText once and reuses for all dimensions.
    """
    if not quiet:
        print("Loading fastText model...")
    ft_model = load_fasttext()

    paths = {}
    for name in NORM_DIMENSIONS:
        paths[name] = train_single_norm(
            name, ft_model=ft_model, force=force, quiet=quiet
        )

    return paths


def load_regressors(*, quiet: bool = False) -> dict[str, object]:
    """Load all trained Ridge regressors, training if necessary.

    Returns a dict mapping feature name -> fitted Ridge model.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Check if all models exist
    all_exist = all(
        (MODELS_DIR / f"{name}.joblib").exists() for name in NORM_DIMENSIONS
    )

    if not all_exist:
        if not quiet:
            print("Norm regressors not found. Training (one-time setup)...")
        train_all_norms(quiet=quiet)

    regressors = {}
    for name in NORM_DIMENSIONS:
        path = MODELS_DIR / f"{name}.joblib"
        regressors[name] = joblib.load(path)

    return regressors
