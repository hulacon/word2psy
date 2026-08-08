"""Download psycholinguistic norm databases to local cache."""

import io
import os
import zipfile
from pathlib import Path

import pandas as pd

from word2psy.exceptions import NormDataError

CACHE_DIR = Path(os.environ.get("WORD2PSY_CACHE", Path.home() / ".cache" / "word2psy"))
NORMS_DIR = CACHE_DIR / "norms"

# Source URLs and parsing specs for each norm database.
NORM_SOURCES = {
    "brysbaert_concreteness": {
        "url": (
            "https://raw.githubusercontent.com/ArtsEngine/concreteness/"
            "master/Concreteness_ratings_Brysbaert_et_al_BRM.txt"
        ),
        "format": "tsv",
        "word_col": "Word",
        "score_cols": {"concreteness": "Conc.M"},
    },
    "nrc_vad": {
        "url": "http://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon.zip",
        "format": "zip_tsv",
        "zip_glob": "*/NRC-VAD-Lexicon.txt",
        "word_col": 0,  # no header
        "score_cols": {"valence": 1, "arousal": 2, "dominance": 3},
        "header": None,
    },
    "kuperman_aoa": {
        "url": "https://osf.io/download/vb9je/",
        "format": "xlsx",
        "word_col": "Word",
        "score_cols": {"age_of_acquisition": "Rating.Mean"},
    },
    "glasgow": {
        "url": (
            "https://static-content.springer.com/esm/"
            "art%3A10.3758%2Fs13428-018-1099-3/MediaObjects/"
            "13428_2018_1099_MOESM2_ESM.csv"
        ),
        "format": "glasgow_csv",  # special handling for two-row header
        "word_col": 0,
        # Mean ("M") column indices per dimension group
        "score_cols": {
            "imageability": 14,
            "familiarity": 17,
            "semantic_size": 23,
            "gender_association": 26,
        },
    },
    "socialness": {
        "url": "https://osf.io/download/29eyh/",
        "format": "csv",
        "word_col": "Word",
        "score_cols": {"socialness": "Mean"},
    },
    "boi": {
        "url": "https://osf.io/download/r84qn/",
        "format": "xlsx",
        "word_col": "Word",
        "score_cols": {"body_object_interaction": "Mean"},
    },
    "lancaster_sensorimotor": {
        "url": "https://osf.io/48wsc/download",
        "format": "csv",
        "word_col": "Word",
        "score_cols": {
            "sensorimotor_touch": "Haptic.mean",
            "sensorimotor_hearing": "Auditory.mean",
            "sensorimotor_smell": "Olfactory.mean",
            "sensorimotor_taste": "Gustatory.mean",
            "sensorimotor_vision": "Visual.mean",
            "sensorimotor_interoception": "Interoceptive.mean",
            "sensorimotor_mouth": "Mouth.mean",
            "sensorimotor_hand": "Hand_arm.mean",
            "sensorimotor_foot": "Foot_leg.mean",
            "sensorimotor_head": "Head.mean",
            "sensorimotor_torso": "Torso.mean",
        },
    },
}


def _download_bytes(url: str) -> bytes:
    """Download a URL and return raw bytes."""
    import urllib.request

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "word2psy/0.1"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.read()
    except Exception as e:
        raise NormDataError(url, str(e)) from e


def _load_raw(name: str) -> pd.DataFrame:
    """Download and parse a norm database into a standardised DataFrame.

    Returns a DataFrame with columns: ``word`` plus one column per score
    (using the standardised names from ``NORM_SOURCES``).
    """
    spec = NORM_SOURCES[name]
    raw = _download_bytes(spec["url"])

    fmt = spec["format"]
    if fmt == "tsv":
        df = pd.read_csv(io.BytesIO(raw), sep="\t")
    elif fmt == "csv":
        df = pd.read_csv(io.BytesIO(raw))
    elif fmt == "glasgow_csv":
        # Glasgow norms have a two-row header: row 0 = dimension names,
        # row 1 = M/SD/N sub-headers.  Skip both and use column indices.
        df = pd.read_csv(io.BytesIO(raw), header=None, skiprows=2)
    elif fmt == "xlsx":
        df = pd.read_excel(io.BytesIO(raw))
    elif fmt == "zip_tsv":
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            import fnmatch

            pattern = spec["zip_glob"]
            matching = [n for n in zf.namelist() if fnmatch.fnmatch(n, pattern)]
            if not matching:
                # Fall back to first .txt file
                matching = [n for n in zf.namelist() if n.endswith(".txt")]
            if not matching:
                raise NormDataError(name, f"No matching file in ZIP ({pattern})")
            with zf.open(matching[0]) as f:
                df = pd.read_csv(
                    f,
                    sep="\t",
                    header=spec.get("header", "infer"),
                )
    else:
        raise NormDataError(name, f"Unknown format: {fmt}")

    # Standardise column names
    word_col = spec["word_col"]
    score_map = spec["score_cols"]

    result = pd.DataFrame()
    result["word"] = df.iloc[:, word_col] if isinstance(word_col, int) else df[word_col]
    for std_name, src_col in score_map.items():
        col_data = df.iloc[:, src_col] if isinstance(src_col, int) else df[src_col]
        result[std_name] = pd.to_numeric(col_data, errors="coerce")

    result = result.dropna(subset=["word"]).copy()
    result["word"] = result["word"].astype(str).str.strip().str.lower()
    result = result.drop_duplicates(subset=["word"], keep="first")

    return result


def download_norm(name: str, *, force: bool = False) -> Path:
    """Download a single norm database and save as parquet.

    Returns the path to the cached parquet file.
    """
    if name not in NORM_SOURCES:
        raise NormDataError(name, f"Unknown norm. Available: {list(NORM_SOURCES)}")

    NORMS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = NORMS_DIR / f"{name}.parquet"

    if out_path.exists() and not force:
        return out_path

    print(f"Downloading {name} norms...")
    df = _load_raw(name)
    df.to_parquet(out_path, index=False)
    print(f"  Saved {len(df)} words to {out_path}")

    return out_path


def download_all(*, force: bool = False) -> dict[str, Path]:
    """Download all norm databases. Returns dict of name -> path."""
    paths = {}
    for name in NORM_SOURCES:
        paths[name] = download_norm(name, force=force)
    return paths


def load_norm(name: str) -> pd.DataFrame:
    """Load a norm database, downloading if necessary.

    If the cached parquet predates a change to the source spec (missing
    newly added score columns), it is re-downloaded.
    """
    path = download_norm(name)
    df = pd.read_parquet(path)
    expected = set(NORM_SOURCES[name]["score_cols"])
    if not expected.issubset(df.columns):
        path = download_norm(name, force=True)
        df = pd.read_parquet(path)
    return df
