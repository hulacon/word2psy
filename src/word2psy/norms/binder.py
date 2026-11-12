"""Binder et al. (2016) experiential attribute ratings, from a file the user supplies.

Binder, Conant, Humphries, Fernandino, Simons, Aguilar & Desai (2016) collected
ratings of 535 English words on 65 experiential attributes (0-6: how strongly
the word's referent is associated with, e.g., vision, sound, upper-limb action
or fear), grouped into 14 domains. The ratings are distributed as a
spreadsheet, ``WordSet1_Ratings.xlsx``, with a request to cite the article and
no licence. word2psy therefore never downloads them: pass the file's path (or
set ``WORD2PSY_BINDER_RATINGS``). :func:`load_binder` checks the layout the
authors' readme describes and returns one row per word.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd

from word2psy.exceptions import NormDataError

ENV_VAR = "WORD2PSY_BINDER_RATINGS"
NORM_NAME = "binder_experiential"

CITATION = (
    "Binder, J. R., Conant, L. L., Humphries, C. J., Fernandino, L., Simons, S. B., "
    "Aguilar, M., & Desai, R. H. (2016). Toward a brain-based componential semantic "
    "representation. Cognitive Neuropsychology, 33(3-4), 130-174. "
    "doi:10.1080/02643294.2016.1147426"
)

# Where the spreadsheet can be obtained. Neither is fetched by this package.
SOURCES = (
    "the Binder lab's page for the article (https://www.neuro.mcw.edu/?p=609): "
    "representations/word_ratings.zip, which contains WordSet1_Ratings.xlsx",
    "Oneata, Elliott & Frank (2025) redistribute the same spreadsheet as "
    "data/binder-norms.xlsx in https://github.com/danoneata/seeing-what-tastes-good",
)

# Attribute -> (domain, the definition raters were given), in the spreadsheet's
# column order (Binder et al., 2016, Table 1).
ATTRIBUTES: dict[str, tuple[str, str]] = {
    "Vision": ("Vision", "something that you can easily see"),
    "Bright": ("Vision", "visually light or bright"),
    "Dark": ("Vision", "visually dark"),
    "Color": ("Vision", "having a characteristic or defining colour"),
    "Pattern": ("Vision", "having a characteristic or defining visual texture or surface pattern"),
    "Large": ("Vision", "large in size"),
    "Small": ("Vision", "small in size"),
    "Motion": ("Vision", "showing a lot of visually observable movement"),
    "Biomotion": ("Vision", "showing movement like that of a living thing"),
    "Fast": ("Vision", "showing visible movement that is fast"),
    "Slow": ("Vision", "showing visible movement that is slow"),
    "Shape": ("Vision", "having a characteristic or defining visual shape or form"),
    "Complexity": ("Vision", "visually complex"),
    "Face": ("Vision", "having a human or human-like face"),
    "Body": ("Vision", "having human or human-like body parts"),
    "Touch": ("Somatic", "something that you could easily recognize by touch"),
    "Temperature": ("Somatic", "hot or cold to the touch"),
    "Texture": ("Somatic", "having a smooth or rough texture to the touch"),
    "Weight": ("Somatic", "light or heavy in weight"),
    "Pain": ("Somatic", "associated with pain or physical discomfort"),
    "Audition": ("Audition", "something that you can easily hear"),
    "Loud": ("Audition", "making a loud sound"),
    "Low": ("Audition", "having a low-pitched sound"),
    "High": ("Audition", "having a high-pitched sound"),
    "Sound": ("Audition", "having a characteristic or recognizable sound or sounds"),
    "Music": ("Audition", "making a musical sound"),
    "Speech": ("Audition", "someone or something that talks"),
    "Taste": ("Gustation", "having a characteristic or defining taste"),
    "Smell": ("Olfaction", "having a characteristic or defining smell or smells"),
    "Head": ("Motor", "associated with actions using the face, mouth, or tongue"),
    "UpperLimb": ("Motor", "associated with actions using the arm, hand, or fingers"),
    "LowerLimb": ("Motor", "associated with actions using the leg or foot"),
    "Practice": ("Motor", "a physical object you have personal experience using"),
    "Landmark": ("Spatial", "having a fixed location, as on a map"),
    "Path": ("Spatial", "showing changes in location along a particular direction or path"),
    "Scene": ("Spatial", "bringing to mind a particular setting or physical location"),
    "Near": ("Spatial", "often physically near to you (within easy reach) in everyday life"),
    "Toward": ("Spatial", "associated with movement toward or into you"),
    "Away": ("Spatial", "associated with movement away from or out of you"),
    "Number": ("Spatial", "associated with a specific number or amount"),
    "Time": ("Temporal", "an event or occurrence that occurs at a typical or predictable time"),
    "Duration": ("Temporal", "an event that has a predictable duration, whether short or long"),
    "Long": ("Temporal", "an event that lasts for a long period of time"),
    "Short": ("Temporal", "an event that lasts for a short period of time"),
    "Caused": ("Causal", "caused by some clear preceding event, action, or situation"),
    "Consequential": ("Causal", "likely to have consequences (cause other things to happen)"),
    "Social": ("Social", "an activity or event that involves an interaction between people"),
    "Human": ("Social", "having human or human-like intentions, plans, or goals"),
    "Communication": ("Social", "a thing or action that people use to communicate"),
    "Self": ("Social", "related to your own view of yourself, a part of your self-image"),
    "Cognition": ("Cognition", "a form of mental activity or a function of the mind"),
    "Benefit": ("Emotion", "someone or something that could help or benefit you or others"),
    "Harm": ("Emotion", "someone or something that could cause harm to you or others"),
    "Pleasant": ("Emotion", "someone or something that you find pleasant"),
    "Unpleasant": ("Emotion", "someone or something that you find unpleasant"),
    "Happy": ("Emotion", "someone or something that makes you feel happy"),
    "Sad": ("Emotion", "someone or something that makes you feel sad"),
    "Angry": ("Emotion", "someone or something that makes you feel angry"),
    "Disgusted": ("Emotion", "someone or something that makes you feel disgusted"),
    "Fearful": ("Emotion", "someone or something that makes you feel afraid"),
    "Surprised": ("Emotion", "someone or something that makes you feel surprised"),
    "Drive": ("Drive", "someone or something that motivates you to do something"),
    "Needs": ("Drive", "someone or something that would be hard for you to live without"),
    "Attention": ("Attention", "someone or something that grabs your attention"),
    "Arousal": ("Attention", "someone or something that makes you feel alert, activated, "
                             "excited, or keyed up in either a positive or negative way"),
}

DOMAINS: tuple[str, ...] = tuple(dict.fromkeys(d for d, _ in ATTRIBUTES.values()))

# Spreadsheet word-class codes (readme: 1 = noun, 2 = verb, 3 = adjective).
WORD_CLASSES = {1: "noun", 2: "verb", 3: "adjective"}

# Spreadsheet columns read besides the 65 attributes.
_META = {"Word": "word", "WC": "word_class", "N": "n_raters", "Mean R": "mean_r",
         "Super Category": "super_category"}


def column_name(attribute: str) -> str:
    """The lower-case name used for an attribute in returned tables (``UpperLimb`` -> ``upperlimb``)."""
    return attribute.lower()


def _where_to_get_it() -> str:
    lines = "\n".join(f"  - {s}" for s in SOURCES)
    return (f"The Binder et al. (2016) ratings are supplied by the user, never downloaded. "
            f"Obtain WordSet1_Ratings.xlsx from either of:\n{lines}\n"
            f"then pass its path or set {ENV_VAR}. Please cite: {CITATION}")


def resolve_path(path: str | os.PathLike | None = None) -> Path:
    """The ratings file from ``path`` or ``$WORD2PSY_BINDER_RATINGS``; raises if neither names a file."""
    raw = path if path is not None else os.environ.get(ENV_VAR)
    if not raw:
        raise NormDataError(NORM_NAME, f"no ratings file given.\n{_where_to_get_it()}")
    p = Path(raw).expanduser()
    if not p.is_file():
        raise NormDataError(NORM_NAME, f"{p} is not a file.\n{_where_to_get_it()}")
    return p


def file_sha256(path: str | os.PathLike) -> str:
    """sha256 of the ratings file, so outputs can say exactly which copy they were built from."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_binder(path: str | os.PathLike | None = None) -> pd.DataFrame:
    """Read the ratings spreadsheet into one row per word.

    Columns: ``word`` (lower case), ``word_class`` (noun / verb / adjective),
    ``n_raters``, ``mean_r`` (rater-to-group agreement for that word),
    ``super_category``, then the 65 attributes under :func:`column_name`.
    Attributes the authors did not rate for a word class (the readme: verbs and
    adjectives on Complexity, adjectives on Practice and Caused) are NaN.

    A word listed more than once (the spreadsheet rates "used" as a verb and as
    an adjective) keeps the row with the most rated attributes, the first such
    row on a tie: a word-level model cannot tell the senses apart.

    The file's sha256 and path are in ``df.attrs``.
    """
    p = resolve_path(path)
    try:
        raw = pd.read_excel(p, sheet_name=0)
    except Exception as e:  # noqa: BLE001 - any read failure means "not the spreadsheet"
        raise NormDataError(NORM_NAME, f"could not read {p} as a spreadsheet ({e}).\n"
                                       f"{_where_to_get_it()}") from e
    missing = [c for c in [*_META, *ATTRIBUTES] if c not in raw.columns]
    if missing:
        raise NormDataError(
            NORM_NAME,
            f"{p} lacks expected columns {missing[:8]}{' …' if len(missing) > 8 else ''} "
            f"({len(missing)} missing); is it WordSet1_Ratings.xlsx?\n{_where_to_get_it()}")

    df = raw[list(_META)].rename(columns=_META)
    df["word"] = df["word"].astype(str).str.strip().str.lower()
    codes = pd.to_numeric(df["word_class"], errors="coerce")
    bad = sorted(set(codes.dropna().astype(int)) - set(WORD_CLASSES)) + (["blank"] if codes.isna().any() else [])
    if bad:
        raise NormDataError(NORM_NAME, f"{p}: unknown word-class codes {bad} (expected 1, 2, 3)")
    df["word_class"] = codes.astype(int).map(WORD_CLASSES)
    df["n_raters"] = pd.to_numeric(df["n_raters"], errors="coerce")
    df["mean_r"] = pd.to_numeric(df["mean_r"], errors="coerce")
    attrs = pd.DataFrame({column_name(a): pd.to_numeric(raw[a], errors="coerce") for a in ATTRIBUTES})
    out = pd.concat([df, attrs], axis=1)

    rated = attrs.notna().sum(axis=1)
    order = (pd.DataFrame({"word": out["word"], "rated": rated, "row": range(len(out))})
             .sort_values(["word", "rated", "row"], ascending=[True, False, True]))
    keep = order.drop_duplicates("word", keep="first")["row"].sort_values().to_numpy()
    out = out.iloc[keep].reset_index(drop=True)

    out.attrs["sha256"] = file_sha256(p)
    out.attrs["path"] = str(p)
    out.attrs["citation"] = CITATION
    return out


def domain_of(column: str) -> str:
    """Binder domain of a returned attribute column (``upperlimb`` -> ``Motor``)."""
    for a, (d, _) in ATTRIBUTES.items():
        if column_name(a) == column:
            return d
    raise KeyError(column)
