"""Tests for the user-supplied Binder et al. (2016) ratings loader.

Offline: every test writes a small synthetic spreadsheet with placeholder
words; the real ratings file is never needed.
"""

import numpy as np
import pandas as pd
import pytest

from word2psy.exceptions import NormDataError
from word2psy.norms import binder


def _sheet(rows):
    """A spreadsheet laid out like WordSet1_Ratings.xlsx: No, Word, WC, N, Mean R, 65 attributes, extras."""
    recs = []
    for i, (word, wc, missing) in enumerate(rows, start=1):
        rec = {"No": i, "Word": word, "WC": wc, "N": 30, "Mean R": 0.8}
        for j, a in enumerate(binder.ATTRIBUTES):
            rec[a] = "na" if a in missing else round((i + j) % 7 * 0.9, 3)
        rec["LEN"] = len(word)
        rec["Super Category"] = "Category A" if wc == 1 else "Category B"
        recs.append(rec)
    return pd.DataFrame(recs)


@pytest.fixture
def ratings_file(tmp_path):
    rows = [("wordone", 1, ()),
            ("wordtwo", 2, ("Complexity",)),
            ("wordthree", 3, ("Complexity", "Practice", "Caused")),
            # one word rated twice, as two word classes: the row with more ratings is kept
            ("wordfour", 2, ("Complexity",)),
            ("wordfour", 3, ("Complexity", "Practice", "Caused"))]
    p = tmp_path / "WordSet1_Ratings.xlsx"
    _sheet(rows).to_excel(p, index=False)
    return p


def test_attribute_map_shape():
    assert len(binder.ATTRIBUTES) == 65
    assert len(binder.DOMAINS) == 14
    assert binder.DOMAINS[0] == "Vision" and binder.DOMAINS[-1] == "Attention"
    assert binder.domain_of("upperlimb") == "Motor"


def test_load(ratings_file):
    df = binder.load_binder(ratings_file)
    assert list(df["word"]) == ["wordone", "wordtwo", "wordthree", "wordfour"]
    assert list(df.columns[:5]) == ["word", "word_class", "n_raters", "mean_r", "super_category"]
    assert len(df.columns) == 5 + 65
    assert df.loc[df.word == "wordfour", "word_class"].item() == "verb"
    assert np.isnan(df.loc[df.word == "wordthree", "practice"].item())
    assert not np.isnan(df.loc[df.word == "wordone", "complexity"].item())
    assert len(df.attrs["sha256"]) == 64
    assert "Cognitive Neuropsychology" in df.attrs["citation"]


def test_env_var(ratings_file, monkeypatch):
    monkeypatch.setenv(binder.ENV_VAR, str(ratings_file))
    assert len(binder.load_binder()) == 4


def test_missing_file_names_sources(tmp_path, monkeypatch):
    monkeypatch.delenv(binder.ENV_VAR, raising=False)
    with pytest.raises(NormDataError) as e:
        binder.load_binder()
    msg = str(e.value)
    assert "neuro.mcw.edu" in msg and "danoneata/seeing-what-tastes-good" in msg
    assert binder.ENV_VAR in msg and "doi:10.1080/02643294.2016.1147426" in msg
    with pytest.raises(NormDataError, match="is not a file"):
        binder.load_binder(tmp_path / "absent.xlsx")


def test_wrong_layout(tmp_path):
    p = tmp_path / "other.xlsx"
    pd.DataFrame({"Word": ["wordone"], "Vision": [1.0]}).to_excel(p, index=False)
    with pytest.raises(NormDataError, match="is it WordSet1_Ratings.xlsx"):
        binder.load_binder(p)
