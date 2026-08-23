"""Contract B §4.1: both output tables must carry `stimulus_id`.

Regression cover for the guard bug where a single
``if "stimulus_id" not in chunks_df.columns`` wrapped the resolution for
*both* frames. When the input CSV already supplied ``stimulus_id``,
passthrough put it on chunks_df, the guard short-circuited, and every
``*_words.csv`` was written without the column -- silently, because
consumers fall back to ``chunk_label``.

Uses a fake model so no weights or downloads are needed.
"""

import sys

import pandas as pd
import pytest

from word2psy.models.base import BaseModel


class FakeWordModel(BaseModel):
    name = "fake_word"
    level = "word"

    def load(self):
        self.model = object()

    def predict(self, text):
        return {"fake_word_length": float(len(text))}


@pytest.fixture
def run_cli(tmp_path, monkeypatch):
    """Run the CLI end to end with the fake model registered."""

    def _run(csv_rows, extra_args):
        src = tmp_path / "in.csv"
        pd.DataFrame(csv_rows).to_csv(src, index=False)
        out = tmp_path / "scores.csv"

        import word2psy.cli as cli

        monkeypatch.setitem(
            cli.MODEL_REGISTRY, "fake_word",
            ("tests.test_stimulus_id", "FakeWordModel", "fake"),
        )
        monkeypatch.setattr(
            cli, "_load_model_class", lambda name: FakeWordModel
        )
        monkeypatch.setattr(
            sys, "argv",
            ["word2psy", "fake_word", str(src), "-o", str(out), "--quiet",
             *extra_args],
        )
        from word2psy.cli import main

        main()
        return (
            pd.read_csv(tmp_path / "scores_words.csv"),
            pd.read_csv(tmp_path / "scores_chunks.csv"),
        )

    return _run


ROWS = [
    {"stimulus_id": "shared0001", "caption": "a red bus"},
    {"stimulus_id": "shared0002", "caption": "two grey cats"},
]


def test_words_table_gets_stimulus_id_when_input_supplies_it(run_cli):
    """The regression: input CSV carries stimulus_id, so chunks got it via
    passthrough and words was left without it."""
    words, chunks = run_cli(
        ROWS, ["--text-column", "caption", "--id-column", "stimulus_id"]
    )
    assert "stimulus_id" in chunks.columns
    assert "stimulus_id" in words.columns
    assert not words["stimulus_id"].isna().any()


def test_words_stimulus_id_matches_its_chunk(run_cli):
    """Every word inherits the id of the chunk it belongs to."""
    words, chunks = run_cli(
        ROWS, ["--text-column", "caption", "--id-column", "stimulus_id"]
    )
    expected = words["chunk_idx"].map(
        dict(zip(chunks["chunk_idx"], chunks["stimulus_id"]))
    )
    assert (words["stimulus_id"] == expected).all()
    assert set(words["stimulus_id"]) == {"shared0001", "shared0002"}


def test_constant_stimulus_id_still_reaches_both_tables(run_cli):
    """--stimulus-id is the other path into the same block."""
    words, chunks = run_cli(
        [{"caption": "a red bus"}],
        ["--text-column", "caption", "--stimulus-id", "movie-42"],
    )
    assert set(chunks["stimulus_id"]) == {"movie-42"}
    assert set(words["stimulus_id"]) == {"movie-42"}


def test_id_column_labels_both_tables_without_passthrough(run_cli):
    """--id-column on a frame that did not already carry stimulus_id."""
    words, chunks = run_cli(
        [{"label": "w1", "caption": "a red bus"},
         {"label": "w2", "caption": "two grey cats"}],
        ["--text-column", "caption", "--id-column", "label"],
    )
    assert set(chunks["stimulus_id"]) == {"w1", "w2"}
    assert set(words["stimulus_id"]) == {"w1", "w2"}
