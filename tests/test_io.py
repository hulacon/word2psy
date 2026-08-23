"""Offline tests for pipeline I/O structure and CLI input parsing.

Uses lightweight fake models so no downloads or model weights are needed.
"""

import pandas as pd
import pytest

from word2psy.cli import read_inputs
from word2psy.exceptions import TextLoadError
from word2psy.models.base import BaseModel
from word2psy.pipeline import score_text


class FakeWordModel(BaseModel):
    name = "fake_word"
    level = "word"

    def load(self):
        self.model = object()

    def predict(self, text):
        return {"length": float(len(text)), "vowels": float(sum(c in "aeiou" for c in text))}


class FakeChunkModel(BaseModel):
    name = "fake_chunk"
    level = "chunk"

    def load(self):
        self.model = object()

    def predict(self, text):
        return {f"fake_chunk_{i:03d}": float(i) + len(text) for i in range(4)}


class TestTwoTableOutput:
    def test_word_features_in_words_df(self):
        words_df, chunks_df = score_text(
            "hello world", [FakeWordModel(device="cpu")], quiet=True
        )
        assert "length" in words_df.columns
        assert "vowels" in words_df.columns
        assert "length" not in chunks_df.columns

    def test_chunk_features_in_chunks_df(self):
        words_df, chunks_df = score_text(
            ["hello world", "more text here"],
            [FakeChunkModel(device="cpu")],
            quiet=True,
        )
        assert len(chunks_df) == 2
        emb_cols = [c for c in chunks_df.columns if c.startswith("fake_chunk_")]
        assert len(emb_cols) == 4
        assert emb_cols == sorted(emb_cols)  # insertion order preserved
        assert "fake_chunk_000" not in words_df.columns

    def test_n_words_per_chunk(self):
        _, chunks_df = score_text(
            ["one two three", "four five"], [FakeWordModel(device="cpu")], quiet=True
        )
        assert chunks_df["n_words"].tolist() == [3, 2]

    def test_chunk_labels(self):
        _, chunks_df = score_text(
            ["a b", "c d"],
            [FakeWordModel(device="cpu")],
            chunk_labels=["stim_A", "stim_B"],
            quiet=True,
        )
        assert chunks_df["chunk_label"].tolist() == ["stim_A", "stim_B"]

    def test_passthrough_columns(self):
        pt = pd.DataFrame({"condition": ["neg", "pos"], "block": [1, 2]})
        _, chunks_df = score_text(
            ["sad text", "happy text"],
            [FakeChunkModel(device="cpu")],
            passthrough=pt,
            quiet=True,
        )
        assert chunks_df["condition"].tolist() == ["neg", "pos"]
        assert chunks_df["block"].tolist() == [1, 2]
        # Passthrough comes before feature columns
        cols = list(chunks_df.columns)
        assert cols.index("condition") < cols.index("fake_chunk_000")

    def test_passthrough_length_mismatch_raises(self):
        pt = pd.DataFrame({"condition": ["neg"]})
        with pytest.raises(ValueError):
            score_text(
                ["a", "b"], [FakeChunkModel(device="cpu")], passthrough=pt, quiet=True
            )

    def test_feature_names_recorded(self):
        wm, cm = FakeWordModel(device="cpu"), FakeChunkModel(device="cpu")
        score_text("hello there", [wm, cm], quiet=True)
        assert wm.feature_names_ == ["length", "vowels"]
        assert cm.feature_names_ == [f"fake_chunk_{i:03d}" for i in range(4)]


class TestReadInputs:
    def test_plain_text_single_file(self, tmp_path):
        p = tmp_path / "input.txt"
        p.write_text("some text")
        text, labels, passthrough = read_inputs([p])
        assert text == "some text"
        assert labels is None
        assert passthrough is None

    def test_plain_text_multiple_files(self, tmp_path):
        p1 = tmp_path / "a.txt"
        p2 = tmp_path / "b.txt"
        p1.write_text("first")
        p2.write_text("second")
        text, _, _ = read_inputs([p1, p2])
        assert text == ["first", "second"]

    def test_csv_input(self, tmp_path):
        p = tmp_path / "stimuli.csv"
        p.write_text("stim_id,word,condition\ns1,dog,animal\ns2,justice,abstract\n")
        text, labels, passthrough = read_inputs(
            [p], text_column="word", id_column="stim_id"
        )
        assert text == ["dog", "justice"]
        assert labels == ["s1", "s2"]
        # `condition` is an experiment variable, not a feature: since 0.6.0 it
        # is not carried unless asked for. `stim_id` is not stimulus_id, so it
        # is not carried either -- the id reaches the output as chunk labels.
        assert passthrough is None

    def test_csv_input_passthrough_on_request(self, tmp_path):
        p = tmp_path / "stimuli.csv"
        p.write_text("stim_id,word,condition\ns1,dog,animal\ns2,justice,abstract\n")
        _, _, passthrough = read_inputs(
            [p], text_column="word", id_column="stim_id",
            passthrough_columns=["stim_id", "condition"],
        )
        assert list(passthrough.columns) == ["stim_id", "condition"]
        assert passthrough["condition"].tolist() == ["animal", "abstract"]

    def test_tsv_input(self, tmp_path):
        p = tmp_path / "stimuli.tsv"
        p.write_text("word\trating\ncat\t5\n")
        text, labels, passthrough = read_inputs([p], text_column="word")
        assert text == ["cat"]
        assert labels is None
        assert passthrough is None

    def test_tsv_input_passthrough_all(self, tmp_path):
        p = tmp_path / "stimuli.tsv"
        p.write_text("word\trating\ncat\t5\n")
        _, _, passthrough = read_inputs(
            [p], text_column="word", passthrough_columns=["all"]
        )
        assert passthrough["rating"].tolist() == [5]

    def test_csv_requires_text_column(self, tmp_path):
        p = tmp_path / "stimuli.csv"
        p.write_text("word\ndog\n")
        with pytest.raises(TextLoadError):
            read_inputs([p])

    def test_csv_bad_column_raises(self, tmp_path):
        p = tmp_path / "stimuli.csv"
        p.write_text("word\ndog\n")
        with pytest.raises(TextLoadError):
            read_inputs([p], text_column="nonexistent")

    def test_csv_mixed_with_txt_raises(self, tmp_path):
        c = tmp_path / "stimuli.csv"
        t = tmp_path / "input.txt"
        c.write_text("word\ndog\n")
        t.write_text("text")
        with pytest.raises(TextLoadError):
            read_inputs([c, t], text_column="word")


class TestPassthroughIsOptIn:
    """word2psy 0.6.0: input columns no longer leak into feature output.

    Copying every non-text column made experiment variables (annotator,
    seg_number, trial index) indistinguishable from model output, and made
    each model re-emit them, so they collided on any (stimulus, feature) key.
    """

    def _csv(self, tmp_path):
        import pandas as pd
        src = tmp_path / "in.csv"
        pd.DataFrame([
            {"stimulus_id": "m1", "seg_number": 1, "annotator": "AB",
             "description": "a dog runs"},
            {"stimulus_id": "m2", "seg_number": 2, "annotator": "CD",
             "description": "a cat sits"},
        ]).to_csv(src, index=False)
        return src

    def test_default_carries_only_stimulus_id(self, tmp_path):
        _, _, pt = read_inputs([self._csv(tmp_path)], text_column="description")
        assert list(pt.columns) == ["stimulus_id"]

    def test_named_columns_are_carried(self, tmp_path):
        _, _, pt = read_inputs([self._csv(tmp_path)], text_column="description",
                               passthrough_columns=["seg_number"])
        assert set(pt.columns) == {"stimulus_id", "seg_number"}
        assert "annotator" not in pt.columns

    def test_all_restores_the_old_behaviour(self, tmp_path):
        _, _, pt = read_inputs([self._csv(tmp_path)], text_column="description",
                               passthrough_columns=["all"])
        assert set(pt.columns) == {"stimulus_id", "seg_number", "annotator"}

    def test_unknown_column_names_the_available_ones(self, tmp_path):
        with pytest.raises(TextLoadError, match="not found"):
            read_inputs([self._csv(tmp_path)], text_column="description",
                        passthrough_columns=["nope"])

    def test_input_without_stimulus_id_passes_nothing(self, tmp_path):
        import pandas as pd
        src = tmp_path / "plain.csv"
        pd.DataFrame([{"label": "a", "text": "a dog runs"}]).to_csv(src, index=False)
        _, _, pt = read_inputs([src], text_column="text")
        assert pt is None

    def test_intrinsic_coordinates_are_always_carried(self, tmp_path):
        """onset/offset are the stimulus's own coordinates, not experiment
        variables: an ASR segment's bounds must reach the output even though
        word2psy cannot derive them itself."""
        import pandas as pd
        src = tmp_path / "seg.csv"
        pd.DataFrame([
            {"stimulus_id": "m1", "onset": 0.0, "offset": 6.0,
             "asr_confidence": 0.9, "text": "a dog runs"},
        ]).to_csv(src, index=False)
        _, _, pt = read_inputs([src], text_column="text")
        assert set(pt.columns) == {"stimulus_id", "onset", "offset"}
        assert "asr_confidence" not in pt.columns
