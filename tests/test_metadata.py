"""Offline tests for the output metadata sidecar (Contract B §4.1).

Checkpoint resolution reads class attributes via the lazy model registry,
so no weights are downloaded.
"""

import json

from word2psy import __version__
from word2psy.metadata import MetadataBuilder, get_model_checkpoint


class TestSidecarSchema:
    def test_common_keys(self):
        builder = MetadataBuilder()
        meta = builder.build()

        assert meta["schema_version"] == "1.0"
        assert meta["extractor"] == "word2psy"
        assert meta["extractor_version"] == __version__
        # Legacy key kept one deprecation cycle
        assert meta["word2psy_version"] == meta["extractor_version"]
        assert "created_at" in meta
        assert "models" in meta

    def test_add_model_entry(self):
        builder = MetadataBuilder()
        builder.add_model(
            "readability",
            [f"readability_{n}" for n in ("flesch_ease", "smog")],
            1.5,
            level="chunk",
        )
        entry = builder.build()["models"]["readability"]

        assert entry["package_version"] == entry["version"]  # legacy twin
        assert entry["checkpoint"] is None  # analytic model
        assert entry["features"]["count"] == 2

    def test_sidecar_roundtrip(self, tmp_path):
        builder = MetadataBuilder()
        meta_path = builder.save(tmp_path / "scores.csv")

        assert meta_path.name == "scores.meta.json"
        meta = json.loads(meta_path.read_text())
        assert meta["schema_version"] == "1.0"


class TestModelCheckpoints:
    def test_clip_text_matches_viz2psy_format(self):
        # psytwill asserts equality with viz2psy's clip checkpoint string
        assert get_model_checkpoint("clip_text") == "ViT-B-32/laion2b_s34b_b79k"

    def test_clap_text_matches_aud2psy(self):
        assert (
            get_model_checkpoint("clap_text") == "laion/larger_clap_music_and_speech"
        )

    def test_learned_models_declare_checkpoints(self):
        expected = {
            "gpt2_surprisal": "gpt2",
            "emotion": "SamLowe/roberta-base-go_emotions",
            "sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "minilm": "sentence-transformers/all-MiniLM-L6-v2",
            "word2vec": "word2vec-google-news-300",
            "fasttext": "crawl-300d-2M-subword",
            "lexical_norms": "crawl-300d-2M-subword+ridge",
        }
        for name, checkpoint in expected.items():
            assert get_model_checkpoint(name) == checkpoint, name

    def test_analytic_models_have_none(self):
        assert get_model_checkpoint("readability") is None
        assert get_model_checkpoint("wordform") is None
