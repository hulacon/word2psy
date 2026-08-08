"""CLAP text encoder — 512-d L2-normalised embeddings per chunk.

Uses the same LAION-CLAP checkpoint as aud2psy's `clap` audio model
(`laion/larger_clap_music_and_speech`), so text and audio embeddings live
in one shared space and can be directly compared (cosine similarity
between a caption/transcript chunk and a stretch of soundtrack). Do not
change the checkpoint, the L2 normalisation, or the `clap_text_{i:03d}`
column naming without coordinating matching changes in aud2psy and
psyquilt's COMPATIBLE_SPACES — the audio-side twin of the clip_text ↔
viz2psy contract.
"""

import torch

from word2psy.models.base import BaseModel

CHECKPOINT = "laion/larger_clap_music_and_speech"
EMBED_DIM = 512


class CLAPTextModel(BaseModel):
    """Extract L2-normalised CLAP text embeddings (aud2psy-compatible)."""

    name = "clap_text"
    level = "chunk"

    def __init__(self, checkpoint: str = CHECKPOINT, device: str | None = None):
        super().__init__(device=device)
        self.checkpoint = checkpoint
        self._processor = None

    def load(self) -> None:
        from transformers import ClapModel, ClapProcessor

        self._processor = ClapProcessor.from_pretrained(self.checkpoint)
        model = ClapModel.from_pretrained(self.checkpoint).eval()
        try:
            self.model = model.to(self.device)
        except Exception:
            self.device = torch.device("cpu")
            self.model = model.to(self.device)

    def predict(self, text: str) -> dict[str, float]:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[dict[str, float]]:
        inputs = self._processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            out = self.model.get_text_features(**inputs)
        # transformers >= 5 returns a ModelOutput; the 512-d projection is
        # pooler_output (older versions returned the tensor directly)
        features = out.pooler_output if hasattr(out, "pooler_output") else out
        features = features / features.norm(dim=-1, keepdim=True)
        return [
            {f"clap_text_{i:03d}": v for i, v in enumerate(row)}
            for row in features.cpu().tolist()
        ]
