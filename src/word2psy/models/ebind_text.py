"""EBind text arm — 1024-d embeddings in the cross-modal shared space.

Uses the same revision-pinned checkpoint as viz2psy's ``ebind`` (image arm)
and aud2psy's ``ebind_audio`` (soundtrack arm), so text, images, and audio
live in one 1024-d space; psytwill's COMPATIBLE_SPACES declares the
pairings. The encoder is EBind's Perception Encoder text backbone
(PE-Core-L14-336). Do not change the checkpoint, the L2 normalization, or
the ``ebind_text_{i:04d}`` naming without coordinating all three repos.

Note: spoken-word stimuli enter the shared space through THIS text arm.
EBind's audio arm (AudioSet-trained ImageBind) hears isolated spoken words
as generic speech and carries no lexical signal — measured in the
2026-08-17 mmmdata pilot (workbench ebind-shared-space).

Embedding indices are fixed-width 4-digit (``ebind_text_0000`` ..
``ebind_text_1023``) — a >999-d space per contracts §4.1.
"""

import logging

import torch

from word2psy.models.base import BaseModel

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = "encord-team/ebind-full"
EMBED_DIM = 1024
# EBind's config requires image+video+text at minimum (the PE vision
# backbone loads alongside text); audio and points are excluded.
_MODALITIES = ["image", "video", "text"]


class EBindTextModel(BaseModel):
    """Extract L2-normalised EBind text embeddings (PE-Core-L14-336 arm)."""

    name = "ebind_text"
    level = "chunk"
    # Must match viz2psy's ebind / aud2psy's ebind_audio checkpoint string
    # exactly — psytwill asserts equality before pairing shared spaces.
    checkpoint = DEFAULT_CHECKPOINT

    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINT, device: str | None = None):
        super().__init__(device=device)
        self.checkpoint = checkpoint
        self._tokenizer = None

    def load(self) -> None:
        from ebind import EBindModel as _EBind
        from ebind.configuration import EBindConfig
        from ebind.consts import PERCEPTION_ENCODER_CHECKPOINT_ARGS
        from ebind.models.perception_encoder.models import PETextProcessor

        config = EBindConfig(modalities=list(_MODALITIES))
        model = _EBind.from_pretrained(self.checkpoint, config=config)
        self.model = model.eval().to(self.device)
        pe_name = PERCEPTION_ENCODER_CHECKPOINT_ARGS["repo_id"].split("/")[1]
        # PETextProcessor tokenizes (and truncates) to PE's context length.
        self._tokenizer = PETextProcessor.from_config(pe_name)

    def predict(self, text: str) -> dict[str, float]:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[dict[str, float]]:
        tokens = torch.stack([self._tokenizer(t) for t in texts]).to(self.device)
        with torch.no_grad():
            features = self.model.forward(text=tokens)["text"].float()
            features = features / features.norm(dim=-1, keepdim=True)
        return [
            {f"ebind_text_{i:04d}": v for i, v in enumerate(row)}
            for row in features.cpu().tolist()
        ]
