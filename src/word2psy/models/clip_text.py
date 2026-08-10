"""CLIP text encoder — produces 512-d L2-normalised embeddings per chunk.

Uses the same model as viz2psy (ViT-B-32, laion2b_s34b_b79k) so that text
and image embeddings live in the same shared space.
"""

import logging

import torch

from word2psy.models.base import BaseModel

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_NAME = "ViT-B-32"
_DEFAULT_PRETRAINED = "laion2b_s34b_b79k"


class CLIPTextModel(BaseModel):
    """Extract L2-normalised CLIP text embeddings.

    Uses OpenCLIP's ViT-B-32 to produce a 512-d embedding per text chunk.
    """

    name = "clip_text"
    level = "chunk"
    # Must match viz2psy's clip checkpoint string exactly — psytwill asserts
    # equality before pairing the shared text/image space.
    checkpoint = f"{_DEFAULT_MODEL_NAME}/{_DEFAULT_PRETRAINED}"

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL_NAME,
        pretrained: str = _DEFAULT_PRETRAINED,
        device: str | None = None,
    ):
        super().__init__(device=device)
        self.model_name = model_name
        self.pretrained = pretrained
        self.checkpoint = f"{model_name}/{pretrained}"
        self._tokenizer = None

    def load(self) -> None:
        import open_clip

        model, _, _ = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
        )
        self.model = model.eval().to(self.device)
        self._tokenizer = open_clip.get_tokenizer(self.model_name)

    def predict(self, text: str) -> dict[str, float]:
        tokens = self._tokenizer([text]).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        emb = features.squeeze(0).cpu().tolist()
        return {f"clip_text_{i:03d}": v for i, v in enumerate(emb)}

    def predict_batch(self, texts: list[str]) -> list[dict[str, float]]:
        tokens = self._tokenizer(texts).to(self.device)

        # Warn if any chunks are likely truncated (CLIP context length = 77 tokens)
        if tokens.shape[1] > 77:
            logger.warning(
                "Some text chunks exceed CLIP's 77-token context length and "
                "will be truncated."
            )

        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)

        results = []
        for row in features.cpu().tolist():
            results.append({f"clip_text_{i:03d}": v for i, v in enumerate(row)})
        return results
