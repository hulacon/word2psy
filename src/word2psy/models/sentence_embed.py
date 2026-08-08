"""MiniLM sentence embeddings — 384-d per chunk.

A text-only semantic space (all-MiniLM-L6-v2 via sentence-transformers),
sharper for text-to-text similarity than CLIP's multimodal space. Use
clip_text when comparing against viz2psy image embeddings; use this for
purely verbal similarity structure.
"""

from word2psy.models.base import BaseModel


class SentenceEmbedModel(BaseModel):
    """L2-normalized 384-d MiniLM sentence embeddings per chunk."""

    name = "minilm"
    level = "chunk"

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
    ):
        super().__init__(device=device)
        self.model_name = model_name

    def load(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.model_name, device=str(self.device))

    def predict(self, text: str) -> dict[str, float]:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[dict[str, float]]:
        embeddings = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return [
            {f"minilm_{i:03d}": float(v) for i, v in enumerate(row)}
            for row in embeddings
        ]
