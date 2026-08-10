"""Discrete emotion probabilities per chunk — GoEmotions RoBERTa.

The text-domain analog of viz2psy's EmoNet: 27 emotion categories plus
neutral, scored as independent (multi-label sigmoid) probabilities.
Model: SamLowe/roberta-base-go_emotions, trained on Reddit comments
annotated with the GoEmotions taxonomy (Demszky et al., 2020).
"""

import torch

from word2psy.models.base import BaseModel


class EmotionModel(BaseModel):
    """28 GoEmotions probabilities per chunk (multi-label, sigmoid)."""

    name = "emotion"
    level = "chunk"
    checkpoint = "SamLowe/roberta-base-go_emotions"  # Hugging Face model id

    def __init__(
        self,
        model_name: str = "SamLowe/roberta-base-go_emotions",
        device: str | None = None,
    ):
        super().__init__(device=device)
        self.model_name = model_name
        self.checkpoint = model_name
        self._tokenizer = None
        self._labels: list[str] = []

    def load(self) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(self.model_name)
            .eval()
            .to(self.device)
        )
        id2label = self.model.config.id2label
        self._labels = [id2label[i] for i in range(len(id2label))]

    def unload(self) -> None:
        self._tokenizer = None
        super().unload()

    def predict(self, text: str) -> dict[str, float]:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[dict[str, float]]:
        enc = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits
        probs = torch.sigmoid(logits.float()).cpu().numpy()

        return [
            {f"emotion_{lab}": float(p) for lab, p in zip(self._labels, row)}
            for row in probs
        ]
