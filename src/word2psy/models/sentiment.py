"""Sentiment probabilities per chunk — cardiffnlp Twitter RoBERTa.

Three-way softmax (negative / neutral / positive). Partly redundant with
the lexical valence norm but operates on whole chunks in context.
"""

import torch

from word2psy.models.base import BaseModel


class SentimentModel(BaseModel):
    """negative/neutral/positive probabilities per chunk (softmax)."""

    name = "sentiment"
    level = "chunk"

    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        device: str | None = None,
    ):
        super().__init__(device=device)
        self.model_name = model_name
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
        self._labels = [id2label[i].lower() for i in range(len(id2label))]

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
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()

        return [
            {f"sentiment_{lab}": float(p) for lab, p in zip(self._labels, row)}
            for row in probs
        ]
