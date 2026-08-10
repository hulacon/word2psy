"""fastText static word embeddings — 300-d per word.

Reuses the crawl-300d-2M-subword model already downloaded for the
lexical_norms backbone, so this model costs no extra downloads. Subword
information means every string gets a vector (no OOV), and this is the
exact space the norm predictions are extrapolated from.
"""

from word2psy.models.base import BaseModel


class FastTextModel(BaseModel):
    """Extract 300-d fastText embeddings for individual words."""

    name = "fasttext"
    level = "word"
    checkpoint = "crawl-300d-2M-subword"  # fastText vector-set name (.bin)

    def load(self) -> None:
        from word2psy.norms.train import load_fasttext

        self.model = load_fasttext()

    def predict(self, text: str) -> dict[str, float]:
        vec = self.model.get_word_vector(text.strip().lower())
        return {f"fasttext_{i:03d}": float(v) for i, v in enumerate(vec)}
