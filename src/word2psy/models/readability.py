"""Classic readability metrics per chunk — pure Python via textstat."""

from word2psy.models.base import BaseModel


class ReadabilityModel(BaseModel):
    """Seven standard readability/complexity metrics per chunk.

    Grade-level metrics (Flesch-Kincaid, Gunning Fog, SMOG, Coleman-Liau,
    ARI, Dale-Chall) increase with complexity; Flesch Reading Ease
    decreases. Short single-word chunks yield degenerate values — these
    metrics are meaningful for sentence-or-longer chunks.
    """

    name = "readability"
    level = "chunk"

    def load(self) -> None:
        import textstat

        self.model = textstat

    def predict(self, text: str) -> dict[str, float]:
        ts = self.model
        return {
            "readability_flesch_ease": float(ts.flesch_reading_ease(text)),
            "readability_flesch_kincaid_grade": float(ts.flesch_kincaid_grade(text)),
            "readability_gunning_fog": float(ts.gunning_fog(text)),
            "readability_smog": float(ts.smog_index(text)),
            "readability_coleman_liau": float(ts.coleman_liau_index(text)),
            "readability_ari": float(ts.automated_readability_index(text)),
            "readability_dale_chall": float(ts.dale_chall_readability_score(text)),
        }
