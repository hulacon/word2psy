"""Classic readability metrics per chunk — pure Python via textstat."""

import math
import re

from word2psy.models.base import BaseModel

_FEATURES = (
    "readability_flesch_ease", "readability_flesch_kincaid_grade",
    "readability_gunning_fog", "readability_smog", "readability_coleman_liau",
    "readability_ari", "readability_dale_chall",
)


class ReadabilityModel(BaseModel):
    """Seven standard readability/complexity metrics per chunk.

    Grade-level metrics (Flesch-Kincaid, Gunning Fog, SMOG, Coleman-Liau,
    ARI, Dale-Chall) increase with complexity; Flesch Reading Ease
    decreases. Short single-word chunks yield degenerate values — these
    metrics are meaningful for sentence-or-longer chunks.
    """

    name = "readability"
    level = "chunk"
    nulls = {c: {"means": "undefined",
                 "when": "the chunk has no word characters (empty or punctuation only)"}
             for c in _FEATURES}

    def load(self) -> None:
        import textstat

        self.model = textstat

    def predict(self, text: str) -> dict[str, float]:
        # textstat scores text with no words as 0.0 on every metric, which
        # reads as a real grade level; there is nothing to measure.
        if not re.search(r"\w", text):
            return {c: math.nan for c in _FEATURES}
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
