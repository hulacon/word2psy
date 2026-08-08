"""Wordform features — length, syllables, phonemes, orthographic OLD20.

Classic lexical-decision covariates. Syllable and phoneme counts come
from CMUdict (NaN phonemes / regex-estimated syllables for words not in
the dictionary). OLD20 (Yarkoni, Balota, & Yap, 2008) is the mean
Levenshtein distance to the 20 closest words in a 40k-word frequency-
ranked lexicon.
"""

import math
import re

import numpy as np

from word2psy.models.base import BaseModel

LEXICON_SIZE = 40_000
_VOWEL_GROUPS = re.compile(r"[aeiouy]+")


def estimate_syllables(word: str) -> int:
    """Regex fallback syllable count for words not in CMUdict."""
    return max(1, len(_VOWEL_GROUPS.findall(word.lower())))


class WordformModel(BaseModel):
    """Length, syllable/phoneme counts, and OLD20 for individual words."""

    name = "wordform"
    level = "word"

    def load(self) -> None:
        import nltk

        try:
            nltk.data.find("corpora/cmudict")
        except LookupError:
            nltk.download("cmudict", quiet=True)
        from nltk.corpus import cmudict

        from wordfreq import top_n_list

        self._cmu = cmudict.dict()
        self._lexicon = top_n_list("en", LEXICON_SIZE)
        self.model = object()  # marker for loaded state

    def unload(self) -> None:
        self._cmu = None
        self._lexicon = None
        super().unload()

    def _old20(self, words: list[str]) -> np.ndarray:
        from rapidfuzz.distance import Levenshtein
        from rapidfuzz.process import cdist

        dists = cdist(
            words, self._lexicon, scorer=Levenshtein.distance, workers=-1
        ).astype(np.float32)
        # OLD20 excludes the word itself: drop one zero-distance entry
        # when the query is in the lexicon.
        out = np.empty(len(words))
        for i, word in enumerate(words):
            row = np.sort(dists[i])[:21]
            if row[0] == 0.0:
                row = row[1:]
            out[i] = row[:20].mean()
        return out

    def _counts(self, word: str) -> tuple[float, float]:
        """(n_syllables, n_phonemes) from CMUdict, with fallbacks."""
        prons = self._cmu.get(word.lower())
        if not prons:
            return float(estimate_syllables(word)), math.nan
        pron = prons[0]
        syllables = sum(1 for p in pron if p[-1].isdigit())
        return float(syllables), float(len(pron))

    def predict(self, text: str) -> dict[str, float]:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[dict[str, float]]:
        words = [t.strip() for t in texts]
        old20 = self._old20([w.lower() for w in words])

        results = []
        for word, dist in zip(words, old20):
            syllables, phonemes = self._counts(word)
            results.append(
                {
                    "length": float(len(word)),
                    "n_syllables": syllables,
                    "n_phonemes": phonemes,
                    "old20": float(dist),
                }
            )
        return results
