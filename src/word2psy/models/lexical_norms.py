"""Lexical norms model — predicts psycholinguistic features for individual words.

Uses fastText embeddings + Ridge regression for 22 norm dimensions, plus
wordfreq for Zipf frequency.  All 23 features are word-level.
"""

import numpy as np

from word2psy.models.base import BaseModel


class LexicalNormsModel(BaseModel):
    """Predict psycholinguistic norms for arbitrary English words.

    Features: concreteness, valence, arousal, dominance, age_of_acquisition,
    imageability, familiarity, semantic_size, gender_association, socialness,
    body_object_interaction, zipf_frequency, and 11 Lancaster sensorimotor
    dimensions.
    """

    name = "lexical_norms"
    level = "word"

    def __init__(self, device: str | None = None):
        # Device is not used (no GPU model), but we keep the interface
        # consistent with BaseModel.
        super().__init__(device=device)
        self._regressors = None
        self._ft_model = None

    def load(self) -> None:
        from word2psy.norms.train import load_fasttext, load_regressors

        self._regressors = load_regressors()
        self._ft_model = load_fasttext()

    def unload(self) -> None:
        self._regressors = None
        self._ft_model = None
        super().unload()

    def predict(self, text: str) -> dict[str, float]:
        word = text.strip().lower()
        vec = self._ft_model.get_word_vector(word).reshape(1, -1)

        scores: dict[str, float] = {}
        for name, model in self._regressors.items():
            scores[name] = float(model.predict(vec)[0])

        # Zipf frequency via wordfreq
        from wordfreq import zipf_frequency

        scores["zipf_frequency"] = zipf_frequency(word, "en")

        return scores

    def predict_batch(self, texts: list[str]) -> list[dict[str, float]]:
        from wordfreq import zipf_frequency

        # Deduplicate to avoid redundant embedding lookups
        words = [t.strip().lower() for t in texts]
        unique_words = list(dict.fromkeys(words))

        # Batch embed unique words
        vecs = np.array(
            [self._ft_model.get_word_vector(w) for w in unique_words],
            dtype=np.float32,
        )

        # Predict all norms at once for unique words
        unique_scores: dict[str, dict[str, float]] = {}
        for w in unique_words:
            unique_scores[w] = {}

        for name, model in self._regressors.items():
            preds = model.predict(vecs)
            for w, p in zip(unique_words, preds):
                unique_scores[w][name] = float(p)

        for w in unique_words:
            unique_scores[w]["zipf_frequency"] = zipf_frequency(w, "en")

        # Map back to original order
        return [dict(unique_scores[w]) for w in words]
