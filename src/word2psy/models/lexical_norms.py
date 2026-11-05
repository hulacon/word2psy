"""Lexical norms model — predicts psycholinguistic features for individual words.

Uses fastText embeddings + Ridge regression for 23 norm dimensions, plus
wordfreq for Zipf frequency.  All 24 features are word-level.
"""

import numpy as np

from word2psy.models.base import BaseModel
from word2psy.models.fasttext_embed import FastTextModel
from word2psy.norms.train import NORM_DIMENSIONS

# The human norm list each ridge regressor is trained on (norms/download.py).
_NORM_SOURCES = {
    "brysbaert_concreteness": "Brysbaert, Warriner & Kuperman (2014) concreteness ratings",
    "nrc_vad": "NRC VAD Lexicon (Mohammad, 2018)",
    "kuperman_aoa": "Kuperman, Stadthagen-Gonzalez & Brysbaert (2012) age-of-acquisition ratings",
    "glasgow": "Glasgow Norms (Scott et al., 2019)",
    "lancaster_sensorimotor": "Lancaster Sensorimotor Norms (Lynott et al., 2020)",
    "socialness": "Diveica, Pexman & Binney (2023) socialness ratings",
    "boi": "Pexman et al. (2019) body-object interaction ratings",
    "animacy": "VanArsdall & Blunt (2022) living/nonliving animacy ratings",
}


def _norm_labels() -> dict[str, dict]:
    """One `probe` entry per ridge-predicted dimension, plus Zipf (`formula`).

    No `derivation` in the model default: Zipf is a lookup, and a key-wise
    merge cannot drop the probes' `derived_from` from it.
    """
    parent = FastTextModel.checkpoint  # exactly as the fasttext model records it
    cols: dict[str, dict] = {}
    for dim, (db, _) in NORM_DIMENSIONS.items():
        if dim.startswith("sensorimotor_"):
            long_name = f"Sensorimotor strength: {dim.removeprefix('sensorimotor_')} (predicted)"
        else:
            long_name = f"{dim.replace('_', ' ').capitalize()} (predicted)"
        cols[f"lexical_norms_{dim}"] = {
            "derivation": "probe",
            "derived_from": parent,
            "LongName": long_name,
            "Description": (f"Ridge regression on the word's fastText {parent} vector, "
                            f"trained on the {_NORM_SOURCES[db]}; predicted for every word, "
                            "including words the list rates"),
        }
    cols["lexical_norms_age_of_acquisition"]["Units"] = "years"
    cols["lexical_norms_zipf_frequency"] = {
        "derivation": "formula",
        "LongName": "Zipf word frequency",
        "Description": "wordfreq lookup: log10 of frequency per billion words (Zipf scale)",
    }
    return cols


class LexicalNormsModel(BaseModel):
    """Predict psycholinguistic norms for arbitrary English words.

    Features: concreteness, valence, arousal, dominance, age_of_acquisition,
    imageability, familiarity, semantic_size, gender_association, socialness,
    body_object_interaction, animacy (VanArsdall & Blunt living/nonliving),
    zipf_frequency, and 11 Lancaster sensorimotor dimensions.
    """

    name = "lexical_norms"
    level = "word"
    # Trained probe: Ridge regressors (fit locally, cached) on the fastText
    # crawl-300d-2M-subword backbone.
    checkpoint = "crawl-300d-2M-subword+ridge"
    labels = {"default": {"interpretability": "nameable"}, "columns": _norm_labels()}

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
            scores[f"lexical_norms_{name}"] = float(model.predict(vec)[0])

        # Zipf frequency via wordfreq
        from wordfreq import zipf_frequency

        scores["lexical_norms_zipf_frequency"] = zipf_frequency(word, "en")

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
                unique_scores[w][f"lexical_norms_{name}"] = float(p)

        for w in unique_words:
            unique_scores[w]["lexical_norms_zipf_frequency"] = zipf_frequency(w, "en")

        # Map back to original order
        return [dict(unique_scores[w]) for w in words]
