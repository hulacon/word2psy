"""Experiential attributes (Binder et al., 2016), predicted per word from fastText.

How strongly English speakers associate a word's referent with each of 65
experiential attributes (vision, sound, upper-limb action, fear, ...; 0-6),
extrapolated to any word by a ridge probe on its fastText vector. The probe
is trained on Binder et al.'s ratings of 535 words, which the user supplies
(``word2psy.norms.binder``); see ``word2psy.norms.experiential`` for building
the weights. Opt-in: name this model explicitly; ``lexical_norms`` is
unchanged.

Out of the rated domain: function words, proper names and multi-word strings
(the rated words are content nouns, verbs and adjectives). Outputs are linear
ridge extrapolations on the 0-6 scale, not clipped, as ``lexical_norms``.

The model emits exactly the attributes its weights hold (all 65, or the subset
they were trained on). Column prefix ``lexical_norms_experiential`` nests inside
``lexical_norms``; consumers resolve it by the longest-match rule.
"""

from __future__ import annotations

import numpy as np

from word2psy.models.base import BaseModel
from word2psy.models.fasttext_embed import FastTextModel
from word2psy.norms import binder
from word2psy.norms.experiential import BACKBONE, COLUMNS, read_weights, weights_path

assert BACKBONE == FastTextModel.checkpoint


def _labels() -> dict[str, dict]:
    cols: dict[str, dict] = {}
    for attr, col in zip(binder.ATTRIBUTES, COLUMNS):
        domain, definition = binder.ATTRIBUTES[attr]
        cols[col] = {
            "LongName": f"Experiential attribute: {attr} ({domain}; predicted)",
            "Description": (
                f"How strongly the word's referent is associated with '{definition}' "
                f"(Binder et al. 2016 attribute {attr}, {domain} domain; 0-6 scale): a ridge probe "
                f"on the word's fastText {BACKBONE} vector trained on the 535 rated words, "
                "predicted for every word including the rated ones; linear, not clipped. Out of "
                "the rated domain: function words, proper names, multi-word strings"),
        }
    return cols


class LexicalNormsExperientialModel(BaseModel):
    """Binder et al. (2016) experiential attributes, a fastText probe (opt-in)."""

    name = "lexical_norms_experiential"
    level = "word"
    checkpoint = f"{BACKBONE}+experiential_v1"
    labels = {
        "default": {"interpretability": "nameable", "derivation": "probe", "derived_from": BACKBONE},
        "columns": _labels(),
    }

    def __init__(self, device: str | None = None):
        super().__init__(device=device)
        self._W = None
        self._b = None
        self._columns: tuple[str, ...] = ()
        self._ft_model = None

    def load(self) -> None:
        from word2psy.norms.train import load_fasttext

        w = read_weights(weights_path())  # before fastText (7 GB): fail fast, and loudly
        self._W, self._b, self._columns = w["W"], w["b"], w["columns"]
        self._ft_model = load_fasttext()

    def unload(self) -> None:
        self._W = self._b = self._ft_model = None
        super().unload()

    def predict(self, text: str) -> dict[str, float]:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[dict[str, float]]:
        words = [t.strip().lower() for t in texts]
        unique = list(dict.fromkeys(words))
        # One product per word, not one matrix product per batch: BLAS may sum a
        # multi-row product in a different order, so a word's value would depend in
        # the last bits on which words share its batch. A word-level column must be a
        # function of the word alone.
        P = [np.asarray(self._ft_model.get_word_vector(w), dtype=np.float64) @ self._W + self._b
             for w in unique]
        by_word = {w: dict(zip(self._columns, map(float, row))) for w, row in zip(unique, P)}
        return [dict(by_word[w]) for w in words]
