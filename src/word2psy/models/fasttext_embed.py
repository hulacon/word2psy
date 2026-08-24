"""fastText static word embeddings — 300-d per word.

Reuses the crawl-300d-2M-subword model already downloaded for the
lexical_norms backbone, so this model costs no extra downloads. Subword
information means every string gets a vector (no OOV), and this is the
exact space the norm predictions are extrapolated from.

**Why the dictionary row, not ``get_word_vector``.** The checkpoint is a
subword ``.bin`` with ``minn = maxn = 4``, and ``get_word_vector`` returns
the unweighted mean of a word's own input row and its character 4-grams.
A word shorter than four characters has *no* n-grams and comes back at full
length; every longer word is diluted by ``1 / (1 + n_ngrams)`` onto n-gram
rows that share a large common direction. The emitted norm is therefore
close to a step function of word length — measured on the 751 word types of
the NSD shared1000 captions, ``"a"`` came back at 3.11 and ``"skateboard"``
at 0.38, an 8x spread on a difference that is orthographic, not semantic.
Any consumer that mean-pools tokens (which is what
``pipeline.pool_word_embeddings`` does, and what every sentence-level use
does) then hands the pooled vector to the shortest, most frequent function
words: pooled over those captions the space collapsed to a participation
ratio of 2.2 against 17.1 for the dictionary rows.

So an in-vocabulary word gets its own input row — the vector the published
``crawl-300d-2M-subword.vec`` distributes — and subword composition is kept
for the OOV case it exists to serve. Coverage on ordinary English text is
near-total (751/751 on those captions), so this is the fallback path, not
the common one.
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
        word = text.strip().lower()
        word_id = self.model.get_word_id(word)
        vec = (
            self.model.get_input_vector(word_id)
            if word_id >= 0
            else self.model.get_word_vector(word)  # OOV: compose from subwords
        )
        return {f"fasttext_{i:03d}": float(v) for i, v in enumerate(vec)}
