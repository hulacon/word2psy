"""GoogleNews word2vec embeddings — 300-d per word.

Included for comparability with the large psycholinguistics literature
built on this checkpoint. For new analyses the ``fasttext`` model is
generally preferable (larger corpus, subword handling, no OOV).

Out-of-vocabulary words get NaN columns (the viz layer's PPCA projection
handles missing data).
"""

import math

from word2psy.models.base import BaseModel

DIM = 300


class Word2VecModel(BaseModel):
    """Extract 300-d GoogleNews word2vec embeddings for individual words."""

    name = "word2vec"
    level = "word"

    def load(self) -> None:
        import os

        from word2psy.norms.train import CACHE_DIR

        # Keep the ~1.7 GB gensim download inside the word2psy cache
        os.environ.setdefault("GENSIM_DATA_DIR", str(CACHE_DIR / "gensim"))

        import gensim.downloader

        self.model = gensim.downloader.load("word2vec-google-news-300")

    def predict(self, text: str) -> dict[str, float]:
        word = text.strip()
        # GoogleNews vocabulary is case-sensitive and mixed-case
        for candidate in (word, word.lower(), word.capitalize()):
            if candidate in self.model:
                vec = self.model[candidate]
                return {f"word2vec_{i:03d}": float(v) for i, v in enumerate(vec)}
        return {f"word2vec_{i:03d}": math.nan for i in range(DIM)}
