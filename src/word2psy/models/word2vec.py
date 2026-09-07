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
    checkpoint = "word2vec-google-news-300"  # gensim-data vector-set name

    def load(self) -> None:
        import os
        from pathlib import Path

        from word2psy.norms.train import CACHE_DIR

        # Keep the ~1.7 GB gensim download inside the word2psy cache
        os.environ.setdefault("GENSIM_DATA_DIR", str(CACHE_DIR / "gensim"))
        base = Path(os.environ["GENSIM_DATA_DIR"])
        cached = base / self.checkpoint / f"{self.checkpoint}.gz"

        if cached.exists():
            # Load the cached vectors directly. gensim.downloader.load()
            # re-fetches information.json from GitHub on EVERY call and
            # rewrites it truncate-then-write in the (shared) data dir, so
            # concurrent loads race: a reader can see an empty file and die
            # with "Expecting value: line 1 column 1" (seen 2/23 array tasks,
            # 2026-09-07). Once the vectors are on disk there is nothing to
            # download and no reason to touch that file — this path is also
            # what makes offline runs work.
            from gensim.models import KeyedVectors

            self.model = KeyedVectors.load_word2vec_format(str(cached), binary=True)
            return

        import gensim.downloader

        self.model = gensim.downloader.load(self.checkpoint)

    def predict(self, text: str) -> dict[str, float]:
        word = text.strip()
        # GoogleNews vocabulary is case-sensitive and mixed-case
        for candidate in (word, word.lower(), word.capitalize()):
            if candidate in self.model:
                vec = self.model[candidate]
                return {f"word2vec_{i:03d}": float(v) for i, v in enumerate(vec)}
        return {f"word2vec_{i:03d}": math.nan for i in range(DIM)}
