"""Text tokenization and word-per-row DataFrame construction."""

import math
import re

import nltk
import pandas as pd

# Ensure punkt tokenizer data is available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def _is_punctuation(token: str) -> bool:
    """Return True if the token consists entirely of punctuation."""
    return bool(re.fullmatch(r"[^\w]+", token))


def split_by_sentence(
    chunks: list[str],
    chunk_labels: list[str] | None = None,
) -> tuple[list[str], list[str], list[int]]:
    """Split chunks into one chunk per sentence.

    Returns
    -------
    sentences : list[str]
        One entry per sentence, across all input chunks.
    labels : list[str]
        ``"{original_label}/s{j}"`` with j the sentence index within its
        original chunk (original labels default to ``chunk_{i}``).
    origin : list[int]
        Index of the original chunk each sentence came from (for expanding
        per-chunk passthrough data).
    """
    if chunk_labels is None:
        chunk_labels = [f"chunk_{i}" for i in range(len(chunks))]

    sentences: list[str] = []
    labels: list[str] = []
    origin: list[int] = []
    for i, (chunk_text, label) in enumerate(zip(chunks, chunk_labels)):
        for j, sentence in enumerate(nltk.sent_tokenize(chunk_text)):
            sentences.append(sentence)
            labels.append(f"{label}/s{j}")
            origin.append(i)
    return sentences, labels, origin


def tokenize_text(
    text: str | list[str],
    *,
    chunk_labels: list[str] | None = None,
    keep_punctuation: bool = False,
) -> pd.DataFrame:
    """Tokenize text into a word-per-row DataFrame.

    Parameters
    ----------
    text : str or list[str]
        Either a single string (treated as one chunk) or a list of strings
        (each element is a separate chunk).
    chunk_labels : list[str], optional
        Labels for each chunk. If ``text`` is a list, must match its length.
        Defaults to ``"chunk_0"``, ``"chunk_1"``, etc.
    keep_punctuation : bool
        If False (default), punctuation-only tokens are dropped.

    Returns
    -------
    pd.DataFrame
        Columns: word_idx, word, sentence_idx, chunk_idx, chunk_label,
        onset, offset.  onset/offset default to NaN.
    """
    if isinstance(text, str):
        chunks = [text]
    else:
        chunks = list(text)

    if chunk_labels is None:
        chunk_labels = [f"chunk_{i}" for i in range(len(chunks))]
    elif len(chunk_labels) != len(chunks):
        raise ValueError(
            f"chunk_labels length ({len(chunk_labels)}) does not match "
            f"number of chunks ({len(chunks)})"
        )

    rows: list[dict] = []
    word_idx = 0
    global_sentence_idx = 0

    for chunk_idx, (chunk_text, chunk_label) in enumerate(
        zip(chunks, chunk_labels)
    ):
        sentences = nltk.sent_tokenize(chunk_text)
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            for token in words:
                if not keep_punctuation and _is_punctuation(token):
                    continue
                rows.append(
                    {
                        "word_idx": word_idx,
                        "word": token,
                        "sentence_idx": global_sentence_idx,
                        "chunk_idx": chunk_idx,
                        "chunk_label": chunk_label,
                        "onset": math.nan,
                        "offset": math.nan,
                    }
                )
                word_idx += 1
            global_sentence_idx += 1

    columns = [
        "word_idx",
        "word",
        "sentence_idx",
        "chunk_idx",
        "chunk_label",
        "onset",
        "offset",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)
