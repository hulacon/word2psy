"""GPT-2 word-level surprisal in sentence context.

Surprisal (-log2 probability given preceding context, in bits) is a
workhorse predictor in reading-time and N400 research. This is a
context-level model: the same word gets different values at different
positions, so the pipeline scores it per chunk without deduplication.

Implementation notes:
- A BOS token is prepended so the first word gets a (sentence-initial)
  surprisal rather than NaN.
- GPT-2 BPE subword log-probs are summed per word; words are aligned to
  the raw text by greedy in-order search (robust to nltk contraction
  splits like "do" + "n't"), and unalignable tokens get NaN.
- Chunks longer than the model's context window are scored with a strided
  sliding window, so every token keeps at least ``stride`` tokens of
  preceding context.
"""

import math

import numpy as np
import torch

from word2psy.models.base import BaseModel

LN2 = math.log(2)


def find_word_spans(
    text: str, words: list[str]
) -> list[tuple[int, int] | None]:
    """Locate each word's character span in ``text`` by in-order search.

    Handles tokenizer transforms that keep substrings intact (e.g. nltk's
    "don't" -> "do", "n't"). Words that cannot be found (e.g. nltk's
    quote transforms ``````/``''``) get None.
    """
    spans: list[tuple[int, int] | None] = []
    cursor = 0
    lower = text.lower()
    for word in words:
        idx = text.find(word, cursor)
        if idx == -1:
            idx = lower.find(word.lower(), cursor)
        if idx == -1:
            spans.append(None)
        else:
            spans.append((idx, idx + len(word)))
            cursor = idx + len(word)
    return spans


def aggregate_word_logprobs(
    text: str,
    word_spans: list[tuple[int, int] | None],
    token_offsets: list[tuple[int, int]],
    token_logprobs: np.ndarray,
) -> list[float]:
    """Sum subword token log-probs into per-word log-probs.

    A token belongs to the word whose span contains the token's first
    non-whitespace character. Returns one log-prob per word; NaN when a
    word has no aligned tokens or any of its tokens lacks a log-prob.
    """
    word_lps: list[list[float]] = [[] for _ in word_spans]

    word_i = 0
    for (ts, te), lp in zip(token_offsets, token_logprobs):
        segment = text[ts:te]
        stripped = segment.lstrip()
        if not stripped:
            continue
        eff_start = ts + (len(segment) - len(stripped))

        # Advance to the word whose span could contain this token
        while word_i < len(word_spans) and (
            word_spans[word_i] is None or word_spans[word_i][1] <= eff_start
        ):
            word_i += 1
        if word_i >= len(word_spans):
            break
        ws, we = word_spans[word_i]
        if ws <= eff_start < we:
            word_lps[word_i].append(float(lp))

    results = []
    for span, lps in zip(word_spans, word_lps):
        if span is None or not lps or any(math.isnan(v) for v in lps):
            results.append(math.nan)
        else:
            results.append(sum(lps))
    return results


class GPT2SurprisalModel(BaseModel):
    """Word surprisal (bits) from GPT-2 given preceding context."""

    name = "gpt2_surprisal"
    level = "context"

    def __init__(
        self,
        model_name: str = "gpt2",
        device: str | None = None,
        max_context: int = 1024,
        stride: int = 512,
    ):
        super().__init__(device=device)
        self.model_name = model_name
        self.max_context = max_context
        self.stride = stride
        self._tokenizer = None

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = (
            AutoModelForCausalLM.from_pretrained(self.model_name)
            .eval()
            .to(self.device)
        )

    def _token_logprobs(self, ids: list[int]) -> np.ndarray:
        """Natural-log prob of each token given preceding tokens (strided)."""
        n = len(ids)
        lps = np.full(n, np.nan)
        filled = 1  # token 0 has no context

        while filled < n:
            begin = max(0, filled - self.stride)
            end = min(begin + self.max_context, n)
            window = torch.tensor([ids[begin:end]], device=self.device)
            with torch.no_grad():
                logits = self.model(window).logits[0].float()
            logprobs = torch.log_softmax(logits, dim=-1)
            # Token at absolute position p is predicted by logits row p-1-begin
            targets = torch.tensor(ids[begin + 1 : end])
            rows = torch.arange(end - begin - 1)
            token_lps = logprobs[rows, targets].cpu().numpy()
            lps[filled:end] = token_lps[filled - begin - 1 :]
            filled = end

        return lps

    def predict_context(
        self, chunk_text: str, words: list[str]
    ) -> list[dict[str, float]]:
        enc = self._tokenizer(
            chunk_text, return_offsets_mapping=True, add_special_tokens=False
        )
        ids = enc["input_ids"]
        offsets = enc["offset_mapping"]

        if not ids:
            return [{"gpt2_surprisal": math.nan} for _ in words]

        # Prepend BOS so the first real token has context
        bos = self._tokenizer.bos_token_id
        lps = self._token_logprobs([bos] + ids)[1:]  # drop BOS position

        spans = find_word_spans(chunk_text, words)
        word_lps = aggregate_word_logprobs(chunk_text, spans, offsets, lps)

        return [
            {"gpt2_surprisal": (-lp / LN2 if not math.isnan(lp) else math.nan)}
            for lp in word_lps
        ]

    def predict(self, text: str) -> dict[str, float]:
        """Surprisal of ``text`` as a single sentence-initial word."""
        return self.predict_context(text, [text])[0]

    def unload(self) -> None:
        self._tokenizer = None
        super().unload()
