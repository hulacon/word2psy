# word2psy

A toolbox for bulk extraction and exploration of psychological and linguistic features
from text — the verbal-domain companion to
[viz2psy](https://github.com/hulacon/viz2psy), which does the same for images and video
frames. Text is scored through a unified command-line interface wrapping computational
models from NLP and human psychology; results are stored as tabular CSV with metadata
sidecars.

A core design goal is **cross-modal comparability**: word2psy's CLIP text embeddings use
the same OpenCLIP checkpoint as viz2psy's CLIP image embeddings (ViT-B-32,
`laion2b_s34b_b79k`), so words and images can be compared directly in a shared 512-d
space.

## Features

- **Unified CLI** for scoring text files, CSV/TSV stimulus lists, or stdin
- **Word-level psycholinguistic norms** extrapolated to arbitrary English words
- **CLIP text embeddings** directly comparable to viz2psy image embeddings
- **Tidy outputs**: word-per-row CSV + chunk-per-row CSV + JSON metadata sidecar
- **Built-in visualization**: timeseries, correlation heatmaps, and 2-D projections

## Installation

Not yet on PyPI — install from source. Requires Python 3.10–3.12 (3.13+ not yet
supported due to the fastText dependency).

```bash
git clone https://github.com/hulacon/word2psy.git
cd word2psy
uv venv --python 3.11
uv pip install -e .
```

On first use, models download automatically to `~/.cache/word2psy` (override with
`WORD2PSY_CACHE`): the fastText vectors are a ~2.4 GB download (7.2 GB on disk), CLIP
weights ~600 MB via Hugging Face, and the five norm databases a few MB each. Norm
regressors are trained locally on first use (a few minutes, one time).

## Quick Start

```bash
# Score a text file with both models
word2psy lexical_norms clip_text input.txt -o features.csv

# Score a CSV stimulus list: each row is one chunk; other columns
# (IDs, conditions) pass through to the chunks output
word2psy clip_text lexical_norms stimuli.csv --text-column word \
    --id-column stim_id -o features.csv

# Score text from stdin (prints to stdout)
echo "The quick brown fox" | word2psy lexical_norms

# All models
word2psy --all input.txt -o features.csv

# Visualize results
word2psy viz timeseries features.csv -o timeseries.png
word2psy viz heatmap features.csv -o heatmap.png
word2psy viz scatter features.csv --method pca -o scatter.png
word2psy viz recommend features.csv   # suggests plots for a given CSV
```

Python API:

```python
from word2psy import score_text
from word2psy.models.lexical_norms import LexicalNormsModel
from word2psy.models.clip_text import CLIPTextModel

words_df, chunks_df = score_text(
    ["I love this product.", "Terrible experience."],
    [LexicalNormsModel(), CLIPTextModel()],
)
# words_df: one row per word, with norm columns
# chunks_df: one row per chunk, with clip_text_000...clip_text_511 columns
```

## Available Models

| Model | Level | Output | Description |
|-------|-------|--------|-------------|
| `lexical_norms` | word | 18 features | 17 psycholinguistic norms predicted from fastText embeddings via ridge regression (concreteness, valence, arousal, dominance, age of acquisition, imageability, 11 Lancaster sensorimotor dimensions), plus Zipf word frequency |
| `clip_text` | chunk | 512-d embedding | OpenCLIP ViT-B-32 (`laion2b_s34b_b79k`) text embeddings, L2-normalized, in the same space as viz2psy image embeddings |

Norm predictions are extrapolations trained on published human rating databases;
5-fold cross-validated accuracy ranges from r² ≈ 0.72 (concreteness) to ≈ 0.31
(sensorimotor head), in line with published embedding-based norm extrapolation work.

More models (word-level surprisal, sentiment/emotion, readability) are planned — see
the roadmap in [CLAUDE.md](CLAUDE.md).

## Output Format

For `word2psy <models> input -o features.csv`, three files are written, mirroring
viz2psy's one-row-per-stimulus CSV layout at each level of analysis:

- **`features_words.csv`** — one row per word. Index columns `word_idx`, `word`,
  `sentence_idx`, `chunk_idx`, `chunk_label`, `onset`, `offset` (onset/offset reserved
  for time-aligned stimuli), followed by one column per word-level feature.
- **`features_chunks.csv`** — one row per chunk: `chunk_idx`, `chunk_label`,
  `n_words`, any passthrough columns from CSV input, then one column per chunk-level
  feature — embeddings appear flat (`clip_text_000`...`clip_text_511`), exactly like a
  viz2psy image CSV, so the two are directly comparable side by side.
- **`features.meta.json`** — provenance sidecar: input stats, both output files,
  models, feature definitions, versions, device, and runtime.

The two tables join on `chunk_idx`. A wordlist CSV scored with `--text-column`
makes each word its own chunk, so `features_chunks.csv` becomes a per-word table
with CLIP embeddings — ready for cross-modal comparison with viz2psy output.

## Norm Sources

The `lexical_norms` model is trained on these published databases (please cite the
originals if you use the corresponding features):

- Brysbaert, Warriner, & Kuperman (2014) — concreteness
- Mohammad (2018), NRC VAD Lexicon — valence, arousal, dominance
- Kuperman, Stadthagen-Gonzalez, & Brysbaert (2012) — age of acquisition
- Scott et al. (2019), Glasgow Norms — imageability
- Lynott et al. (2020), Lancaster Sensorimotor Norms — 11 sensorimotor dimensions
- Speer et al., [wordfreq](https://github.com/rspeer/wordfreq) — Zipf frequency

## Hardware Requirements

- **RAM**: 16 GB recommended (the fastText model alone loads ~7 GB)
- **Disk**: ~12 GB for model weights and caches
- **GPU**: optional; CLIP uses CUDA or Apple MPS when available, CPU otherwise

## License

MIT License. See [LICENSE](LICENSE) for details.
