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
- **Interactive dashboard**: a self-contained HTML explorer (`word2psy viz browse`)
  with timeseries, 2-D/3-D clustering, and trajectory views — click any word to
  see its full feature profile in context, and browse the whole stimulus set

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
`WORD2PSY_CACHE`): the fastText vectors are a ~2.4 GB download (7.2 GB on disk),
word2vec ~1.7 GB, CLIP ~600 MB, GPT-2 ~550 MB, the sentiment and emotion RoBERTas
~500 MB each, MiniLM ~90 MB, and the seven norm databases a few MB each. Norm
regressors are trained locally on first use (a few minutes, one time). Models are
loaded and freed one at a time, so peak memory is set by the largest requested model
(fastText, ~7 GB) rather than the sum.

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

# Sentence-level scoring: every sentence becomes its own chunk
word2psy sentiment emotion input.txt --by-sentence -o features.csv

# Visualize results. Every viz subcommand accepts the base path you gave to -o
# (features.csv) or either output file (features_words.csv / features_chunks.csv);
# a base path reads the words table unless you ask for --level chunks.
word2psy viz timeseries features.csv -o timeseries.png
word2psy viz heatmap features.csv --level chunks -o heatmap.png
word2psy viz scatter features.csv --method pca -o scatter.png
word2psy viz recommend features.csv   # suggests plots for a given CSV

# Interactive HTML dashboard (opens in your browser)
word2psy viz browse features.csv -o viewer.html --open
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
| `lexical_norms` | word | 23 features | 22 psycholinguistic norms predicted from fastText embeddings via ridge regression (concreteness, valence, arousal, dominance, age of acquisition, imageability, familiarity, semantic size, gender association, socialness, body-object interaction, 11 Lancaster sensorimotor dimensions), plus Zipf word frequency |
| `wordform` | word | 4 features | Length, syllable and phoneme counts (CMUdict), and orthographic OLD20 neighborhood distance |
| `fasttext` | word | 300-d embedding | fastText `crawl-300d-2M-subword` static embeddings; subword-based, so every string gets a vector (no OOV) |
| `word2vec` | word | 300-d embedding | GoogleNews word2vec embeddings for comparability with the legacy literature; out-of-vocabulary words get NaN |
| `gpt2_surprisal` | context | 1 feature | Word surprisal in bits (−log₂ probability given preceding context) from GPT-2; the same word gets different values at different positions |
| `sentiment` | chunk | 3 features | Negative/neutral/positive probabilities (cardiffnlp RoBERTa) |
| `emotion` | chunk | 28 features | GoEmotions category probabilities (multi-label RoBERTa) — the text analog of viz2psy's EmoNet |
| `readability` | chunk | 7 features | Flesch, Flesch-Kincaid, Gunning Fog, SMOG, Coleman-Liau, ARI, Dale-Chall |
| `minilm` | chunk | 384-d embedding | all-MiniLM-L6-v2 sentence embeddings — text-only semantic space, sharper than CLIP for verbal similarity |
| `clip_text` | chunk | 512-d embedding | OpenCLIP ViT-B-32 (`laion2b_s34b_b79k`) text embeddings, L2-normalized, in the same space as viz2psy image embeddings |

Levels: **word** features depend on the word type alone and land in the words CSV;
**context** features are word-level but depend on surrounding text (no
deduplication); **chunk** features land in the chunks CSV.

Norm predictions are extrapolations trained on published human rating databases;
5-fold cross-validated accuracy ranges from r² ≈ 0.72 (concreteness) to ≈ 0.31
(sensorimotor head), in line with published embedding-based norm extrapolation work.

## Output Format

For `word2psy <models> input -o features.csv`, three files are written, mirroring
viz2psy's one-row-per-stimulus CSV layout at each level of analysis:

- **`features_words.csv`** — one row per word. Index columns `word_idx`, `word`,
  `sentence_idx`, `chunk_idx`, `chunk_label`, `onset`, `offset` (onset/offset reserved
  for time-aligned stimuli), followed by one column per word-level feature.
- **`features_chunks.csv`** — one row per chunk: `chunk_idx`, `chunk_label`,
  `n_words`, any passthrough columns from CSV input, then one column per chunk-level
  feature — embeddings appear flat (`clip_text_000`...`clip_text_511`), exactly like a
  viz2psy image CSV, so the two are directly comparable side by side. When word-level
  models are run, per-chunk aggregates of their features are appended as
  `{feature}_{mean,sd,min,max}` columns (disable with `--no-word-aggregates`).
- **`features.meta.json`** — provenance sidecar: input stats, both output files,
  models, feature definitions, versions, device, and runtime.

The two tables join on `chunk_idx`. A wordlist CSV scored with `--text-column`
makes each word its own chunk, so `features_chunks.csv` becomes a per-word table
with CLIP embeddings — ready for cross-modal comparison with viz2psy output.

## Cross-modal similarity with viz2psy

`clip_text` shares its checkpoint with viz2psy's `clip` image model, so text and
image embeddings live in one 512-d space. `word2psy crossmodal` joins the two
tools' CSVs into a text × image cosine-similarity matrix:

```bash
viz2psy clip images/*.png -o image_scores.csv          # in viz2psy
word2psy clip_text words.csv --text-column word -o text_scores.csv
word2psy crossmodal text_scores.csv image_scores.csv -o similarity.csv
```

It prints the top-k images per text chunk and saves the full matrix. Raw CLIP
text–image similarities sit in a narrow band (matches ≈ 0.25–0.35); relative
comparisons within your stimulus set are what carry signal.

## Norm Sources

The `lexical_norms` model is trained on these published databases (please cite the
originals if you use the corresponding features):

- Brysbaert, Warriner, & Kuperman (2014) — concreteness
- Mohammad (2018), NRC VAD Lexicon — valence, arousal, dominance
- Kuperman, Stadthagen-Gonzalez, & Brysbaert (2012) — age of acquisition
- Scott et al. (2019), Glasgow Norms — imageability, familiarity, semantic size,
  gender association
- Lynott et al. (2020), Lancaster Sensorimotor Norms — 11 sensorimotor dimensions
- Diveica, Pexman, & Binney (2023) — socialness
- Pexman et al. (2019) — body-object interaction
- Speer et al., [wordfreq](https://github.com/rspeer/wordfreq) — Zipf frequency

## Hardware Requirements

- **RAM**: 16 GB recommended (peak ~6 GB with all models, since models load one at a
  time)
- **Disk**: ~12 GB for model weights and caches
- **GPU**: optional; CLIP uses CUDA or Apple MPS when available, CPU otherwise

## License

MIT License. See [LICENSE](LICENSE) for details.
