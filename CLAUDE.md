# CLAUDE.md

## What this project is

word2psy extracts numerical psychological and linguistic features from **text**, mirroring
[viz2psy](https://github.com/hulacon/viz2psy), which does the same for **images and video
frames**. The audience is psychology / cognitive-neuroscience researchers who need bulk
numerical summaries of verbal stimuli. The primary unit of analysis is the **single word**;
chunk-level (sentence/passage) scoring exists, and richer phrase- and sentence-level
functionality is an aspirational goal.

Design decisions should mirror viz2psy wherever it makes sense: same CLI feel, same
CSV-plus-sidecar output layout, same "registry of wrapped models" architecture.

### Cross-modal compatibility is a hard requirement

`CLIPTextModel` uses the **same OpenCLIP checkpoint as viz2psy's CLIP image model**
(`ViT-B-32`, pretrained `laion2b_s34b_b79k`), so text embeddings from word2psy and image
embeddings from viz2psy live in one shared 512-d space and can be directly compared
(e.g., cosine similarity between a word and a picture). Do not change this checkpoint,
its L2 normalization, or the `clip_text_{i:03d}` column naming without coordinating a
matching change in viz2psy.

## Architecture (`src/word2psy/`)

- **`models/base.py`** — `BaseModel` ABC. Subclasses set class attrs `name` and `level`
  (`"word"` or `"chunk"`) and implement `load()` and `predict(text) -> dict[str, float]`;
  override `predict_batch()` for real batching. Device auto-detect (cuda → mps → cpu).
- **`models/`** — implemented models: `clip_text` (chunk-level, 512-d embeddings),
  `lexical_norms` (word-level, 17 psycholinguistic features).
- **`tokenize.py`** — nltk (punkt) sentence + word tokenization into the word-per-row
  DataFrame. Punctuation tokens dropped by default (`--keep-punctuation` to keep).
- **`pipeline.py`** — `score_text(text, models)` returns `(df, embeddings)`. Word-level
  models append scalar columns to the DataFrame (deduplicated by unique word before
  inference); chunk-level embedding models return arrays saved separately to HDF5 via
  `save_embeddings()`.
- **`metadata.py`** — builds the `.json` sidecar documenting inputs, models, features,
  timing, and device.
- **`norms/`** — the lexical-norms subsystem:
  - `download.py` fetches 5 public norm databases (Brysbaert concreteness, NRC VAD,
    Kuperman AoA, Glasgow imageability, Lancaster sensorimotor) and caches them as
    parquet. `NORM_SOURCES` holds URLs + parsing specs.
  - `train.py` fits `RidgeCV` regressors on fastText `crawl-300d-2M-subword` vectors to
    extrapolate each norm to arbitrary words; caches fitted models as joblib.
    `NORM_DIMENSIONS` is the single source of truth for norm feature names — the CLI's
    metadata step imports it too. Multi-word norm entries are dropped at training time
    and CV folds are shuffled (norm files are ordered; unshuffled CV gave a bogus
    negative r² for concreteness). Baseline 5-fold CV r² (Aug 2026): concreteness .72,
    imageability .69, AoA .60, valence .55, arousal/dominance .45, sensorimotor
    .31–.53 — consistent with published embedding→norm extrapolation results.
- **`viz/`** — matplotlib/seaborn plots: `timeseries`, `heatmap` (feature correlations),
  `scatter` (PCA/PPCA/UMAP/t-SNE/MDS projections), plus `feature_config.py` which
  detects which models produced a CSV's columns and recommends visualizations.
- **`cli.py`** — single `word2psy` entry point. `MODEL_REGISTRY` maps model name →
  (module path, class name, description); models are imported lazily so `--help` and
  `--list-models` stay fast. `word2psy viz <subcommand>` routes to the viz layer.
  **To add a model**: create `models/<name>.py` subclassing `BaseModel`, add a registry
  entry, add tests.

## Data structures (mirror viz2psy)

- **Primary output**: word-per-row CSV. Index columns: `word_idx`, `word`,
  `sentence_idx`, `chunk_idx`, `chunk_label`, `onset`, `offset` (onset/offset are
  reserved for time-aligned stimuli such as narration transcripts; NaN by default).
  Feature columns are appended flat, one per scalar feature.
- **Chunk embeddings**: `<output>.h5` sidecar with a `chunk_index` dataset plus one
  `{model_name}_embeddings` dataset per embedding model (gzip-compressed float32).
- **Metadata**: `<output>.json` sidecar.
- **Feature naming**: embeddings `"{model}_{i:03d}"` (e.g. `clip_text_000`); norms use
  plain names (`concreteness`, `valence`, …) and `sensorimotor_*` for Lancaster
  dimensions.

## Cache and downloads

Everything heavy lives in `~/.cache/word2psy` (override with `WORD2PSY_CACHE`):
`norms/*.parquet`, `models/*.joblib` (trained regressors), and
`fasttext/crawl-300d-2M-subword.bin` (**~7 GB**, one-time download). CLIP weights
(~400 MB) go to open_clip's own HuggingFace cache. Norm regressors are trained lazily
on first use of `lexical_norms`.

Norm source URLs (`norms/download.py`) point at third-party hosts (OSF, Springer,
saifmohammad.com, GitHub) and can rot — if a download fails, check the URL before
assuming a code bug.

## Dev environment

- Use `uv` for environments. **Pin Python ≤ 3.12**: `fasttext-wheel` does not build on
  newer Pythons (the machine's system Python is 3.14 — don't use it for this project).
- A `.venv` (Python 3.11) exists at the repo root; use `.venv/bin/python` /
  `.venv/bin/pytest`. Recreate with: `uv venv --python 3.11 && uv pip install -e ".[dev]"`
- Tests: `pytest`. Only `test_tokenize` and `test_viz` are truly offline.
  **`test_pipeline` uses the real models** — on a cold cache it silently downloads
  CLIP (~2.2 GB), the fastText zip (~2.4 GB → 7.2 GB extracted), and all norm
  databases, and trains the regressors. `test_clip_text` and `test_lexical_norms`
  likewise. With warm caches (the normal state on this machine) the whole suite is
  fine to run. TODO: add a `slow`/`integration` pytest marker + lightweight fake
  models so a genuinely offline unit suite exists.

## Known gaps (as of Aug 2026)

- Verified working as of Aug 2026: editable install on Python 3.11; full test suite
  (56 tests); all five norm-database downloads incl. parsing sanity checks; both
  models run end-to-end through the CLI (CSV + HDF5 + metadata sidecars all
  well-formed; lexical_norms predictions show strong face validity).
- `README.md` is aspirational and out of sync: it advertises models (sentiment,
  emotion, readability, topics, liwc, ner, morality) that don't exist, a
  `word2psy-viz` entry point that isn't defined, and Plotly dashboards (the viz layer
  is matplotlib). Only `clip_text` and `lexical_norms` are implemented. Keep README
  claims matched to `MODEL_REGISTRY`.
- Not published to PyPI. `pyproject.toml` URLs say `github.com/bhutch/word2psy`; the
  sibling project lives under `github.com/hulacon` — reconcile before publishing.

## Roadmap

Ordered; items become "next up" as their predecessors land.

1. **Phase 2 — first full end-to-end run** (done Aug 2026 except README): all caches
   populated (CLIP, fastText, norms, trained regressors); both models validated
   end-to-end via the CLI with strong face validity on a sample word list.
   **Pending**: README rewrite to match reality (see Known gaps).
2. **Interactive HTML dashboard** (viz2psy parity — its `--browse` viewer is the
   reference). A `word2psy viz browse scores.csv -o viewer.html` style command emitting
   a self-contained interactive HTML file (Plotly or similar) for exploring scores:
   feature distributions, timeseries along the word sequence, 2-D projections with
   word-level hover. Should reuse `viz/feature_config.py`'s model-detection logic to
   decide which panels to show. The README already promises Plotly dashboards — this
   item is what makes that true.
3. **Cross-modal demo**: cosine similarity between word2psy `clip_text` embeddings and
   viz2psy image embeddings in the shared space; becomes the flagship README example.
4. **Model expansion** (value-per-effort order for word stimuli): GPT-2 word-level
   surprisal; sentiment/emotion (chunk-level transformers); readability (pure Python);
   then topics / NER / moral foundations.
5. **Phrase/sentence-level features** (aspirational): sentence-transformer embeddings;
   decide compositional vs. direct scoring for phrase-level norms.
