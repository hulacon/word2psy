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
  (`"word"`, `"chunk"`, or `"context"`) and implement `load()` and
  `predict(text) -> dict[str, float]`; override `predict_batch()` for real batching, or
  `predict_context(chunk_text, words)` for context-level models. Device auto-detect
  (cuda → mps → cpu). `unload()` frees weights — the pipeline unloads each model after
  scoring so peak RAM is one model at a time (the machine has 16 GB; fastText alone is
  ~7 GB resident).
- **`models/`** — implemented (10): word-level `lexical_norms` (23 features),
  `wordform` (length/syllables/phonemes/OLD20), `fasttext` (300-d, reuses the norms
  backbone), `word2vec` (300-d GoogleNews via gensim, NaN for OOV); context-level
  `gpt2_surprisal` (bits, BOS-prepended, strided beyond 1024 tokens); chunk-level
  `sentiment` (3, cardiffnlp RoBERTa), `emotion` (28, GoEmotions RoBERTa — EmoNet
  analog), `readability` (7, textstat), `minilm` (384-d sentence-transformers),
  `clip_text` (512-d).
  Levels: "word" = function of word type (deduplicated before inference); "context" =
  word-level but position-dependent (no dedup, scored chunk by chunk); "chunk" = one
  row per chunk.
- **`tokenize.py`** — nltk (punkt) sentence + word tokenization into the word-per-row
  DataFrame. Punctuation tokens dropped by default (`--keep-punctuation` to keep).
- **`pipeline.py`** — `score_text(text, models)` returns `(words_df, chunks_df)`.
  Word-level models append columns to the words table (deduplicated by unique word
  before inference); chunk-level models append columns to the chunks table
  (embeddings flat). `by_sentence=True` (CLI `--by-sentence`) re-chunks input so
  each sentence is its own chunk (labels `{orig}/s{j}`, passthrough rows repeated).
  `aggregate_word_features` (on by default, CLI `--no-word-aggregates`) appends
  per-chunk `{feature}_{mean,sd,min,max}` of word-level scalar features to the
  chunks table — NaN-aware, sd ddof=1 (NaN for 1-word chunks), embeddings
  excluded; surfaced in the dashboard via the `word_aggregates` pseudo-config.
- **`metadata.py`** — builds the `.json` sidecar documenting inputs, models, features,
  timing, and device.
- **`crossmodal.py`** — text × image cosine similarity in the shared CLIP space:
  joins a word2psy chunks CSV (`clip_text_###`) with a viz2psy CSV (`clip_###`;
  the regexes keep the two schemes apart even in a combined frame), re-L2-normalizes
  defensively, returns a labeled DataFrame. CLI: `word2psy crossmodal text.csv
  image.csv -o sim.csv --top-k N` (text path goes through the same
  `resolve_scores_paths` as `viz browse`).
- **`norms/`** — the lexical-norms subsystem:
  - `download.py` fetches 5 public norm databases (Brysbaert concreteness, NRC VAD,
    Kuperman AoA, Glasgow imageability, Lancaster sensorimotor) and caches them as
    parquet. `NORM_SOURCES` holds URLs + parsing specs.
  - `train.py` fits `RidgeCV` regressors on fastText `crawl-300d-2M-subword` vectors to
    extrapolate each norm to arbitrary words; caches fitted models as joblib.
    `NORM_DIMENSIONS` is the single source of truth for norm feature names. Multi-word
    norm entries are dropped at training time and CV folds are shuffled (norm files are
    ordered; unshuffled CV gave a bogus negative r² for concreteness). `load_norm`
    re-downloads a cached parquet if the source spec gained columns. Baseline 5-fold
    CV r² (Aug 2026): concreteness .72, imageability .69, socialness/semantic_size
    .63–.64, gender_association/BOI .61, AoA .60, familiarity .58, valence .55,
    arousal/dominance .45, sensorimotor .31–.53 — consistent with published
    embedding→norm extrapolation results.
- **`viz/`** — matplotlib/seaborn plots: `timeseries`, `heatmap` (feature correlations),
  `scatter` (PCA/PPCA/UMAP/t-SNE/MDS projections), plus `feature_config.py` which
  detects which models produced a CSV's columns (all 10 models, with word/chunk
  `level` awareness) and recommends visualizations. `dashboard.py` builds the
  `viz browse` interactive HTML dashboard: projections and per-row data are
  precomputed in Python and embedded as JSON into a self-contained HTML template
  (Plotly.js from CDN, no plotly Python dependency); overview views are
  timeseries / MDS-or-PCA clustering (2-D/3-D, word labels on points) /
  trajectory, and clicking a point opens a detail overlay that renders the word
  large (viz2psy's image analog) with sentence context, feature-panel dropdown,
  and a slider to browse the whole set. MDS is used up to `mds_max` (500) rows,
  PCA beyond; rows with NaN embeddings (word2vec OOV) are dropped from
  projections via an `indices` mapping.
- **`cli.py`** — single `word2psy` entry point. `MODEL_REGISTRY` maps model name →
  (module path, class name, description); models are imported lazily so `--help` and
  `--list-models` stay fast. `word2psy viz <subcommand>` routes to the viz layer.
  **To add a model**: create `models/<name>.py` subclassing `BaseModel`, add a registry
  entry, add tests.

## Data structures (mirror viz2psy)

`-o scores.csv` produces three files (I/O v2, Aug 2026 — HDF5 was dropped in favor
of viz2psy-style flat CSVs at each level):

- **`scores_words.csv`** — one row per word token. Index columns: `word_idx`, `word`,
  `sentence_idx`, `chunk_idx`, `chunk_label`, `onset`, `offset` (onset/offset are
  reserved for time-aligned stimuli such as narration transcripts; NaN by default).
  Word-level feature columns appended flat.
- **`scores_chunks.csv`** — one row per chunk: `chunk_idx`, `chunk_label`, `n_words`,
  passthrough columns from tabular input, then chunk-level features flat —
  embeddings included (`clip_text_000`…`511`). This file is the structural twin of a
  viz2psy image CSV; cross-modal comparisons happen here. Joins to the words file on
  `chunk_idx`.
- **`scores.meta.json`** — provenance sidecar (both output files recorded).
- **Input**: `.txt` files (each file = one chunk), stdin (one chunk), or a single
  `.csv`/`.tsv` with `--text-column` (each row = one chunk; `--id-column` sets
  `chunk_label`; all other columns pass through to the chunks file).
- **Feature naming**: embeddings `"{model}_{i:03d}"` (e.g. `clip_text_000`); norms use
  plain names (`concreteness`, `valence`, …) and `sensorimotor_*` for Lancaster
  dimensions.
- `score_text(text, models, ...)` returns `(words_df, chunks_df)`; after scoring,
  each model instance exposes `feature_names_`. Chunk-level scalar models (future
  sentiment etc.) need no special handling — their columns land in the chunks table.

## Cache and downloads

Everything heavy lives in `~/.cache/word2psy` (override with `WORD2PSY_CACHE`):
`norms/*.parquet` (7 databases), `models/*.joblib` (22 trained regressors),
`fasttext/crawl-300d-2M-subword.bin` (**~7 GB**, one-time download), and
`gensim/` (word2vec, ~1.7 GB). Transformer weights (CLIP, GPT-2, the two RoBERTas,
MiniLM — ~2 GB total) go to the HuggingFace cache. Norm regressors are trained lazily
on first use of `lexical_norms`. All caches are fully populated on this machine as of
Aug 2026.

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
  (158 tests); all seven norm-database downloads incl. parsing sanity checks; all 10
  models run end-to-end through the CLI in one `--all` invocation (43 s, ~5.7 GB peak
  RSS; words CSV 635 cols, chunks CSV 937 cols on a 2-sentence test), with strong
  face validity across norms, surprisal, OLD20, emotion, and sentiment.
- `README.md` was rewritten (Aug 2026) to match reality — keep its claims matched to
  `MODEL_REGISTRY` as models are added.
- Not published to PyPI. (`pyproject.toml` URLs fixed to `github.com/hulacon`,
  Aug 2026.)

## Roadmap

Ordered; items become "next up" as their predecessors land.

1. **Phase 2 — first full end-to-end run** (done Aug 2026): all caches populated
   (CLIP, fastText, norms, trained regressors); both models validated end-to-end via
   the CLI with strong face validity on a sample word list; README rewritten.
2. **Interactive HTML dashboard** (done Aug 2026 — `word2psy viz browse scores.csv
   -o viewer.html --open`; see `viz/dashboard.py` above). Validated in-browser
   against a real `--all` run on an 8-sentence stimulus CSV: all 40 model × view
   combinations render or degrade gracefully, and detail panels show strong face
   validity (e.g. "the" maxes Zipf frequency; concrete vs. abstract words separate
   in norm-space MDS; a joyful sentence tops GoEmotions joy). Deliberately not
   ported from viz2psy: the animated trajectory (payload cost, low value for text)
   and the separate popup viewer window (the detail view is an in-page overlay
   instead).
3. **Cross-modal demo** (done Aug 2026 — `word2psy crossmodal`, see `crossmodal.py`
   above; per Ben, deliberately NOT the flagship README example, just a short
   section). Validated end-to-end with the real viz2psy CLI: 6 PIL-drawn icon
   images (heart/tree/car/house/sun/fish) scored with `viz2psy clip`, the 6
   matching words with `word2psy clip_text` — all 6 words ranked their matching
   image first (match sims .29–.34 vs non-match ≤ .25). Note: viz2psy is not
   installable from PyPI (its `deepgaze-pytorch` dep is GitHub-only); its venv
   at ../viz2psy/.venv was built by installing deps minus deepgaze, then
   `uv pip install -e . --no-deps` (+ plotly/kaleido) — saliency won't run
   there but everything else does.
4. **Model expansion** (done Aug 2026 — 10 models; the model space is considered
   feature-complete for dashboard design). Deliberately deferred: topics, NER, moral
   foundations, LIWC-style categories, GloVe. Note for dashboard: all current model
   outputs are numeric — adding categorical outputs (POS tags, captions) later would
   be an interface change.
5. **Phrase/sentence-level features** — partially landed Aug 2026: `--by-sentence`
   gives every chunk-level model sentence resolution (no I/O format change), and
   per-chunk word-feature aggregates (`{feature}_{mean,sd,min,max}`, on by
   default) cover the common "mean concreteness of the passage" use case.
   Sentence-transformer embeddings were already covered by `minilm`.

   **Remaining — phrase-level norms (next session): decide compositional vs.
   direct scoring empirically.** Design:
   - *Test set*: the multi-word entries the norm downloads already contain but
     `train.py` drops at training time — Brysbaert concreteness alone has ~2.9k
     human-rated two-word expressions; check NRC VAD, socialness, and BOI for
     more. These are free held-out ground truth: no new data collection.
   - *Contenders*, scored against held-out human ratings (r², same protocol as
     the word-norm CV benchmarks): (a) **compositional** — mean of constituent
     word predictions; (b) **direct-fastText** — the phrase string straight
     through the existing fastText→ridge regressors (subword averaging makes
     this quasi-compositional in embedding space, so it may not differ much
     from (a)); (c) **direct-MiniLM** — phrase embedded with minilm, new ridge
     regressors trained on the single-word ratings.
   - *Report* overall r² plus the non-compositional tail specifically (idioms
     like "hot dog", negated phrases) where (a) should fail distinctively.
   - *Decision rule*: if (b) or (c) meaningfully beats (a), add a phrase-level
     norms path (likely a `phrase_norms` chunk-level model); if not, close the
     item — aggregates + compositional means already are the answer, and that
     null result is worth recording in the README.
