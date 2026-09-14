# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `word2psy viz browse` accepts multiple scores paths, or a directory of
  per-model `*_words.csv` / `*_chunks.csv` files, and merges them into one
  all-models dashboard (`word2psy.viz.merge`): words tables merge on
  `(stimulus_id, word_idx)`, chunks tables on `(stimulus_id, chunk_idx)`,
  with the shared scaffold columns kept once.

## [0.9.1] - 2026-09-07

### Fixed

- `word2vec` loads the cached GoogleNews vectors directly once they are on
  disk, instead of going through `gensim.downloader.load()` every time.
  The downloader re-fetches `information.json` from GitHub on every call
  and rewrites it truncate-then-write inside the data dir, so concurrent
  loads against a shared cache race: a reader can see the file empty and
  fail with `Expecting value: line 1 column 1 (char 0)`. Seen in 2 of 23
  concurrent SLURM array tasks on a shared PIRG cache. Also makes offline
  runs work once the vectors exist. The first-ever download is unchanged.

## [0.9.0] - 2026-08-25

### Added

- **`lexical_norms` gains `animacy`** — the general living/nonliving
  scale from VanArsdall & Blunt (2022; 1,200 concrete nouns, 100–700),
  the classic ventral-stream dimension the battery previously lacked on
  every arm. Trained through the existing fastText+ridge machinery like
  every other norm (human ratings anchor the mapping; ridge extends
  coverage to arbitrary words); the source sheet also carries the five
  finer animacy scales (thought, reproduction, person-similarity, goals,
  movement) if a decomposition is ever wanted. The xlsx norm loader
  learned an optional `sheet` key for multi-sheet workbooks.
  lexical_norms is now 24 features (23 norms + Zipf). **Cached
  regressors from earlier versions do not include animacy** — the model
  auto-trains the missing regressor on next load; existing extractions
  need a re-run to pick up the new column.

## [0.8.0] - 2026-08-25

### Added

- **`interaction` model:** seven social-interaction scalar rates per chunk,
  the readability pattern applied to interactional language —
  `interaction_first_person_sing`, `_first_person_plur`, `_second_person`,
  `_third_person` (person deixis; "it" excluded), `_person_noun`,
  `_discourse_marker` (closed lists in the module, so every score is
  explainable by pointing at the tokens), and `_question` (fraction of
  sentences ending in "?"). Rates are per word token on the pipeline's own
  tokenization; a chunk with no word tokens scores NaN. Analytic
  (`checkpoint` null); nltk is the only dependency, already required.
  Like readability, single-word chunks yield degenerate values — the
  metrics are meaningful for sentence-or-longer chunks.

## [0.7.0] - 2026-08-23

### Changed

- **BREAKING (values, not columns): `fasttext` emits the dictionary input
  row for in-vocabulary words**, instead of `get_word_vector`. The
  checkpoint is a subword `.bin` with `minn = maxn = 4`, and
  `get_word_vector` returns the unweighted mean of a word's own row and
  its character 4-grams — so a word under four characters has no n-grams
  and keeps its full norm, while every longer word is diluted by
  `1 / (1 + n_ngrams)` onto n-gram rows that share a large common
  direction. The emitted norm became close to a step function of word
  *length*: measured over the 751 word types in the NSD shared1000
  captions, `"a"` came back at 3.11 and `"skateboard"` at 0.38.

  Any mean-pooling consumer then hands the pooled vector to the shortest,
  most frequent function words — `"a"` alone is 14% of those caption
  tokens. Pooled to caption level the space had a participation ratio of
  **2.2 against 17.1** for the same words' dictionary rows, i.e. 1,000
  captions occupying about two effective dimensions. Type-level
  participation ratio was 54.9 against 82.7.

  Subword composition is kept for the OOV case it exists to serve;
  coverage on ordinary English is near-total (751/751 on those captions).
  Column names, dimensionality and `checkpoint` are unchanged, so
  **existing extractions are silently stale and must be re-run** — there
  is no schema change for a consumer to detect. The chunk-level
  `word2vec` space is unaffected and was never diluted this way.

## [0.6.0] - 2026-08-22

### Changed

- **BREAKING: `--passthrough` is opt-in.** `read_inputs` copied every
  non-text column of a CSV input into the chunks output, so experiment
  variables became indistinguishable from model output, and twelve models
  scoring one input re-emitted twelve identical copies of each — psytwill
  refused `movies/annot/chunks` with 82,848 duplicate `(stimulus, feature)`
  keys over exactly this. `--passthrough` now names the columns to carry,
  or `all` for the previous behaviour. The default carries identity and
  the stimulus's own intrinsic coordinates (`stimulus_id`, `onset`,
  `offset`, `time`, `voice`) and nothing else. Position is already
  conveyed by `chunk_idx` / `word_idx`.

## [0.5.1] - 2026-08-22

### Fixed

- **`stimulus_id` reaches the words table too.** One
  `if "stimulus_id" not in chunks_df.columns` guard wrapped id resolution
  for *both* output frames, so whenever the input already supplied the
  column the guard short-circuited and every `*_words.csv` was written
  without it. Silent, because consumers fall back to `chunk_label` — in
  the MMMData store that made the caption *text* the stimulus id for
  movie word-level rows. 335 tables across 5 models and 8 sources were
  affected. The two frames are now resolved separately.

## [0.5.0] - 2026-08-22

### Added

- **Chunk-level pooling for static word embeddings.** `word2vec` and
  `fasttext` are the only spaces that exist per token and not per chunk,
  so their chunks table carried metadata columns and nothing else — 10
  columns against `clip_text`'s 522. They could not be compared against
  chunk-level spaces at all. `pipeline.pool_word_embeddings()` now
  mean-pools a chunk's word vectors into the chunks table under the
  *same* column names (`word2vec_000`...), which is what makes the
  result a first-class space to a consumer: psytwill detects a space by
  the `{prefix}_{NNN}` pattern and nothing else.

  Pooling is NaN-aware, so word2vec's out-of-vocabulary tokens drop out
  rather than poisoning the vector, and a new `{prefix}_n_pooled` column
  records how many words actually contributed — a vector pooled over 2
  of 9 words is not the same evidence as one pooled over all 9. A chunk
  with no in-vocabulary word gets an all-NaN vector and `n_pooled = 0`,
  not a zero vector.

  Deliberately *not* the `mean/sd/min/max` form used for scalar features:
  per-dimension spread of an embedding is not interpretable the way it is
  for a lexical norm, and four stats would emit 1,200 columns for a 300-d
  space while breaking the prefix pattern consumers match on.

  On by default; `--no-embedding-pooling` opts out. The sidecar records
  it as `models.{name}.chunk_pooling` so a pooled chunk vector is
  distinguishable from one a model emitted directly, since the column
  names are identical either way.

### Fixed

- The internal embedding-column pattern matched exactly three digits, so
  a word-level space wider than 1,000 dimensions would have been treated
  as a scalar feature — aggregated as `_mean`/`_sd`/`_min`/`_max` and
  never pooled. Now matches three or more, consistent with Contract B
  §4.1's fixed-width indices. No shipped model is affected: both
  word-level embeddings are 300-d, and `ebind_text` (1024-d) is
  chunk-level and never enters the words table.

## [0.4.0] - 2026-08-20

**Breaking:** two models' feature columns are renamed. `lexical_norms` and
`wordform` emitted bare column names (`valence`, `concreteness`,
`zipf_frequency`, `length`, `old20`, ...) in violation of Contract B §4.1,
which requires every feature column to start with its model's declared
prefix. psytwill attributes a column to a model by that prefix and nothing
else, so all 27 columns landed in its null-model bucket — 0 of 16 attributed
in the first non-embedding extraction ever run against the new layout
(MMMData, 2026-08-20). Every earlier extraction used only embedding models,
whose columns are prefixed by construction, which is why it went unseen.

### Changed

- `lexical_norms` now emits `lexical_norms_<dimension>` for all 23 columns
  (`lexical_norms_valence`, `lexical_norms_sensorimotor_touch`,
  `lexical_norms_zipf_frequency`, ...).
- `wordform` now emits `wordform_length`, `wordform_n_syllables`,
  `wordform_n_phonemes`, `wordform_old20`.
- `viz` feature configs, the dashboard's scalar lists, and the docstring
  examples follow the new names.

No legacy-alias shim ships with this release: unlike viz2psy 0.6.0's column
rename, no extraction of these two models existed on disk anywhere at the
time of the change, so there is nothing to migrate. Re-extract rather than
rename in place (§7).

### Added

- `tests/test_column_prefixes.py` — asserts, per model, that every emitted
  non-reserved column starts with the model's registry name. Stubs the
  regressors and fastText so it needs none of the ~7 GB of weights, and
  carries a mutation test proving the assertion has teeth. This is the check
  whose absence let the defect ship: the family-wide `stimfeat_preflight.py`
  validates that the *declared* prefix namespace is collision-free and never
  looks at an emitted column.

## [0.3.1] - 2026-08-18

Housekeeping release for public use — documentation, packaging metadata, and
citations; no feature-output changes.

### Added

- `ebind` optional-dependency extra (the GitHub-only `ebind` package was
  previously undeclared, so `ebind_text` — and therefore `--all` — failed
  on a clean install).
- README rows and cross-modal documentation for `clap_text` and
  `ebind_text` (both previously absent from the README); the cross-modal
  section now covers all three shared spaces (CLIP, CLAP, EBind).
- `CITATION.cff` for citing word2psy itself.
- README "Related packages" section (viz2psy, aud2psy, psytwill).
- "Citing" section with full references for every model (the norm
  databases were already cited; the neural models were not).

### Fixed

- `word2psy.models` lazy registry listed only 2 of the 12 model classes;
  all 12 are now importable from `word2psy.models`.
- `viz` feature detection (`FEATURE_CONFIGS`) was missing an `ebind_text`
  entry, so its columns went unrecognized in dashboards (and the contract
  test failed).
- `requires-python` now `>=3.10,<3.13` to match the documented fastText
  constraint (previously pip would install on 3.13 and fail to build).
- Removed unused `setuptools-scm` build requirement.
- CLAUDE.md model/norm counts refreshed; `psyquilt` corrected to
  `psytwill`.

## [0.3.0] - 2026-08-17

### Added

- `ebind_text` model (chunk-level): 1024-d L2-normalised text embeddings
  from EBind's Perception Encoder text arm (checkpoint
  `encord-team/ebind-full`, revision-pinned). Shares one cross-modal
  space with viz2psy `ebind` and aud2psy `ebind_audio`; columns are
  fixed-width 4-digit (`ebind_text_0000..1023`). Spoken-word stimuli
  enter the shared space through this arm — EBind's audio arm hears
  isolated words as generic speech (2026-08-17 mmmdata pilot).

## [0.2.0] - 2026-08-10

Conform the output sidecar to the constellation Contract B extractor-output
convention (mmmdata-agents `docs/constellation-contracts.md` §4.1). Column
names are unchanged in this release.

### Added

- Sidecar (`.meta.json`): `schema_version` ("1.0"), `extractor`,
  `extractor_version`, and per-model `package_version` + **`checkpoint`**
  (exact architecture+weights identifier, e.g. `ViT-B-32/laion2b_s34b_b79k`
  for clip_text — checkpoint identity backs the cross-modal space guarantees
  in psytwill). Legacy `word2psy_version` and per-model `version` keys
  retained for one deprecation cycle.
- `BaseModel.checkpoint` class attribute (None for analytic models such as
  `readability` and `wordform`).
- **`stimulus_id` column** (first column of both output tables, §4.1
  identity rules): the `--id-column` labels per chunk for CSV input, else
  the input file's stem; `--stimulus-id` overrides with a constant.

### Fixed

- `word2psy ... -o dir/scores.csv` (and `word2psy crossmodal -o`) now create
  missing parent directories instead of failing.
