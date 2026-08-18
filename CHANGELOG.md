# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
