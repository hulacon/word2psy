# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
