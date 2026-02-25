# word2psy

A toolbox for bulk extraction and exploration of psychological and linguistic features from text. Features are extracted using a command line interface which wraps multiple computational models used in natural language processing and human psychology. Features are stored in tabular format (csv) and a basic html viewer for interacting with the data is provided.

## Features

- **Unified CLI** for processing text files, CSV columns, and corpora
- **Pre-integrated models** covering sentiment, emotion, semantics, readability, and more
- **Interactive visualizations** with Plotly-based dashboards
- **Metadata sidecar files** documenting outputs and feature definitions

## Installation

```bash
pip install word2psy
```

Or from source:

```bash
git clone https://github.com/hulacon/word2psy.git
cd word2psy
pip install -e .
```

## Quick Start

```bash
# Score texts with multiple models
word2psy sentiment emotion texts/*.txt -o scores.csv

# Score a column in a CSV
word2psy sentiment emotion corpus.csv --text-column response -o scores.csv

# Visualize results
word2psy-viz scores.csv --browse -o viewer.html
```

```python
from word2psy.models.sentiment import SentimentModel
from word2psy.pipeline import score_texts

model = SentimentModel()
df = score_texts(model, ["I love this product.", "Terrible experience."])
```

## Available Models

| Model | Output | Description |
|-------|--------|-------------|
| `sentiment` | TBD | Valence / sentiment scores |
| `emotion` | TBD | Discrete emotion probabilities |
| `embeddings` | TBD | Contextual text embeddings |
| `readability` | TBD | Readability and complexity metrics |
| `topics` | TBD | Topic distributions |
| `liwc` | TBD | Linguistic category proportions |
| `ner` | TBD | Named entity counts |
| `morality` | TBD | Moral foundations scores |

## Documentation

| Document | Description |
|----------|-------------|
| [CLI](docs/cli.md) | `word2psy` command line reference |
| [Models](docs/models.md) | Available models, outputs, and references |
| [Visualization](docs/visualization.md) | `word2psy-viz` CLI and interactive features |
| [API](docs/api.md) | Python API reference |
| [Changelog](CHANGELOG.md) | Version history and release notes |

## Hardware Requirements

- **GPU**: Recommended for transformer-based models; CPU fallback supported
- **RAM**: 16GB minimum, 32GB recommended for parallel model execution
- **Disk**: Model weights downloaded automatically on first use

## Citation

TBD

## License

MIT License. See [LICENSE](LICENSE) for details.
