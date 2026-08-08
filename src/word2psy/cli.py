#!/usr/bin/env python3
"""CLI for extracting psychological and linguistic features from text.

Examples
--------
    # List available models
    word2psy --list-models

    # Score text with lexical norms
    # (writes features_words.csv, features_chunks.csv, features.meta.json)
    word2psy lexical_norms input.txt -o features.csv

    # Score with CLIP text embeddings
    word2psy clip_text input.txt -o features.csv

    # Multiple models
    word2psy clip_text lexical_norms input.txt -o features.csv

    # Score a CSV stimulus list: each row is one chunk; other columns
    # (IDs, conditions) pass through to the chunks output
    word2psy clip_text lexical_norms stimuli.csv --text-column word \
        --id-column stim_id -o features.csv

    # All models
    word2psy --all input.txt -o features.csv

    # Read from stdin
    echo "The quick brown fox" | word2psy lexical_norms

    # Download norm databases (one-time setup)
    word2psy --download-norms

    # Visualize results
    word2psy viz timeseries features.csv -o timeseries.png
    word2psy viz heatmap features.csv -o heatmap.png
    word2psy viz scatter features.csv --features "sensorimotor_*" -o scatter.png
    word2psy viz recommend features.csv
"""

import argparse
import importlib
import sys
import time
from pathlib import Path

from word2psy.exceptions import (
    DeviceError,
    InferenceError,
    ModelLoadError,
    TextLoadError,
    Word2PsyError,
)

MODEL_REGISTRY = {
    # word-level
    "lexical_norms": (
        "word2psy.models.lexical_norms",
        "LexicalNormsModel",
        "22 psycholinguistic word norms + Zipf frequency",
    ),
    "wordform": (
        "word2psy.models.wordform",
        "WordformModel",
        "Length, syllables, phonemes, orthographic OLD20",
    ),
    "fasttext": (
        "word2psy.models.fasttext_embed",
        "FastTextModel",
        "300-dim fastText subword embeddings (no OOV)",
    ),
    "word2vec": (
        "word2psy.models.word2vec",
        "Word2VecModel",
        "300-dim GoogleNews word2vec embeddings (NaN for OOV)",
    ),
    # context-level
    "gpt2_surprisal": (
        "word2psy.models.gpt2_surprisal",
        "GPT2SurprisalModel",
        "Word surprisal in context (bits) from GPT-2",
    ),
    # chunk-level
    "sentiment": (
        "word2psy.models.sentiment",
        "SentimentModel",
        "Negative/neutral/positive probabilities (RoBERTa)",
    ),
    "emotion": (
        "word2psy.models.emotion",
        "EmotionModel",
        "28 GoEmotions probabilities (RoBERTa)",
    ),
    "readability": (
        "word2psy.models.readability",
        "ReadabilityModel",
        "7 classic readability metrics",
    ),
    "minilm": (
        "word2psy.models.sentence_embed",
        "SentenceEmbedModel",
        "384-dim MiniLM sentence embeddings",
    ),
    "clip_text": (
        "word2psy.models.clip_text",
        "CLIPTextModel",
        "512-dim CLIP ViT-B-32 text embeddings (shared space with viz2psy)",
    ),
}


def _load_model_class(name: str):
    """Dynamically import and return a model class."""
    module_path, class_name, _ = MODEL_REGISTRY[name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def list_models():
    """Print available models and their descriptions."""
    print("Available models:\n")
    for name, (_, _, desc) in MODEL_REGISTRY.items():
        print(f"  {name:16s}  {desc}")
    print()


def _parse_models_and_inputs(args: list[str]) -> tuple[list[str], list[Path]]:
    """Separate model names from input paths in positional arguments."""
    models = []
    inputs = []
    for arg in args:
        if arg in MODEL_REGISTRY and not inputs:
            models.append(arg)
        else:
            inputs.append(Path(arg))
    return models, inputs


def _read_text(input_path: Path) -> str:
    """Read text from a file."""
    try:
        return input_path.read_text(encoding="utf-8")
    except Exception as e:
        raise TextLoadError(input_path, str(e)) from e


TABULAR_SUFFIXES = {".csv": ",", ".tsv": "\t"}


def read_inputs(
    inputs: list[Path],
    text_column: str | None = None,
    id_column: str | None = None,
):
    """Read input files into (text, chunk_labels, passthrough).

    Plain text files: each file is one chunk; returns (str | list[str],
    None, None). CSV/TSV: a single tabular file where each row is one
    chunk; ``text_column`` selects the text, ``id_column`` (optional)
    provides chunk labels, and all other columns are returned as a
    passthrough DataFrame carried into the chunks output.
    """
    tabular = [p for p in inputs if p.suffix.lower() in TABULAR_SUFFIXES]

    if not tabular:
        parts = [_read_text(p) for p in inputs]
        return (parts if len(parts) > 1 else parts[0]), None, None

    if len(inputs) > 1:
        raise TextLoadError(
            tabular[0], "CSV/TSV input must be a single file (got multiple inputs)"
        )
    path = tabular[0]
    if not text_column:
        raise TextLoadError(path, "--text-column is required for CSV/TSV input")

    import pandas as pd

    try:
        df = pd.read_csv(path, sep=TABULAR_SUFFIXES[path.suffix.lower()])
    except Exception as e:
        raise TextLoadError(path, str(e)) from e

    for col_arg, col in (("--text-column", text_column), ("--id-column", id_column)):
        if col is not None and col not in df.columns:
            raise TextLoadError(
                path,
                f"{col_arg} {col!r} not found; available columns: {list(df.columns)}",
            )

    text = df[text_column].astype(str).tolist()
    labels = df[id_column].astype(str).tolist() if id_column else None
    passthrough = df.drop(columns=[text_column])
    return text, labels, passthrough


def _parse_figsize(value: str) -> tuple[int, int]:
    """Parse a figsize string like '12x8' into a tuple."""
    parts = value.split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Invalid figsize: {value!r}. Use WxH format (e.g., 12x8).")
    return (int(parts[0]), int(parts[1]))


def _viz_main(argv: list[str]):
    """Handle 'word2psy viz ...' subcommands."""
    parser = argparse.ArgumentParser(
        prog="word2psy viz",
        description="Visualize word2psy output.",
    )
    sub = parser.add_subparsers(dest="viz_cmd")

    # --- timeseries ---
    p_ts = sub.add_parser("timeseries", help="Plot features over word/chunk sequence.")
    p_ts.add_argument("csv", type=Path, help="Input CSV from word2psy.")
    p_ts.add_argument("-o", "--output", type=Path, help="Save figure to file.")
    p_ts.add_argument("--features", nargs="+", help="Feature glob patterns.")
    p_ts.add_argument("--index-col", help="Column to use as x-axis.")
    p_ts.add_argument("--figsize", type=_parse_figsize, help="Figure size WxH (e.g., 16x10).")
    p_ts.add_argument("--title", help="Plot title.")
    p_ts.add_argument("--show-diff", action="store_true", help="Overlay first-order differences.")
    p_ts.add_argument("--rolling-window", type=int, help="Overlay rolling average.")

    # --- heatmap ---
    p_hm = sub.add_parser("heatmap", help="Correlation heatmap of features.")
    p_hm.add_argument("csv", type=Path, help="Input CSV from word2psy.")
    p_hm.add_argument("-o", "--output", type=Path, help="Save figure to file.")
    p_hm.add_argument("--features", nargs="+", help="Feature glob patterns.")
    p_hm.add_argument("--method", choices=["pearson", "spearman"], default="pearson")
    p_hm.add_argument("--figsize", type=_parse_figsize, help="Figure size WxH.")
    p_hm.add_argument("--title", help="Plot title.")

    # --- scatter ---
    p_sc = sub.add_parser("scatter", help="2D projection scatter plot.")
    p_sc.add_argument("csv", type=Path, help="Input CSV from word2psy.")
    p_sc.add_argument("-o", "--output", type=Path, help="Save figure to file.")
    p_sc.add_argument("--features", nargs="+", help="Feature glob patterns.")
    p_sc.add_argument("--method", choices=["pca", "ppca", "umap", "tsne", "mds", "mds_nonmetric"], default="pca")
    p_sc.add_argument("--color-by", help="Column to color points by.")
    p_sc.add_argument("--figsize", type=_parse_figsize, help="Figure size WxH.")
    p_sc.add_argument("--title", help="Plot title.")

    # --- recommend ---
    p_rec = sub.add_parser("recommend", help="Suggest visualizations for a CSV.")
    p_rec.add_argument("csv", type=Path, help="Input CSV from word2psy.")

    args = parser.parse_args(argv)

    if args.viz_cmd is None:
        parser.print_help()
        sys.exit(1)

    import pandas as pd

    df = pd.read_csv(args.csv)

    if args.viz_cmd == "timeseries":
        from word2psy.viz.timeseries import plot_timeseries

        fig = plot_timeseries(
            df,
            features=args.features,
            index_col=args.index_col,
            figsize=args.figsize,
            title=args.title,
            show_diff=args.show_diff,
            rolling_window=args.rolling_window,
        )
        if args.output:
            fig.savefig(args.output, dpi=150, bbox_inches="tight")
            print(f"Saved to {args.output}")
        else:
            import matplotlib.pyplot as plt
            plt.show()

    elif args.viz_cmd == "heatmap":
        from word2psy.viz.heatmap import plot_heatmap

        fig = plot_heatmap(
            df,
            features=args.features,
            method=args.method,
            figsize=args.figsize,
            title=args.title,
        )
        if args.output:
            fig.savefig(args.output, dpi=150, bbox_inches="tight")
            print(f"Saved to {args.output}")
        else:
            import matplotlib.pyplot as plt
            plt.show()

    elif args.viz_cmd == "scatter":
        from word2psy.viz.scatter import plot_scatter

        fig = plot_scatter(
            df,
            features=args.features,
            method=args.method,
            color_by=args.color_by,
            figsize=args.figsize if args.figsize else (8, 6),
            title=args.title,
        )
        if args.output:
            fig.savefig(args.output, dpi=150, bbox_inches="tight")
            print(f"Saved to {args.output}")
        else:
            import matplotlib.pyplot as plt
            plt.show()

    elif args.viz_cmd == "recommend":
        from word2psy.viz.feature_config import get_visualization_recommendations

        recs = get_visualization_recommendations(df.columns.tolist())

        print(f"Detected models: {', '.join(recs['detected_models']) or 'none'}\n")
        for viz_type, info in recs.items():
            if viz_type == "detected_models":
                continue
            status = "YES" if info["available"] else "no"
            print(f"  {viz_type:12s}  [{status}]  {info['description']}")
            if info["available"]:
                if "features" in info:
                    print(f"                  Features: {', '.join(info['features'][:8])}")
                    if len(info.get("features", [])) > 8:
                        print(f"                  ... and {len(info['features']) - 8} more")
                if "groups" in info:
                    for group_name, cols in info["groups"].items():
                        print(f"                  {group_name}: {len(cols)} features")
        print()


def main():
    # Route 'viz' subcommand
    if len(sys.argv) > 1 and sys.argv[1] == "viz":
        _viz_main(sys.argv[2:])
        return

    parser = argparse.ArgumentParser(
        description="Extract psychological and linguistic features from text.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  word2psy lexical_norms input.txt -o features.csv\n"
            "  word2psy clip_text lexical_norms input.txt -o features.csv\n"
            "  word2psy clip_text stimuli.csv --text-column word -o features.csv\n"
            "  word2psy --all input.txt -o features.csv\n"
            "  echo 'hello world' | word2psy lexical_norms\n"
            "  word2psy --list-models\n"
            "  word2psy --download-norms\n"
            "  word2psy viz timeseries features.csv -o plot.png\n"
            "  word2psy viz recommend features.csv"
        ),
    )
    parser.add_argument(
        "args",
        nargs="*",
        help="Model name(s) followed by input text file(s).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Save results to this CSV (prints to stdout if omitted).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for model inference (default: 64).",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        default=None,
        help="Device for inference (default: auto-detect).",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all available models.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit.",
    )
    parser.add_argument(
        "--download-norms",
        action="store_true",
        help="Download norm databases and exit (one-time setup).",
    )
    parser.add_argument(
        "--keep-punctuation",
        action="store_true",
        help="Keep punctuation tokens in the word table.",
    )
    parser.add_argument(
        "--text-column",
        help="For CSV/TSV input: column containing the text (each row = one chunk).",
    )
    parser.add_argument(
        "--id-column",
        help="For CSV/TSV input: column to use as chunk labels.",
    )

    args = parser.parse_args()

    if args.list_models:
        list_models()
        sys.exit(0)

    if args.download_norms:
        from word2psy.norms.download import download_all

        download_all()
        print("All norm databases downloaded.")
        sys.exit(0)

    # Parse positional arguments
    models, inputs = _parse_models_and_inputs(args.args)

    if args.all:
        if models:
            print(
                "Error: Cannot specify both --all and model names.",
                file=sys.stderr,
            )
            sys.exit(1)
        models = list(MODEL_REGISTRY.keys())

    if not models:
        parser.print_help()
        sys.exit(1)

    # Validate model names
    invalid = [m for m in models if m not in MODEL_REGISTRY]
    if invalid:
        print(f"Error: Unknown model(s): {', '.join(invalid)}", file=sys.stderr)
        print(
            f"Available: {', '.join(MODEL_REGISTRY.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Read input text
    chunk_labels = None
    passthrough = None
    if inputs:
        text, chunk_labels, passthrough = read_inputs(
            inputs, text_column=args.text_column, id_column=args.id_column
        )
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
        if not text.strip():
            print("Error: No text provided on stdin.", file=sys.stderr)
            sys.exit(1)
    else:
        print("Error: No input files provided.", file=sys.stderr)
        sys.exit(1)

    # Instantiate models
    from word2psy.metadata import MetadataBuilder
    from word2psy.pipeline import score_text

    model_instances = []
    for model_name in models:
        model_cls = _load_model_class(model_name)
        model = model_cls(device=args.device) if args.device else model_cls()
        model_instances.append(model)

    try:
        start = time.time()
        words_df, chunks_df = score_text(
            text,
            model_instances,
            chunk_labels=chunk_labels,
            passthrough=passthrough,
            batch_size=args.batch_size,
            quiet=args.quiet,
            keep_punctuation=args.keep_punctuation,
        )
        total_time = time.time() - start

        if args.output:
            stem = args.output.with_suffix("")
            words_path = stem.parent / f"{stem.name}_words.csv"
            chunks_path = stem.parent / f"{stem.name}_chunks.csv"

            words_df.to_csv(words_path, index=False, float_format="%.6g")
            chunks_df.to_csv(chunks_path, index=False, float_format="%.6g")

            # Save metadata sidecar
            metadata = MetadataBuilder()
            metadata.set_input_text(
                path=inputs[0] if inputs else None,
                n_words=len(words_df),
                n_chunks=len(chunks_df),
                n_sentences=(
                    words_df["sentence_idx"].nunique() if len(words_df) else 0
                ),
            )
            metadata.set_output(
                "words", words_path, len(words_df), len(words_df.columns)
            )
            metadata.set_output(
                "chunks", chunks_path, len(chunks_df), len(chunks_df.columns)
            )
            if model_instances:
                metadata.set_device(str(model_instances[0].device))

            for m in model_instances:
                metadata.add_model(
                    m.name,
                    getattr(m, "feature_names_", []),
                    total_time / len(models),
                    level=m.level,
                )

            meta_path = metadata.save(args.output)

            if not args.quiet:
                print(f"Saved {len(words_df)} word rows to {words_path}")
                print(f"Saved {len(chunks_df)} chunk rows to {chunks_path}")
                print(f"Metadata saved to {meta_path}")
        else:
            print(words_df.to_string(index=False))
            chunk_feature_cols = [
                c
                for c in chunks_df.columns
                if c not in ("chunk_idx", "chunk_label", "n_words")
                and (passthrough is None or c not in passthrough.columns)
            ]
            if chunk_feature_cols:
                print(
                    f"\n[{len(chunks_df)} chunks x {len(chunk_feature_cols)} "
                    "chunk-level features computed - use -o to save the chunks "
                    "table]",
                    file=sys.stderr,
                )

    except DeviceError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except TextLoadError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ModelLoadError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except InferenceError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
