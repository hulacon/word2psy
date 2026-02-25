#!/usr/bin/env python3
"""CLI for extracting psychological and linguistic features from text.

Examples
--------
    # List available models
    word2psy --list-models

    # Score text with lexical norms
    word2psy lexical_norms input.txt -o features.csv

    # Score with CLIP text embeddings
    word2psy clip_text input.txt -o features.csv

    # Multiple models
    word2psy clip_text lexical_norms input.txt -o features.csv

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
    "clip_text": (
        "word2psy.models.clip_text",
        "CLIPTextModel",
        "512-dim CLIP ViT-B-32 text embeddings",
    ),
    "lexical_norms": (
        "word2psy.models.lexical_norms",
        "LexicalNormsModel",
        "17 psycholinguistic word norms",
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
    if inputs:
        text_parts = []
        for p in inputs:
            text_parts.append(_read_text(p))
        text = text_parts if len(text_parts) > 1 else text_parts[0]
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
    from word2psy.pipeline import save_embeddings, score_text

    model_instances = []
    for model_name in models:
        model_cls = _load_model_class(model_name)
        model = model_cls(device=args.device) if args.device else model_cls()
        model_instances.append(model)

    try:
        start = time.time()
        df, embeddings = score_text(
            text,
            model_instances,
            batch_size=args.batch_size,
            quiet=args.quiet,
            keep_punctuation=args.keep_punctuation,
        )
        total_time = time.time() - start

        if args.output:
            df.to_csv(args.output, index=False)

            # Save embeddings if any
            if embeddings:
                h5_path = save_embeddings(embeddings, args.output, df)
                if not args.quiet:
                    print(f"Embeddings saved to {h5_path}")

            # Save metadata sidecar
            metadata = MetadataBuilder()
            chunks = [text] if isinstance(text, str) else list(text)
            metadata.set_input_text(
                path=inputs[0] if inputs else None,
                n_words=len(df),
                n_chunks=df["chunk_idx"].nunique(),
                n_sentences=df["sentence_idx"].nunique(),
            )
            metadata.set_output(args.output, len(df), len(df.columns))
            if model_instances:
                metadata.set_device(str(model_instances[0].device))

            for m in model_instances:
                feat_names = [
                    c
                    for c in df.columns
                    if c
                    not in (
                        "word_idx",
                        "word",
                        "sentence_idx",
                        "chunk_idx",
                        "chunk_label",
                        "onset",
                        "offset",
                    )
                ]
                if m.level == "chunk" and m.name in embeddings:
                    dim = embeddings[m.name].shape[1]
                    feat_names = [f"{m.name}_{i:03d}" for i in range(dim)]
                elif m.level == "word":
                    from word2psy.norms.train import NORM_DIMENSIONS

                    feat_names = list(NORM_DIMENSIONS.keys()) + ["zipf_frequency"]
                metadata.add_model(m.name, feat_names, total_time / len(models))

            meta_path = metadata.save(args.output)

            if not args.quiet:
                print(f"Saved {len(df)} rows to {args.output}")
                print(f"Metadata saved to {meta_path}")
        else:
            print(df.to_string(index=False))

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
