"""`word2psy sidecar refresh`: bring existing sidecars up to schema 1.1.

Contract B 1.1 adds a `nulls` map to every model entry (what a NaN in each
column means). Feature values do not change between 1.0 and 1.1, so an old
family is brought forward by rewriting its `.meta.json` only; a CSV is never
written.

Two things are written per model entry:

- `chunk_aggregates` -- the `{feature}_{stat}` columns the pipeline appended
  to the chunks table. 1.0 sidecars never listed them, and a `nulls` key must
  name an inventoried column, so the inventory is rebuilt from the chunks
  table's actual header first.
- `nulls` -- declared entries for the model's columns plus the derived
  aggregate entries (`metadata.model_nulls`).

Every one of the model's columns is then read back from the family's tables,
and a NaN in an undeclared column refuses the refresh for that sidecar
(nothing is written).
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from word2psy.exceptions import Word2PsyError
from word2psy.metadata import AGGREGATE_STATS, SCHEMA_VERSION, model_nulls


@dataclass
class RefreshResult:
    path: Path
    status: str  # "refreshed" | "unchanged" | "refused" | "skipped"
    undeclared: dict[str, list[str]] = field(default_factory=dict)  # model -> NaN columns
    note: str = ""


def find_sidecars(paths: list[str | Path]) -> list[Path]:
    """Sidecar files named directly, plus every `*.meta.json` under a directory."""
    found: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            found.extend(sorted(p.rglob("*.meta.json")))
        elif p.name.endswith(".meta.json") and p.is_file():
            found.append(p)
        else:
            raise Word2PsyError(f"{p} is neither a directory nor an existing .meta.json sidecar")
    return found


def _table_path(sidecar: Path, recorded: str) -> Path:
    """The recorded table, or the same filename beside the sidecar if the tree moved."""
    p = Path(recorded)
    if p.is_file():
        return p
    beside = sidecar.parent / p.name
    if beside.is_file():
        return beside
    raise Word2PsyError(
        f"{sidecar}: output table {recorded} is missing (also not beside the sidecar). "
        "The refresh checks nulls against the data, so it cannot proceed without it."
    )


def _inventory(name: str, features: dict) -> list[str]:
    """Column names from a `features` block: the list, or the expanded pattern."""
    if "columns" in features:
        return list(features["columns"])
    if "pattern" in features:
        lo, hi = features["range"]
        width = max(3, len(str(hi)))
        stem = features["pattern"].split("{")[0]
        return [f"{stem}{i:0{width}d}" for i in range(lo, hi + 1)]
    return []


def refresh_sidecar(path: str | Path, *, dry_run: bool = False) -> RefreshResult:
    import pandas as pd

    path = Path(path)
    meta = json.loads(path.read_text())
    if meta.get("extractor") != "word2psy":
        return RefreshResult(path, "skipped", note=f"extractor {meta.get('extractor')!r}")
    tables = {kind: pd.read_csv(_table_path(path, t["path"]))
              for kind, t in (meta.get("output") or {}).items()}
    chunk_header = list(tables["chunks"].columns) if "chunks" in tables else []

    new = copy.deepcopy(meta)
    undeclared: dict[str, list[str]] = {}
    for name, entry in new["models"].items():
        cols = _inventory(name, entry.get("features") or {})
        pooling = entry.get("chunk_pooling")
        pooled = _inventory(name, pooling["features"]) if pooling else []
        aggregates = [f"{f}_{s}" for f in cols for s in AGGREGATE_STATS
                      if f"{f}_{s}" in chunk_header]
        if aggregates:
            entry["chunk_aggregates"] = {"stats": list(AGGREGATE_STATS), "nan_policy": "omit",
                                         "sd_ddof": 1, "columns": aggregates}
        entry["nulls"] = model_nulls(name, cols, aggregates, pooled)
        mine = set(cols) | set(aggregates) | set(pooled)
        nan_cols: set[str] = set()
        for df in tables.values():
            # numeric columns only: a string column's empty cell reads back as NaN
            num = df[[c for c in df.columns if c in mine]].select_dtypes(include="number")
            nan_cols |= {c for c in num.columns if num[c].isna().any()}
        bad = sorted(nan_cols - set(entry["nulls"]))
        if bad:
            undeclared[name] = bad
    if undeclared:
        return RefreshResult(path, "refused", undeclared,
                             note="NaN in undeclared column(s): a producer defect; nothing written")
    new["schema_version"] = SCHEMA_VERSION
    if new == meta:
        return RefreshResult(path, "unchanged")
    from word2psy import __version__

    new.setdefault("refreshed", []).append({
        "by": f"word2psy {__version__}",
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fields": ["schema_version", "models.*.nulls", "models.*.chunk_aggregates"],
        "from_schema_version": meta.get("schema_version"),
    })
    if not dry_run:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(new, indent=2))
        os.replace(tmp, path)
    return RefreshResult(path, "refreshed")


def main(argv: list[str]) -> int:
    """`word2psy sidecar refresh PATH... [--dry-run]`."""
    import argparse
    import sys
    from collections import Counter

    parser = argparse.ArgumentParser(prog="word2psy sidecar",
                                     description="Maintain existing .meta.json sidecars.")
    sub = parser.add_subparsers(dest="sidecar_cmd")
    p_rf = sub.add_parser(
        "refresh",
        help="Bring sidecars to the current Contract B schema (1.1: per-model `nulls`, "
             "plus the chunk-aggregate inventory). Rewrites JSON only, never a CSV; refuses "
             "a sidecar whose tables hold NaN in an undeclared column.",
    )
    p_rf.add_argument("paths", nargs="+",
                      help=".meta.json files, or directories searched recursively "
                           "(sidecars from other extractors are skipped)")
    p_rf.add_argument("--dry-run", action="store_true", help="Check and report; write nothing.")
    args = parser.parse_args(argv)
    if args.sidecar_cmd is None:
        parser.print_help()
        return 1
    counts: Counter = Counter()
    for sidecar in find_sidecars(args.paths):
        r = refresh_sidecar(sidecar, dry_run=args.dry_run)
        counts[r.status] += 1
        if r.status == "refused":
            cols = "; ".join(f"{m}: {', '.join(c[:8])}{' ...' if len(c) > 8 else ''}"
                             for m, c in r.undeclared.items())
            print(f"REFUSED {sidecar}: {cols}", file=sys.stderr)
    verb = "would refresh" if args.dry_run else "refreshed"
    print(f"word2psy sidecar refresh: {counts['refreshed']} {verb}, {counts['unchanged']} unchanged, "
          f"{counts['refused']} refused, {counts['skipped']} skipped (other extractors)")
    return 1 if counts["refused"] else 0
