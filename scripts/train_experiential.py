#!/usr/bin/env python3
"""Build the ``lexical_norms_experiential`` weights from the Binder et al. (2016) ratings.

Developer script: the ratings are user-supplied, so the weights are not package
data (see ``word2psy.norms.experiential``).

    python scripts/train_experiential.py --ratings WordSet1_Ratings.xlsx \\
        [--attributes keep.txt] [--out experiential_v1.npz]

``--attributes`` names a file with one attribute per line (``vision`` or
``lexical_norms_experiential_vision``; ``#`` lines skipped); default all 65.
Nothing here encodes a pass bar: which attributes to keep is the caller's call.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from word2psy.norms import experiential


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ratings", help="WordSet1_Ratings.xlsx (default: $WORD2PSY_BINDER_RATINGS)")
    ap.add_argument("--attributes", type=Path, help="file listing the attributes to keep, one per line")
    ap.add_argument("--out", type=Path, help=f"weights file (default: {experiential.weights_path()})")
    a = ap.parse_args(argv)
    attrs = a.attributes.read_text().splitlines() if a.attributes else None
    experiential.train(a.ratings, attributes=attrs, out=a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
