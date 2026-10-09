#!/usr/bin/env python3
"""Print a compact file/type summary for a dataset directory."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    files = [p for p in args.path.rglob("*") if p.is_file()]
    by_suffix = Counter(p.suffix.lower() or "<none>" for p in files)
    print(f"path={args.path}")
    print(f"files={len(files)}")
    for suffix, count in by_suffix.most_common():
        print(f"{suffix}\t{count}")

    print("\nfirst_files:")
    for path in files[:40]:
        print(path)


if __name__ == "__main__":
    main()
