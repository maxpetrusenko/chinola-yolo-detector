#!/usr/bin/env python3
"""Create a balanced sample from the Kaggle passion fruit classification images."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/raw/kaggle/passion-fruit-disease-detection/indabax-cameroon-2021-1/Train.csv"),
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("data/raw/kaggle/passion-fruit-disease-detection/indabax-cameroon-2021-1/Images"),
    )
    parser.add_argument("--out", type=Path, default=Path("data/processed/kaggle-sample-images"))
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    manifest = []
    for class_name, group in df.groupby("class"):
        sample = group.sample(n=min(args.per_class, len(group)), random_state=args.seed)
        for row in sample.to_dict("records"):
            image_id = row["Image_ids"]
            class_label = row["class"]
            src = args.images / f"{image_id}.jpg"
            if not src.exists():
                continue
            dst = args.out / f"{image_id}_{class_label}.jpg"
            shutil.copy2(src, dst)
            manifest.append((dst.name, image_id, class_label))

    with open(args.out / "manifest.tsv", "w") as f:
        f.write("file\timage_id\tclass\n")
        for row in manifest:
            f.write("\t".join(row) + "\n")
    print(f"sampled={len(manifest)}")
    print(f"out={args.out}")


if __name__ == "__main__":
    main()
