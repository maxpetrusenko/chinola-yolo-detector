#!/usr/bin/env python3
"""Create a passion-fruit-only YOLO dataset from the draft labels."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


MANGO_MARKERS = (
    "mango",
    "mangos",
    "aam",
    "kachche",
    "golden-mango",
    "mango-mania",
)


def is_mango_name(name: str) -> bool:
    lower = name.lower()
    return any(marker in lower for marker in MANGO_MARKERS)


def convert_label(source: Path, target: Path) -> int:
    kept = []
    for line in source.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id = parts[0]
        if class_id != "1":
            continue
        kept.append("0 " + " ".join(parts[1:]))
    if kept:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(kept) + "\n")
    return len(kept)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/processed/chinola-yolo-draft"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/chinola-yolo-passion-only"))
    args = parser.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)

    images_written = 0
    boxes_written = 0
    skipped_mango_name = 0
    skipped_no_passion = 0

    for split in ("train", "val"):
        for image_path in sorted((args.source / "images" / split).glob("*")):
            if is_mango_name(image_path.name):
                skipped_mango_name += 1
                continue
            label_path = args.source / "labels" / split / f"{image_path.stem}.txt"
            if not label_path.exists():
                skipped_no_passion += 1
                continue
            target_label = args.out / "labels" / split / label_path.name
            kept = convert_label(label_path, target_label)
            if kept == 0:
                skipped_no_passion += 1
                continue
            target_image = args.out / "images" / split / image_path.name
            target_image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, target_image)
            images_written += 1
            boxes_written += kept

    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: passion_fruit\n"
    )
    print(f"images_written={images_written}")
    print(f"boxes_written={boxes_written}")
    print(f"skipped_mango_name={skipped_mango_name}")
    print(f"skipped_no_passion={skipped_no_passion}")
    print(f"data_yaml={args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
