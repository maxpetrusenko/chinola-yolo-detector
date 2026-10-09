#!/usr/bin/env python3
"""Merge reviewed YOLO datasets into one training dataset."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def copy_split(dataset: Path, out: Path, split: str, prefix: str) -> tuple[int, int]:
    image_dir = dataset / "images" / split
    label_dir = dataset / "labels" / split
    if not image_dir.exists():
        return 0, 0

    copied_images = 0
    copied_boxes = 0
    for image in sorted(image_dir.iterdir()):
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        target_name = f"{prefix}__{image.name}"
        target_image = out / "images" / split / target_name
        target_label = out / "labels" / split / f"{Path(target_name).stem}.txt"
        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, target_image)

        source_label = label_dir / f"{image.stem}.txt"
        if source_label.exists():
            text = source_label.read_text()
            target_label.write_text(text)
            copied_boxes += sum(1 for line in text.splitlines() if line.strip())
        else:
            target_label.write_text("")
        copied_images += 1
    return copied_images, copied_boxes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    args = parser.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)

    totals = []
    for dataset in args.dataset:
        prefix = dataset.name.replace("_", "-")
        for split in ("train", "val"):
            images, boxes = copy_split(dataset, args.out, split, prefix)
            totals.append((dataset, split, images, boxes))

    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: passion_fruit\n"
    )
    for dataset, split, images, boxes in totals:
        print(f"{dataset.name}/{split}: images={images} boxes={boxes}")
    print(f"out={args.out}")


if __name__ == "__main__":
    main()
