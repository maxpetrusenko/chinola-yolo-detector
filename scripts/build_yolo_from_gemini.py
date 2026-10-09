#!/usr/bin/env python3
"""Build a one-class YOLO draft dataset from Gemini detection CSV."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image


def yolo_line(box: tuple[float, float, float, float], width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    x1, x2 = max(0.0, x1), min(float(width), x2)
    y1, y2 = max(0.0, y1), min(float(height), y2)
    return (
        "0 "
        f"{((x1 + x2) / 2 / width):.6f} "
        f"{((y1 + y2) / 2 / height):.6f} "
        f"{((x2 - x1) / width):.6f} "
        f"{((y2 - y1) / height):.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/processed/kaggle-gemini-passion-draft"))
    parser.add_argument("--min-confidence", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)

    grouped: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    with open(args.detections, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["class_name"] != "passion_fruit":
                continue
            if float(row["confidence"]) < args.min_confidence:
                continue
            grouped[row["image"]].append((float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])))

    images = sorted(grouped)
    random.Random(args.seed).shuffle(images)
    val_count = max(1, int(len(images) * args.val_ratio)) if images else 0
    split_for = {name: "val" if i < val_count else "train" for i, name in enumerate(images)}

    written = 0
    boxes = 0
    for image_name in images:
        split = split_for[image_name]
        src = args.images / image_name
        if not src.exists():
            continue
        with Image.open(src) as img:
            width, height = img.size
        labels = [yolo_line(box, width, height) for box in grouped[image_name]]
        if not labels:
            continue
        dst_img = args.out / "images" / split / image_name
        dst_label = args.out / "labels" / split / f"{Path(image_name).stem}.txt"
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_img)
        dst_label.write_text("\n".join(labels) + "\n")
        written += 1
        boxes += len(labels)

    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: passion_fruit\n"
    )
    print(f"images_written={written}")
    print(f"boxes_written={boxes}")
    print(f"data_yaml={args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
