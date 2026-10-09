#!/usr/bin/env python3
"""Build a draft YOLO dataset from zero-shot detection CSV boxes."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

import yaml
from PIL import Image


DEFAULT_CLASS_ALIASES = {
    "mango": "mango",
    "green mango": "mango",
    "ripe mango": "mango",
    "passion fruit": "passion_fruit",
    "green passion fruit": "passion_fruit",
    "ripe passion fruit": "passion_fruit",
}


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def nms(boxes: list[dict], threshold: float) -> list[dict]:
    kept: list[dict] = []
    for box in sorted(boxes, key=lambda item: item["confidence"], reverse=True):
        if all(iou(box["xyxy"], kept_box["xyxy"]) < threshold for kept_box in kept):
            kept.append(box)
    return kept


def yolo_line(class_id: int, box: tuple[float, float, float, float], width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    x1, x2 = max(0.0, x1), min(float(width), x2)
    y1, y2 = max(0.0, y1), min(float(height), y2)
    x_center = ((x1 + x2) / 2) / width
    y_center = ((y1 + y2) / 2) / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"


def normalized_class_name(class_name: str, class_mode: str) -> str:
    value = class_name.strip().lower().replace("_", " ")
    if class_mode == "fruit":
        return "fruit"
    if class_mode == "source":
        return value.replace(" ", "_")
    return DEFAULT_CLASS_ALIASES.get(value, value.replace(" ", "_"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--min-conf", type=float, default=0.05)
    parser.add_argument("--class-mode", choices=["fruit", "family", "source"], default="family")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    grouped: dict[str, list[dict]] = defaultdict(list)
    class_to_id: dict[str, int] = {}
    with open(args.detections, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            confidence = float(row["confidence"])
            if confidence < args.min_conf:
                continue
            class_name = normalized_class_name(row["class_name"], args.class_mode)
            class_id = class_to_id.setdefault(class_name, len(class_to_id))
            grouped[row["image"]].append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "xyxy": (
                        float(row["x1"]),
                        float(row["y1"]),
                        float(row["x2"]),
                        float(row["y2"]),
                    ),
                }
            )

    images = sorted(grouped)
    random.Random(args.seed).shuffle(images)
    val_count = max(1, int(len(images) * args.val_ratio)) if images else 0
    splits = {
        "val": set(images[:val_count]),
        "train": set(images[val_count:]),
    }

    for split in splits:
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)

    written = 0
    for split, split_images in splits.items():
        for image_name in split_images:
            source = args.images / image_name
            if not source.exists():
                continue
            with Image.open(source) as img:
                width, height = img.size
            boxes = nms(grouped[image_name], threshold=0.6)
            label_lines = [yolo_line(box["class_id"], box["xyxy"], width, height) for box in boxes]
            if not label_lines:
                continue
            shutil.copy2(source, args.out / "images" / split / image_name)
            label_path = args.out / "labels" / split / f"{Path(image_name).stem}.txt"
            label_path.write_text("\n".join(label_lines) + "\n")
            written += 1

    data = {
        "path": str(args.out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {class_id: name for name, class_id in sorted(class_to_id.items(), key=lambda item: item[1])},
    }
    with open(args.out / "data.yaml", "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print(f"images_with_labels={written}")
    print(f"classes={data['names']}")
    print(f"data_yaml={args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
