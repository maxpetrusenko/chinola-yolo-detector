#!/usr/bin/env python3
"""Convert Pascal VOC XML labels to a one-class YOLO dataset."""

from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image


PASSION_NAMES = {"pf", "rpf", "passion_fruit", "passion fruit", "passionfruit"}


def parse_xml(path: Path, names: set[str]) -> list[tuple[float, float, float, float]]:
    root = ET.parse(path).getroot()
    boxes = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        if names and name not in names:
            continue
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        xmin = float(bnd.findtext("xmin") or 0)
        ymin = float(bnd.findtext("ymin") or 0)
        xmax = float(bnd.findtext("xmax") or 0)
        ymax = float(bnd.findtext("ymax") or 0)
        if xmax > xmin and ymax > ymin:
            boxes.append((xmin, ymin, xmax, ymax))
    return boxes


def yolo_line(box: tuple[float, float, float, float], width: int, height: int) -> str:
    xmin, ymin, xmax, ymax = box
    xmin = max(0.0, min(float(width), xmin))
    xmax = max(0.0, min(float(width), xmax))
    ymin = max(0.0, min(float(height), ymin))
    ymax = max(0.0, min(float(height), ymax))
    return (
        "0 "
        f"{((xmin + xmax) / 2 / width):.6f} "
        f"{((ymin + ymax) / 2 / height):.6f} "
        f"{((xmax - xmin) / width):.6f} "
        f"{((ymax - ymin) / height):.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--names", nargs="*", default=sorted(PASSION_NAMES))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)

    names = {name.lower() for name in args.names}
    xmls = sorted(args.annotations.rglob("*.xml"))
    random.Random(args.seed).shuffle(xmls)
    val_count = int(len(xmls) * args.val_ratio)

    written = 0
    boxes_written = 0
    for index, xml_path in enumerate(xmls):
        image_path = next((args.images / xml_path.relative_to(args.annotations)).with_suffix(ext) for ext in [".jpg", ".jpeg", ".png"] if (args.images / xml_path.relative_to(args.annotations)).with_suffix(ext).exists())
        boxes = parse_xml(xml_path, names)
        if not boxes:
            continue
        split = "val" if index < val_count else "train"
        with Image.open(image_path) as image:
            width, height = image.size
        labels = [yolo_line(box, width, height) for box in boxes]
        dst_image = args.out / "images" / split / image_path.name
        dst_label = args.out / "labels" / split / f"{image_path.stem}.txt"
        dst_image.parent.mkdir(parents=True, exist_ok=True)
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, dst_image)
        dst_label.write_text("\n".join(labels) + "\n")
        written += 1
        boxes_written += len(labels)

    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: passion_fruit\n"
    )
    print(f"images_written={written}")
    print(f"boxes_written={boxes_written}")
    print(f"data_yaml={args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
