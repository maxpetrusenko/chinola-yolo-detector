#!/usr/bin/env python3
"""Run YOLO-World zero-shot detection and export CSV detections."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO


DEFAULT_CLASSES = [
    "mango",
    "green mango",
    "ripe mango",
    "passion fruit",
    "green passion fruit",
    "ripe passion fruit",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Image/video/frame folder")
    parser.add_argument("--out", default="reports/zero-shot-detections.csv")
    parser.add_argument("--model", default="yolov8s-world.pt")
    parser.add_argument("--classes", nargs="*", default=DEFAULT_CLASSES)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--max-det", type=int, default=200)
    parser.add_argument("--save", action="store_true", help="Save annotated predictions")
    parser.add_argument("--project", default="reports")
    parser.add_argument("--name", default="zero-shot")
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    project = str(Path(args.project).resolve())
    model = YOLO(args.model)
    model.set_classes(args.classes)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        max_det=args.max_det,
        save=args.save,
        project=project,
        name=args.name,
    )

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"])
        for result in results:
            names = result.names
            image = Path(result.path).name
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(box.cls.item())
                x1, y1, x2, y2 = [round(float(v), 2) for v in box.xyxy[0].tolist()]
                writer.writerow(
                    [
                        image,
                        class_id,
                        names.get(class_id, str(class_id)),
                        round(float(box.conf.item()), 4),
                        x1,
                        y1,
                        x2,
                        y2,
                    ]
                )

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
