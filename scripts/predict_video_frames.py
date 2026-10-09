#!/usr/bin/env python3
"""Run a trained YOLO detector on frames/images and export detections CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", default="reports/detections.csv")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    results = model.predict(source=args.source, conf=args.conf, save=True)

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
