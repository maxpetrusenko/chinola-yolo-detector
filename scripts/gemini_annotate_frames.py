#!/usr/bin/env python3
"""Use Gemini vision to draft target boxes for sampled frames."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import re
import time
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image


PROMPTS = {
    "fruit": """Detect every visible passion fruit in this image.

Return only JSON. No markdown.

Schema:
[
  {
    "label": "passion_fruit",
    "ripeness": "green" | "ripe" | "unknown",
    "box_2d": [ymin, xmin, ymax, xmax],
    "confidence": 0.0-1.0
  }
]

Rules:
- box_2d coordinates are normalized to 0-1000.
- Count only real passion fruit fruit bodies: whole fruit or clearly visible cut halves.
- Include green, yellow, purple, red, ripe, unripe, wrinkled, damaged, and spotted passion fruit when visually clear.
- Do not label loose pulp, seeds, leaves, flowers, people, hands, text, labels, drawings, shadows, stems, rocks, mangoes, potatoes, or other fruit.
- Do not create many tiny boxes over seeds or spots. One box should cover one fruit body.
- If the object could be a different fruit or is too blurry/hidden to identify from this image alone, skip it.
- If uncertain but still plausible, use lower confidence.
- If no fruit is visible, return [].
""",
    "flower": """Detect every visible passion fruit flower in this image.

Return only JSON. No markdown.

Schema:
[
  {
    "label": "passion_flower",
    "box_2d": [ymin, xmin, ymax, xmax],
    "confidence": 0.0-1.0
  }
]

Rules:
- box_2d coordinates are normalized to 0-1000.
- Count only real visible passion fruit flowers with white petals: open or partly open Passiflora flowers where petals/corona/floral structure is visible.
- Do not count flowers without visible white petals.
- Do not count closed buds, leaves, tendrils, fruit, grass, posts, rocks, shadows, or blurry uncertain shapes.
- One box should cover one flower head.
- If no flower is visible, return [].
""",
    "leaf_disease": """Detect clearly suspicious passion fruit leaf disease or damage areas in this image.

Return only JSON. No markdown.

Schema:
[
  {
    "label": "leaf_disease",
    "box_2d": [ymin, xmin, ymax, xmax],
    "confidence": 0.0-1.0
  }
]

Rules:
- box_2d coordinates are normalized to 0-1000.
- Count visible diseased or damaged leaf regions only: brown spots, yellowing patches, lesions, necrotic edges, severe insect or pathogen-like damage.
- Do not box every healthy leaf.
- Do not label fruit, flowers, stems, soil, shadows, sun glare, rocks, irrigation lines, or blurry uncertain areas.
- Group a connected diseased leaf cluster into one box.
- If leaves look healthy or image is too unclear, return [].
""",
    "fruit_condition": """Detect visible passion fruit bodies and classify whether each fruit looks healthy or damaged/diseased.

Return only JSON. No markdown.

Schema:
[
  {
    "label": "passion_fruit_healthy" | "passion_fruit_damaged_or_diseased",
    "ripeness": "green" | "ripe" | "unknown",
    "box_2d": [ymin, xmin, ymax, xmax],
    "confidence": 0.0-1.0
  }
]

Rules:
- box_2d coordinates are normalized to 0-1000.
- Count only real visible passion fruit fruit bodies: whole fruit or clearly visible cut halves.
- Use "passion_fruit_damaged_or_diseased" when the fruit body shows clear rot, brown spots, lesions, severe pest damage, fungus-like damage, cracking, shriveling, deformation, or woodiness/disease symptoms.
- Use "passion_fruit_healthy" for fruit that appears normal from the image, even if green/unripe.
- Do not label leaves, flowers, stems, soil, people, hands, text, labels, drawings, shadows, rocks, mangoes, potatoes, or other fruit.
- Do not create many tiny boxes over seeds or spots. One box should cover one fruit body.
- If a fruit is too blurry/hidden to classify, skip it.
- If no passion fruit body is visible, return [].
""",
}

TARGET_LABELS = {
    "fruit": "passion_fruit",
    "flower": "passion_flower",
    "leaf_disease": "leaf_disease",
    "fruit_condition": ("passion_fruit_healthy", "passion_fruit_damaged_or_diseased"),
}


def api_key() -> str:
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    raise SystemExit("Missing Gemini key. Set GEMINI_API_KEY.")


def image_paths(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    suffixes = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(p for p in source.rglob("*") if p.suffix.lower() in suffixes)


def parse_json(text: str) -> list[dict]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\[", text):
            try:
                value, _ = decoder.raw_decode(text[match.start() :])
                break
            except json.JSONDecodeError:
                continue
        else:
            return []
    return value if isinstance(value, list) else []


def normalized_box(box: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    ymin, xmin, ymax, xmax = [float(v) for v in box]
    x1 = max(0.0, min(width, xmin / 1000.0 * width))
    y1 = max(0.0, min(height, ymin / 1000.0 * height))
    x2 = max(0.0, min(width, xmax / 1000.0 * width))
    y2 = max(0.0, min(height, ymax / 1000.0 * height))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def annotate_path(
    path: Path,
    model_name: str,
    key: str,
    target: str,
    min_confidence: float,
    max_retries: int,
) -> tuple[str, list[list], int, str | None]:
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            client = genai.Client(api_key=key)
            config = types.GenerateContentConfig(response_mime_type="application/json")
            image = Image.open(path)
            width, height = image.size
            response = client.models.generate_content(
                model=model_name,
                contents=[image, PROMPTS[target]],
                config=config,
            )
            detections = parse_json(response.text or "[]")
            rows = []
            expected_labels = TARGET_LABELS[target]
            if isinstance(expected_labels, str):
                expected_labels = (expected_labels,)
            for item in detections:
                label = str(item.get("label", "")).strip().lower()
                if label not in expected_labels:
                    continue
                confidence = float(item.get("confidence", 0.5))
                if confidence < min_confidence:
                    continue
                box = item.get("box_2d")
                if not isinstance(box, list) or len(box) != 4:
                    continue
                x1, y1, x2, y2 = normalized_box(box, width, height)
                if x2 - x1 < 3 or y2 - y1 < 3:
                    continue
                rows.append(
                    [
                        path.name,
                        expected_labels.index(label),
                        label,
                        round(confidence, 4),
                        round(x1, 2),
                        round(y1, 2),
                        round(x2, 2),
                        round(y2, 2),
                        item.get("ripeness", "unknown"),
                        model_name,
                    ]
                )
            return path.name, rows, len(detections), None
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            if attempt < max_retries:
                time.sleep(2**attempt)
    return path.name, [], 0, last_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", default="reports/gemini-detections.csv", type=Path)
    parser.add_argument("--model", default="gemini-3.5-flash")
    parser.add_argument("--target", choices=sorted(PROMPTS), default="fruit")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--min-confidence", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=1, help="Parallel Gemini requests. Each request still contains one image.")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--resume", action="store_true", help="Append to an existing CSV and skip images already present.")
    args = parser.parse_args()

    paths = image_paths(args.source)
    if args.limit:
        paths = paths[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    completed_images = set()
    if args.resume and args.out.exists():
        with open(args.out, newline="") as existing:
            reader = csv.DictReader(existing)
            completed_images = {row["image"] for row in reader if row.get("image")}
        paths = [path for path in paths if path.name not in completed_images]

    key = api_key()

    mode = "a" if args.resume and args.out.exists() else "w"
    with open(args.out, mode, newline="") as f:
        writer = csv.writer(f)
        if mode == "w":
            writer.writerow(["image", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "ripeness", "model"])
        workers = max(1, args.workers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(annotate_path, path, args.model, key, args.target, args.min_confidence, args.max_retries): path
                for path in paths
            }
            for future in concurrent.futures.as_completed(futures):
                image_name, rows, raw_count, error = future.result()
                if error:
                    print(f"failed {image_name}: {error}", flush=True)
                    continue
                writer.writerows(rows)
                f.flush()
                print(f"annotated {image_name}: {raw_count} raw, {len(rows)} kept", flush=True)

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
