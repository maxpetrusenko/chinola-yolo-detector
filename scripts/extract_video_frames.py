#!/usr/bin/env python3
"""Extract sampled frames from one video or a directory of videos."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    return slug[:80] or "video"


def videos_from_source(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(path for path in source.iterdir() if path.suffix.lower() in VIDEO_SUFFIXES)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=0.5)
    parser.add_argument("--width", type=int, default=960)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    videos = videos_from_source(args.source)
    if not videos:
        raise SystemExit(f"No videos found in {args.source}")

    total = 0
    for video in videos:
        prefix = slugify(video.stem)
        pattern = args.out / f"{prefix}_frame_%05d.jpg"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-vf",
            f"fps={args.fps},scale={args.width}:-1",
            str(pattern),
        ]
        subprocess.run(cmd, check=True)
        count = len(list(args.out.glob(f"{prefix}_frame_*.jpg")))
        total += count
        print(f"{video.name}: frames={count}")

    print(f"total_frames={total}")
    print(f"out={args.out}")


if __name__ == "__main__":
    main()
