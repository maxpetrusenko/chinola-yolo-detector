#!/usr/bin/env python3
"""Download the Kaggle passion fruit disease dataset."""

from __future__ import annotations

import subprocess
from pathlib import Path


DATASET = "roynjuguna/passion-fruit-disease-detection"
OUT_DIR = Path("data/raw/kaggle/passion-fruit-disease-detection")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            DATASET,
            "-p",
            str(OUT_DIR),
            "--unzip",
        ],
        check=True,
    )
    print(f"Downloaded {DATASET} -> {OUT_DIR}")


if __name__ == "__main__":
    main()
