#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: scripts/extract_frames.sh <video> <out-dir> [fps]" >&2
  exit 2
fi

video="$1"
out_dir="$2"
fps="${3:-1}"

mkdir -p "$out_dir"
ffmpeg -i "$video" -vf "fps=${fps}" "${out_dir}/frame_%05d.jpg"
