#!/bin/bash
# Encode recorded .webm clips to web mp4 + poster jpg into site/assets/clips/,
# trimming the first 2s (connect/menu). Usage: tools/encode_clips.sh [SRC_DIR]
set -e
SRC="${1:-tmp/vids}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/site/assets/clips"
MAMBA_EXE='/fsx/home/chenhao.zheng/.local/bin/micromamba'
run(){ "$MAMBA_EXE" run -n cooksim "$@"; }
mkdir -p "$DEST"
for w in "$SRC"/*.webm; do
  name=$(basename "$w" .webm)
  run ffmpeg -y -loglevel error -ss 2 -i "$w" -movflags +faststart -pix_fmt yuv420p \
    -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -crf 27 -preset veryfast "$DEST/$name.mp4"
  run ffmpeg -y -loglevel error -ss 6 -i "$w" -frames:v 1 -q:v 4 "$DEST/$name.jpg"
  echo "$name -> $(( $(stat -c%s "$DEST/$name.mp4") / 1024 ))KB"
done
echo "ENCODE DONE"
