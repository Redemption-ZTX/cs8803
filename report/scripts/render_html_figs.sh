#!/bin/bash
# Render HTML figures to PDF via headless Chrome with explicit page sizing.
set -euo pipefail

cd "$(dirname "$0")/.."

HTML_DIR="html"
OUT_DIR="figures"
mkdir -p "$OUT_DIR"

# Figure-specific page sizes (width_inches, height_inches)
declare -A PAPER_W=( [pipeline]=5.5   [architecture]=5.5 )
declare -A PAPER_H=( [pipeline]=5.5   [architecture]=2.1 )

for html in "$HTML_DIR"/*.html; do
  name=$(basename "$html" .html)
  out="$OUT_DIR/${name}.pdf"
  pw=${PAPER_W[$name]:-11.0}
  ph=${PAPER_H[$name]:-3.5}
  echo "rendering $html -> $out (${pw}in x ${ph}in)"
  google-chrome \
    --headless=new \
    --disable-gpu \
    --no-sandbox \
    --no-pdf-header-footer \
    --hide-scrollbars \
    --virtual-time-budget=1500 \
    --print-to-pdf="$out" \
    --no-margins \
    "file://$(pwd)/$html" 2>/dev/null
done

# List output
ls -la "$OUT_DIR"/pipeline.pdf
