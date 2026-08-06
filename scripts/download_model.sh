#!/bin/bash
# Download a Whisper model into models/.
#
# Usage:  ./scripts/download_model.sh [model]
#   model defaults to "large-v3" (~2.9 GB, best quality).
#   Lighter options: base, small, medium.
set -euo pipefail

MODEL="${1:-large-v3}"
DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$DIR/models/ggml-${MODEL}.bin"
URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-${MODEL}.bin"

if [ -f "$DEST" ]; then
    echo "Already present: $DEST"
    exit 0
fi

mkdir -p "$DIR/models"
echo "Downloading ggml-${MODEL}.bin ..."
echo "This is a large file — large-v3 is about 2.9 GB."
curl -L --fail --progress-bar -o "$DEST.part" "$URL"
mv "$DEST.part" "$DEST"
echo "Done: $DEST"

if [ "$MODEL" != "large-v3" ]; then
    echo
    echo "Note: stt_app.py looks for models/ggml-large-v3.bin by default."
    echo "Edit MODEL_PATH near the top of stt_app.py to point at this file."
fi
