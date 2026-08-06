#!/bin/bash
# Launch local Whisper STT
DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/venv/bin/activate"
python3 "$DIR/stt.py"
