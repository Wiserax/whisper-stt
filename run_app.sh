#!/bin/bash
# Launch Whisper STT GUI
DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/venv/bin/activate"
python3 "$DIR/stt_app.py"
