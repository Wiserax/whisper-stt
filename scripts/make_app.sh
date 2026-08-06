#!/bin/bash
# Build Whisper.app — a double-clickable launcher for the GUI.
#
# The app bundle hardcodes absolute paths, so it is not committed to git.
# Run this once after cloning to generate your own copy.
set -euo pipefail

DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP="$DIR/Whisper.app"
PY="$DIR/venv/bin/python3"

if [ ! -x "$PY" ]; then
    echo "No virtualenv found at $DIR/venv"
    echo "Run the setup steps in README.md first."
    exit 1
fi

rm -rf "$APP"
osacompile -o "$APP" -e \
    "do shell script \"$PY $DIR/stt_app.py > $DIR/whisper.log 2>&1 &\""

echo "Built: $APP"
echo
echo "First launch: right-click the app and choose Open, then grant"
echo "Microphone and Accessibility permissions when macOS asks."
