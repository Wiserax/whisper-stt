# Whisper STT

Press **Ctrl+K**, speak, press again — your words are pasted straight into whatever
app your cursor is in. Everything runs locally on your Mac. No API key, no account,
no audio ever leaves the machine.

Two front-ends share the same engine:

- **`stt_app.py`** — a PyQt6 desktop app with a live waveform ring, language switcher,
  and a scrollable history of past transcriptions you can re-copy or re-paste.
- **`stt.py`** — a minimal terminal version, no GUI, same hotkey.

## Requirements

- macOS on Apple Silicon (the code uses Quartz, AppKit and CoreGraphics APIs)
- Python 3.9 or newer
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp), installed via Homebrew

## Setup

**1. Install whisper.cpp**

```bash
brew install whisper-cpp
```

This provides `/opt/homebrew/bin/whisper-cli`, which does the actual transcription.

**2. Clone and create a virtualenv**

```bash
git clone https://github.com/Wiserax/whisper-stt.git
cd whisper-stt
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**3. Download a model**

```bash
./scripts/download_model.sh
```

This fetches `ggml-large-v3.bin` (about 2.9 GB) from Hugging Face into `models/`.
The model is not in git because of its size. If you would rather trade accuracy for
disk space, pass a smaller name — `./scripts/download_model.sh base` — and update
`MODEL_PATH` at the top of `stt_app.py` to match.

**4. Run it**

```bash
./run_app.sh    # desktop app
./run.sh        # terminal version
```

To get a double-clickable app in Finder instead, run `./scripts/make_app.sh`. It
generates `Whisper.app` with the paths for your own machine, which is why the bundle
itself is not committed.

## macOS permissions

The app needs two permissions, both requested on first use:

- **Microphone** — to record.
- **Accessibility** — to send the Cmd+V keystroke that pastes into other apps.
  Grant this under System Settings → Privacy & Security → Accessibility.

The permission prompt is attributed to whichever binary launched the process: Terminal
if you used `run_app.sh`, `Whisper.app` if you double-clicked it. If you use both, grant
it to both. Without Accessibility the transcription still lands on your clipboard — it
just won't paste itself.

## Usage

Ctrl+K starts recording, Ctrl+K again stops it and transcribes. The hotkey is global,
so it works while another app is focused; the app remembers which window you came from
and pastes back into it.

Language defaults to Russian and can be switched to English or auto-detect in the
dropdown. In the terminal version, change the `LANGUAGE` constant near the top of
`stt.py`. To change the hotkey, edit `GLOBAL_HOTKEY` in `stt_app.py`.

## Is there an API key?

No. Transcription runs entirely on-device through whisper.cpp, so there is nothing to
configure and no usage cost. If you are looking for the cloud version, that is a
separate project — [matrix-whisper](https://github.com/Wiserax/matrix-whisper) — which
uses the Groq Whisper API and reads its key from a local `.env` file that is gitignored.

## License

MIT — see [LICENSE](LICENSE).
