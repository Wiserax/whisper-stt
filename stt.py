#!/usr/bin/env python3
"""Local Whisper STT — Ctrl+K to record, transcribe, and paste."""

import threading
import time
import os
import subprocess
import tempfile
import wave
import numpy as np
from Quartz import (
    CGEventCreateKeyboardEvent, CGEventPost,
    kCGHIDEventTap, CGEventSetFlags, kCGEventFlagMaskCommand,
)
import sounddevice as sd
from pynput import keyboard

# ── Config ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "ggml-large-v3.bin")
FALLBACK_MODEL = "/opt/homebrew/share/whisper-cpp/for-tests-ggml-tiny.bin"
LANGUAGE = "ru"            # "ru", "en", "auto"
SAMPLE_RATE = 16000
HOTKEY = frozenset([keyboard.Key.ctrl_l, keyboard.KeyCode.from_char('k')])
# ────────────────────────────────────────────────────────


class VoiceTyper:
    def __init__(self):
        if os.path.exists(MODEL_PATH):
            self.model_path = MODEL_PATH
            print(f"[+] Using model: large-v3")
        elif os.path.exists(FALLBACK_MODEL):
            self.model_path = FALLBACK_MODEL
            print(f"[!] large-v3 not ready yet, using tiny fallback")
        else:
            print("[!] No model found! Download ggml-large-v3.bin to stt/models/")
            exit(1)

        print("[*] Hotkey: Ctrl+K (toggle recording)")
        print("    Text will be pasted at your cursor position.")
        print()

        self.recording = False
        self.audio_chunks = []
        self.stream = None
        self.pressed_keys = set()
        self._lock = threading.Lock()
        self._last_toggle = 0

    # ── Audio ───────────────────────────────────────────
    def _audio_callback(self, indata, frames, time_info, status):
        if self.recording:
            self.audio_chunks.append(indata.copy())

    def _start_recording(self):
        self.audio_chunks = []
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
        )
        self.stream.start()
        self.recording = True
        print("  >> Recording... (Ctrl+K to stop)")

    def _stop_and_transcribe(self):
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        if not self.audio_chunks:
            print("  -- No audio captured.")
            return

        audio = np.concatenate(self.audio_chunks, axis=0).flatten()
        duration = len(audio) / SAMPLE_RATE
        print(f"  >> Transcribing {duration:.1f}s of audio...")

        # Save as WAV for whisper-cli
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
            with wave.open(f, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes((audio * 32767).astype(np.int16).tobytes())

        try:
            lang_args = ["-l", LANGUAGE] if LANGUAGE != "auto" else []
            result = subprocess.run(
                ["whisper-cli",
                 "-m", self.model_path,
                 "-f", wav_path,
                 "--no-timestamps",
                 "-t", "4",
                 ] + lang_args,
                capture_output=True, text=True, timeout=30
            )
            text = result.stdout.strip()
            # Clean up whisper output artifacts
            text = text.replace("[BLANK_AUDIO]", "").strip()
        finally:
            os.unlink(wav_path)

        if text:
            subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
            print(f"  => {text}")
            time.sleep(0.1)
            # Cmd+V via Quartz CGEvent (works across all apps)
            # keycode 9 = 'v'
            ev_down = CGEventCreateKeyboardEvent(None, 9, True)
            CGEventSetFlags(ev_down, kCGEventFlagMaskCommand)
            ev_up = CGEventCreateKeyboardEvent(None, 9, False)
            CGEventSetFlags(ev_up, kCGEventFlagMaskCommand)
            CGEventPost(kCGHIDEventTap, ev_down)
            CGEventPost(kCGHIDEventTap, ev_up)
        else:
            print("  -- Nothing recognized.")

    # ── Hotkey ──────────────────────────────────────────
    def _toggle(self):
        now = time.time()
        if now - self._last_toggle < 0.3:
            return
        self._last_toggle = now

        with self._lock:
            if not self.recording:
                self._start_recording()
            else:
                self._stop_and_transcribe()

    def _on_press(self, key):
        self.pressed_keys.add(key)
        if self.pressed_keys >= HOTKEY:
            self.pressed_keys.clear()
            threading.Thread(target=self._toggle, daemon=True).start()

    def _on_release(self, key):
        self.pressed_keys.discard(key)

    # ── Run ─────────────────────────────────────────────
    def run(self):
        print("Ready! Listening for Ctrl+K...\n")
        with keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        ) as listener:
            try:
                listener.join()
            except KeyboardInterrupt:
                print("\nBye!")


if __name__ == "__main__":
    VoiceTyper().run()
