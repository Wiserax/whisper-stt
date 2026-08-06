#!/usr/bin/env python3
"""Whisper STT — Premium voice-to-text with local Whisper inference."""

import sys
import os
import math
import threading
import time
import tempfile
import wave
import subprocess

import numpy as np
import sounddevice as sd
from pynput import keyboard as kb

from PyQt6.QtCore import (
    Qt, QTimer, pyqtSignal, QThread, QRectF, QPointF, QSize,
    pyqtProperty, QPropertyAnimation, QEasingCurve,
)
from PyQt6.QtGui import (
    QFont, QColor, QPainter, QPainterPath, QPen, QBrush,
    QLinearGradient, QRadialGradient, QGuiApplication, QIcon,
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QFrame, QTextEdit,
    QScrollArea, QSizePolicy, QGraphicsDropShadowEffect,
)

# ── Config ──────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "ggml-large-v3.bin")
FALLBACK_MODEL = "/opt/homebrew/share/whisper-cpp/for-tests-ggml-tiny.bin"
SAMPLE_RATE = 16000
WHISPER_CLI = "/opt/homebrew/bin/whisper-cli"
GLOBAL_HOTKEY = "<ctrl>+k"
HOTKEY_HINT = "Ctrl+K"

LANGUAGES = [
    ("ru", "Russian"),
    ("en", "English"),
    ("auto", "Auto-detect"),
]

def _check_accessibility(prompt=True):
    """Return True if this process is trusted for Accessibility events.

    If ``prompt`` is True and the process is not yet trusted, macOS shows
    the system dialog asking the user to grant Accessibility permission.
    The dialog attribution goes to the *responsible binary* of the current
    process — Terminal when launched from a shell, the Whisper.app applet
    when launched by double-click.
    """
    try:
        from ApplicationServices import AXIsProcessTrustedWithOptions
        from CoreFoundation import CFDictionaryCreate, kCFTypeDictionaryKeyCallBacks, kCFTypeDictionaryValueCallBacks
        # The documented option key is the CFString "AXTrustedCheckOptionPrompt".
        # Passing it as a Python string works with pyobjc's bridging.
        options = {"AXTrustedCheckOptionPrompt": bool(prompt)}
        return bool(AXIsProcessTrustedWithOptions(options))
    except Exception:
        # If the framework import itself fails we can't verify — assume not
        # trusted so the UI shows the banner instead of silent failure.
        return False


def _frontmost_app_info():
    """Return (pid, bundle_id) of the current frontmost app, or (None, None)."""
    try:
        from AppKit import NSWorkspace
        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        if app is None:
            return (None, None)
        return (int(app.processIdentifier()), str(app.bundleIdentifier() or ""))
    except Exception:
        return (None, None)


def _activate_app(pid):
    """Bring the app with the given PID to the front. Returns True on success."""
    if not pid:
        return False
    try:
        from AppKit import NSRunningApplication
        app = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
        if app is None:
            return False
        # NSApplicationActivateIgnoringOtherApps = 1 << 1 = 2
        app.activateWithOptions_(2)
        return True
    except Exception:
        return False


def _paste_at_cursor(text, target_pid=None):
    """Copy text to clipboard and simulate Cmd+V in the target app.

    If target_pid is given, reactivate that app first so the keystroke
    lands in the user's original window instead of Whisper itself.
    """
    from Quartz import (
        CGEventCreateKeyboardEvent, CGEventPost,
        kCGHIDEventTap, CGEventSetFlags, kCGEventFlagMaskCommand,
    )
    subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)

    if target_pid:
        _activate_app(target_pid)
        time.sleep(0.18)
    else:
        time.sleep(0.1)

    ev_down = CGEventCreateKeyboardEvent(None, 9, True)   # keycode 9 = 'v'
    CGEventSetFlags(ev_down, kCGEventFlagMaskCommand)
    ev_up = CGEventCreateKeyboardEvent(None, 9, False)
    CGEventSetFlags(ev_up, kCGEventFlagMaskCommand)
    CGEventPost(kCGHIDEventTap, ev_down)
    CGEventPost(kCGHIDEventTap, ev_up)


# ── Design tokens ───────────────────────────────────────────

class T:
    """Design tokens — Apple-grade dark theme."""
    BG_PRIMARY = "#0D0D0F"
    BG_SECONDARY = "#161618"
    BG_ELEVATED = "#1C1C1F"
    BG_CARD = "#222226"
    BG_INPUT = "#1A1A1E"

    ACCENT = "#7C5CFC"
    ACCENT_LIGHT = "#A78BFA"
    ACCENT_CYAN = "#22D3EE"
    ACCENT_GLOW = "rgba(124, 92, 252, 0.3)"

    TEXT_PRIMARY = "#F5F5F7"
    TEXT_SECONDARY = "#A1A1AA"
    TEXT_TERTIARY = "#52525B"
    TEXT_DISABLED = "#3F3F46"

    SUCCESS = "#34D399"
    WARNING = "#FBBF24"
    ERROR = "#F87171"
    RECORDING = "#EF4444"
    RECORDING_GLOW = "rgba(239, 68, 68, 0.35)"

    BORDER = "#27272A"
    BORDER_SUBTLE = "#1E1E22"

    R_SM = 8
    R_MD = 12
    R_LG = 16
    R_XL = 24
    R_PILL = 100

    S1 = 4; S2 = 8; S3 = 12; S4 = 16
    S5 = 20; S6 = 24; S7 = 32; S8 = 40


# ── Waveform visualizer ────────────────────────────────────

class WaveformRing(QWidget):
    """Circular waveform around the mic button."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(220, 220)
        self._bars = [0.0] * 48
        self._target_bars = [0.0] * 48
        self._active = False
        self._phase = 0.0
        self._audio_level = 0.0

        self._timer = QTimer(self)
        self._timer.setInterval(33)  # ~30fps
        self._timer.timeout.connect(self._animate)
        self._timer.start()

    def set_active(self, active):
        self._active = active

    def set_audio_level(self, level):
        self._audio_level = min(1.0, level * 5)

    def _animate(self):
        self._phase += 0.06
        for i in range(len(self._bars)):
            if self._active:
                base = 0.2 + 0.8 * self._audio_level
                self._target_bars[i] = base * abs(
                    math.sin(self._phase + i * 0.25)
                ) * (0.4 + 0.6 * abs(math.sin(self._phase * 1.7 + i * 0.4)))
            else:
                self._target_bars[i] = 0.04 + 0.08 * abs(
                    math.sin(self._phase * 0.3 + i * 0.15)
                )
            self._bars[i] += (self._target_bars[i] - self._bars[i]) * 0.18
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        cx, cy = w / 2, h / 2
        inner_r = 58
        max_bar_len = 36

        bar_count = len(self._bars)

        for i, val in enumerate(self._bars):
            angle = (2 * math.pi * i) / bar_count - math.pi / 2
            bar_len = max(2, val * max_bar_len)

            x1 = cx + inner_r * math.cos(angle)
            y1 = cy + inner_r * math.sin(angle)
            x2 = cx + (inner_r + bar_len) * math.cos(angle)
            y2 = cy + (inner_r + bar_len) * math.sin(angle)

            # Color: violet → red when recording, violet → cyan when idle
            t = i / (bar_count - 1)
            if self._active:
                r = int(239 * (1 - t * 0.3))
                g = int(68 + 30 * t)
                b = int(68 + 50 * t)
            else:
                r = int(124 * (1 - t) + 34 * t)
                g = int(92 * (1 - t) + 211 * t)
                b = int(252 * (1 - t) + 238 * t)
            alpha = int(120 + 135 * val)
            color = QColor(r, g, b, alpha)

            pen = QPen(color, 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        p.end()


# ── Mic button ──────────────────────────────────────────────

class MicButton(QPushButton):
    """Large circular mic button with glow effect."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(100, 100)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._recording = False
        self._hover = False
        self._pulse_phase = 0.0

        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def set_recording(self, rec):
        self._recording = rec
        self.update()

    def _tick(self):
        self._pulse_phase += 0.08
        if self._recording or self._hover:
            self.update()

    def enterEvent(self, event):
        self._hover = True
        self.update()

    def leaveEvent(self, event):
        self._hover = False
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        cx, cy = w / 2, h / 2
        r = 44

        # Outer glow
        if self._recording:
            pulse = 0.5 + 0.5 * math.sin(self._pulse_phase)
            glow_r = r + 8 + 6 * pulse
            glow = QRadialGradient(QPointF(cx, cy), glow_r)
            glow.setColorAt(0, QColor(239, 68, 68, int(80 * pulse)))
            glow.setColorAt(1, QColor(239, 68, 68, 0))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(glow))
            p.drawEllipse(QPointF(cx, cy), glow_r, glow_r)

        elif self._hover:
            glow = QRadialGradient(QPointF(cx, cy), r + 10)
            glow.setColorAt(0, QColor(124, 92, 252, 50))
            glow.setColorAt(1, QColor(124, 92, 252, 0))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(glow))
            p.drawEllipse(QPointF(cx, cy), r + 10, r + 10)

        # Circle background
        if self._recording:
            bg = QColor(T.RECORDING)
        else:
            grad = QLinearGradient(cx - r, cy - r, cx + r, cy + r)
            grad.setColorAt(0, QColor(T.ACCENT))
            grad.setColorAt(1, QColor("#5B3FD9"))
            bg = grad

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(bg))
        p.drawEllipse(QPointF(cx, cy), r, r)

        # Icon: mic when idle, stop square when recording
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(255, 255, 255, 230)))

        if self._recording:
            # Stop icon (rounded square)
            sq = 16
            path = QPainterPath()
            path.addRoundedRect(QRectF(cx - sq/2, cy - sq/2, sq, sq), 3, 3)
            p.drawPath(path)
        else:
            # Mic icon
            mic_w, mic_h = 12, 18
            mic_r = mic_w / 2

            # Mic body
            body = QPainterPath()
            body.addRoundedRect(
                QRectF(cx - mic_w/2, cy - mic_h/2 - 4, mic_w, mic_h),
                mic_r, mic_r
            )
            p.drawPath(body)

            # Mic arc
            pen = QPen(QColor(255, 255, 255, 230), 2.5,
                       Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            arc_w = mic_w + 10
            arc_h = mic_h + 4
            p.drawArc(
                QRectF(cx - arc_w/2, cy - arc_h/2 - 4, arc_w, arc_h),
                -30 * 16, -120 * 16
            )

            # Stem
            stem_top = cy + mic_h/2 - 2
            p.drawLine(QPointF(cx, stem_top), QPointF(cx, stem_top + 10))
            p.drawLine(QPointF(cx - 6, stem_top + 10), QPointF(cx + 6, stem_top + 10))

        p.end()


# ── Status badge ────────────────────────────────────────────

class StatusBadge(QWidget):
    """Animated status indicator."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(28)
        self._status = "ready"
        self._text = "Ready"
        self._pulse_phase = 0.0

        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def _tick(self):
        self._pulse_phase += 0.1
        self.update()

    def set_status(self, status, text=""):
        self._status = status
        self._text = text
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        h = self.height()

        dot_r = 4
        dot_x = 12
        dot_y = h / 2

        colors = {
            "ready": QColor(T.SUCCESS),
            "recording": QColor(T.RECORDING),
            "transcribing": QColor(T.WARNING),
            "error": QColor(T.ERROR),
        }
        color = colors.get(self._status, QColor(T.ACCENT_LIGHT))

        # Pulse for non-ready states
        pulse = 0.5 + 0.5 * math.sin(self._pulse_phase)
        if self._status != "ready":
            glow_r = dot_r + 4 * pulse
            gc = QColor(color)
            gc.setAlphaF(0.3 * pulse)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(gc))
            p.drawEllipse(QPointF(dot_x, dot_y), glow_r, glow_r)

        p.setBrush(QBrush(color))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(dot_x, dot_y), dot_r, dot_r)

        p.setPen(QPen(QColor(T.TEXT_SECONDARY)))
        p.setFont(QFont(".AppleSystemUIFont", 11))
        p.drawText(int(dot_x + dot_r + 10), int(dot_y + 4), self._text)
        p.end()

    def minimumSizeHint(self):
        return QSize(200, 28)


# ── Transcription card ──────────────────────────────────────

class TranscriptionCard(QFrame):
    """Single transcription result with copy button."""

    def __init__(self, text, duration, lang, parent=None):
        super().__init__(parent)
        self.text = text
        self.setStyleSheet(f"""
            TranscriptionCard {{
                background: {T.BG_CARD};
                border: 1px solid {T.BORDER_SUBTLE};
                border-radius: {T.R_MD}px;
                padding: 0;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        # Header row: timestamp + duration + lang
        header = QHBoxLayout()
        header.setSpacing(8)

        time_str = time.strftime("%H:%M:%S")
        meta = QLabel(f"{time_str}  ·  {duration:.1f}s  ·  {lang.upper()}")
        meta.setStyleSheet(f"color: {T.TEXT_TERTIARY}; font-size: 11px;")
        meta.setFont(QFont("Menlo", 10))
        header.addWidget(meta)
        header.addStretch()

        # Copy button
        copy_btn = QPushButton("Copy")
        copy_btn.setFixedSize(56, 26)
        copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        copy_btn.setStyleSheet(f"""
            QPushButton {{
                background: {T.BG_ELEVATED};
                color: {T.TEXT_SECONDARY};
                border: 1px solid {T.BORDER};
                border-radius: 6px;
                font-size: 11px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background: {T.BG_CARD};
                color: {T.TEXT_PRIMARY};
                border-color: {T.ACCENT};
            }}
        """)
        copy_btn.clicked.connect(self._copy)
        header.addWidget(copy_btn)

        # Paste button
        paste_btn = QPushButton("Paste")
        paste_btn.setFixedSize(56, 26)
        paste_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        paste_btn.setStyleSheet(f"""
            QPushButton {{
                background: {T.BG_ELEVATED};
                color: {T.TEXT_SECONDARY};
                border: 1px solid {T.BORDER};
                border-radius: 6px;
                font-size: 11px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background: {T.BG_CARD};
                color: {T.TEXT_PRIMARY};
                border-color: {T.ACCENT};
            }}
        """)
        paste_btn.clicked.connect(self._paste)
        header.addWidget(paste_btn)

        layout.addLayout(header)

        # Text
        text_label = QLabel(text)
        text_label.setWordWrap(True)
        text_label.setStyleSheet(f"""
            color: {T.TEXT_PRIMARY};
            font-size: 14px;
            line-height: 1.5;
        """)
        text_label.setFont(QFont(".AppleSystemUIFont", 13))
        text_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(text_label)

    def _copy(self):
        clipboard = QGuiApplication.clipboard()
        clipboard.setText(self.text)

    def _paste(self):
        threading.Thread(
            target=_paste_at_cursor, args=(self.text,), daemon=True
        ).start()


# ── Transcription worker ───────────────────────────────────

class TranscribeWorker(QThread):
    """Runs whisper-cli in a background thread."""
    finished = pyqtSignal(str, float)  # text, duration
    error = pyqtSignal(str)

    def __init__(self, audio_data, model_path, language):
        super().__init__()
        self.audio_data = audio_data
        self.model_path = model_path
        self.language = language

    def run(self):
        duration = len(self.audio_data) / SAMPLE_RATE

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
            with wave.open(f, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(
                    (self.audio_data * 32767).astype(np.int16).tobytes()
                )

        try:
            lang_args = ["-l", self.language] if self.language != "auto" else []
            result = subprocess.run(
                [WHISPER_CLI,
                 "-m", self.model_path,
                 "-f", wav_path,
                 "--no-timestamps",
                 "-t", "4",
                 ] + lang_args,
                capture_output=True, text=True, timeout=60
            )
            text = result.stdout.strip()
            text = text.replace("[BLANK_AUDIO]", "").strip()

            if text:
                self.finished.emit(text, duration)
            else:
                self.error.emit("No speech detected")
        except subprocess.TimeoutExpired:
            self.error.emit("Transcription timed out")
        except Exception as e:
            self.error.emit(str(e))
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass


# ── Main window ─────────────────────────────────────────────

class WhisperApp(QMainWindow):
    _hotkey_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Whisper STT")
        self.setFixedSize(440, 680)

        # Resolve model
        if os.path.exists(MODEL_PATH):
            self.model_path = MODEL_PATH
            self.model_name = "large-v3"
        elif os.path.exists(FALLBACK_MODEL):
            self.model_path = FALLBACK_MODEL
            self.model_name = "tiny (fallback)"
        else:
            self.model_path = None
            self.model_name = "none"

        self.whisper_ok = os.path.exists(WHISPER_CLI)

        # Ask macOS for Accessibility permission up-front. Without it pynput
        # cannot receive global keys and CGEventPost cannot synthesize Cmd+V.
        # The call pops the system prompt the first time; subsequent launches
        # return the remembered verdict instantly.
        self.ax_ok = _check_accessibility(prompt=True)

        self.language = "ru"
        self.recording = False
        self.audio_chunks = []
        self.stream = None
        self._worker = None
        self._last_toggle = 0
        self._target_pid = None
        self._own_pid = os.getpid()

        self._setup_ui()
        self._start_hotkey_listener()

    def _idle_hint(self):
        return (
            "Click mic to record (hotkey disabled — no Accessibility)"
            if not self.ax_ok else
            "Click or Ctrl+K to record"
        )

    def _setup_ui(self):
        # Central widget
        central = QWidget()
        central.setStyleSheet(f"background: {T.BG_PRIMARY};")
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(24, 20, 24, 24)
        root.setSpacing(0)

        # ── Title ───────────────────────────────
        title = QLabel("Whisper")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont(".AppleSystemUIFont", 28, QFont.Weight.Bold))
        title.setStyleSheet(f"""
            color: {T.TEXT_PRIMARY};
            margin-bottom: 2px;
        """)
        root.addWidget(title)

        subtitle = QLabel("Local speech-to-text")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setFont(QFont(".AppleSystemUIFont", 13))
        subtitle.setStyleSheet(f"color: {T.TEXT_TERTIARY}; margin-bottom: 8px;")
        root.addWidget(subtitle)

        # ── Status badge ────────────────────────
        self.status = StatusBadge()
        if not self.whisper_ok:
            self.status.set_status("error", "whisper-cli not found")
        elif not self.model_path:
            self.status.set_status("error", "No model found")
        elif not self.ax_ok:
            self.status.set_status("error", "Grant Accessibility, then relaunch")
        else:
            self.status.set_status("ready", f"Model: {self.model_name}")
        status_wrap = QHBoxLayout()
        status_wrap.addStretch()
        status_wrap.addWidget(self.status)
        status_wrap.addStretch()
        root.addLayout(status_wrap)

        root.addSpacing(12)

        # ── Waveform + mic button ───────────────
        mic_container = QWidget()
        mic_container.setFixedSize(220, 220)

        self.waveform = WaveformRing(mic_container)
        self.waveform.move(0, 0)

        self.mic_btn = MicButton(mic_container)
        self.mic_btn.move(60, 60)  # center in 220x220
        self.mic_btn.clicked.connect(self._toggle_recording)

        mic_wrap = QHBoxLayout()
        mic_wrap.addStretch()
        mic_wrap.addWidget(mic_container)
        mic_wrap.addStretch()
        root.addLayout(mic_wrap)

        root.addSpacing(4)

        # ── Hint text ───────────────────────────
        hint_text = (
            "Click mic to record (hotkey disabled — no Accessibility)"
            if not self.ax_ok else
            "Click or Ctrl+K to record"
        )
        self.hint_label = QLabel(hint_text)
        self.hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hint_label.setFont(QFont(".AppleSystemUIFont", 12))
        self.hint_label.setStyleSheet(f"color: {T.TEXT_TERTIARY};")
        root.addWidget(self.hint_label)

        root.addSpacing(12)

        # ── Controls row ────────────────────────
        controls = QHBoxLayout()
        controls.setSpacing(12)

        # Language selector
        lang_label = QLabel("Language")
        lang_label.setFont(QFont(".AppleSystemUIFont", 11))
        lang_label.setStyleSheet(f"color: {T.TEXT_SECONDARY};")
        controls.addWidget(lang_label)

        self.lang_combo = QComboBox()
        for code, name in LANGUAGES:
            self.lang_combo.addItem(name, code)
        self.lang_combo.setCurrentIndex(0)  # Russian
        self.lang_combo.currentIndexChanged.connect(self._lang_changed)
        self.lang_combo.setFixedWidth(140)
        self.lang_combo.setStyleSheet(f"""
            QComboBox {{
                background: {T.BG_ELEVATED};
                color: {T.TEXT_PRIMARY};
                border: 1px solid {T.BORDER};
                border-radius: {T.R_SM}px;
                padding: 5px 12px;
                font-size: 12px;
            }}
            QComboBox:hover {{
                border-color: {T.ACCENT};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 24px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid {T.TEXT_SECONDARY};
                margin-right: 8px;
            }}
            QComboBox QAbstractItemView {{
                background: {T.BG_ELEVATED};
                color: {T.TEXT_PRIMARY};
                border: 1px solid {T.BORDER};
                selection-background-color: {T.ACCENT};
                padding: 4px;
            }}
        """)
        controls.addWidget(self.lang_combo)

        controls.addStretch()

        # Clear history
        clear_btn = QPushButton("Clear All")
        clear_btn.setFixedHeight(30)
        clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        clear_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {T.TEXT_TERTIARY};
                border: none;
                font-size: 11px;
                padding: 4px 12px;
            }}
            QPushButton:hover {{
                color: {T.ERROR};
            }}
        """)
        clear_btn.clicked.connect(self._clear_history)
        controls.addWidget(clear_btn)

        root.addLayout(controls)

        root.addSpacing(8)

        # ── Separator ───────────────────────────
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: {T.BORDER_SUBTLE};")
        root.addWidget(sep)

        root.addSpacing(8)

        # ── Transcription history ───────────────
        history_label = QLabel("Transcriptions")
        history_label.setFont(QFont(".AppleSystemUIFont", 11, QFont.Weight.DemiBold))
        history_label.setStyleSheet(f"color: {T.TEXT_SECONDARY};")
        root.addWidget(history_label)

        root.addSpacing(6)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll.setStyleSheet(f"""
            QScrollArea {{
                background: transparent;
                border: none;
            }}
            QScrollBar:vertical {{
                background: {T.BG_PRIMARY};
                width: 6px;
                border-radius: 3px;
            }}
            QScrollBar::handle:vertical {{
                background: {T.BORDER};
                border-radius: 3px;
                min-height: 30px;
            }}
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {{
                height: 0;
            }}
        """)

        self.history_widget = QWidget()
        self.history_widget.setStyleSheet("background: transparent;")
        self.history_layout = QVBoxLayout(self.history_widget)
        self.history_layout.setContentsMargins(0, 0, 4, 0)
        self.history_layout.setSpacing(8)
        self.history_layout.addStretch()

        # Empty state
        self.empty_label = QLabel("Your transcriptions will appear here")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setFont(QFont(".AppleSystemUIFont", 12))
        self.empty_label.setStyleSheet(f"""
            color: {T.TEXT_DISABLED};
            padding: 40px 0;
        """)
        self.history_layout.insertWidget(0, self.empty_label)

        scroll.setWidget(self.history_widget)
        root.addWidget(scroll, 1)

        # ── Audio level timer ───────────────────
        self._level_timer = QTimer(self)
        self._level_timer.setInterval(50)
        self._level_timer.timeout.connect(self._update_audio_level)

    # ── Recording ───────────────────────────────

    def _toggle_recording(self):
        if not self.whisper_ok or not self.model_path:
            self.status.set_status("error", "whisper-cli or model missing")
            return

        if not self.recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        # Capture which app the user was in when they triggered recording,
        # so we can paste back into it when transcription completes.
        pid, _bundle = _frontmost_app_info()
        if pid and pid != self._own_pid:
            self._target_pid = pid
        else:
            self._target_pid = None

        self.audio_chunks = []
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
        )
        self.stream.start()
        self.recording = True
        self._record_start = time.time()

        self.mic_btn.set_recording(True)
        self.waveform.set_active(True)
        self.status.set_status("recording", "Recording...")
        self.hint_label.setText("Click or Ctrl+K to stop")
        self.hint_label.setStyleSheet(f"color: {T.RECORDING};")
        self._level_timer.start()

    def _stop_recording(self):
        self.recording = False
        self._level_timer.stop()

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        self.mic_btn.set_recording(False)
        self.waveform.set_active(False)
        self.waveform.set_audio_level(0)

        if not self.audio_chunks:
            self.status.set_status("ready", f"Model: {self.model_name}")
            self.hint_label.setText(self._idle_hint())
            self.hint_label.setStyleSheet(f"color: {T.TEXT_TERTIARY};")
            return

        audio = np.concatenate(self.audio_chunks, axis=0).flatten()
        duration = len(audio) / SAMPLE_RATE

        self.status.set_status("transcribing", f"Transcribing {duration:.1f}s...")
        self.hint_label.setText("Processing...")
        self.hint_label.setStyleSheet(f"color: {T.WARNING};")

        self._worker = TranscribeWorker(audio, self.model_path, self.language)
        self._worker.finished.connect(self._on_transcribed)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _audio_callback(self, indata, frames, time_info, status):
        if self.recording:
            self.audio_chunks.append(indata.copy())

    def _update_audio_level(self):
        if self.audio_chunks:
            chunk = self.audio_chunks[-1]
            level = float(np.abs(chunk).mean())
            self.waveform.set_audio_level(level)

            elapsed = time.time() - self._record_start
            self.status.set_status("recording", f"Recording... {elapsed:.1f}s")

    # ── Results ─────────────────────────────────

    def _on_transcribed(self, text, duration):
        self.status.set_status("ready", f"Model: {self.model_name}")
        self.hint_label.setText(self._idle_hint())
        self.hint_label.setStyleSheet(f"color: {T.TEXT_TERTIARY};")

        # Hide empty state
        self.empty_label.hide()

        # Add card (newest on top)
        card = TranscriptionCard(text, duration, self.language)
        self.history_layout.insertWidget(0, card)

        # Always put text on the clipboard so manual paste works.
        QGuiApplication.clipboard().setText(text)

        # Auto-paste into the user's original app only if we captured one.
        # If recording started via the mouse inside Whisper, _target_pid is
        # None and we skip auto-paste to avoid pasting into our own UI.
        target = self._target_pid
        self._target_pid = None
        if target:
            threading.Thread(
                target=_paste_at_cursor, args=(text, target), daemon=True
            ).start()

        self._worker = None

    def _on_error(self, msg):
        self.status.set_status("error", msg)
        self.hint_label.setText("Click or Ctrl+K to try again")
        self.hint_label.setStyleSheet(f"color: {T.TEXT_TERTIARY};")
        self._worker = None

        # Reset after 3s
        def _reset():
            self.status.set_status("ready", f"Model: {self.model_name}")
            self.hint_label.setText(self._idle_hint())
        QTimer.singleShot(3000, _reset)

    # ── Language ────────────────────────────────

    def _lang_changed(self, idx):
        self.language = self.lang_combo.itemData(idx)

    # ── Clear ───────────────────────────────────

    def _clear_history(self):
        # Remove all TranscriptionCards, keep empty_label and stretch
        for i in reversed(range(self.history_layout.count())):
            item = self.history_layout.itemAt(i)
            w = item.widget() if item else None
            if isinstance(w, TranscriptionCard):
                self.history_layout.takeAt(i)
                w.deleteLater()
        self.empty_label.show()

    # ── Global hotkey (Ctrl+K) ─────────────────

    def _start_hotkey_listener(self):
        self._hotkey = None
        self._kb_listener = None
        self._hotkey_signal.connect(self._on_hotkey_fired)

        if not self.ax_ok:
            # Without Accessibility the Listener can start but will never
            # receive events — macOS silently drops them. Skip it entirely
            # so there's no confusion about why the hotkey "doesn't fire".
            return

        def _fire():
            now = time.time()
            if now - self._last_toggle < 0.4:
                return
            self._last_toggle = now
            self._hotkey_signal.emit()

        self._hotkey = kb.HotKey(kb.HotKey.parse(GLOBAL_HOTKEY), _fire)
        self._kb_listener = kb.Listener(
            on_press=lambda k: self._hotkey.press(self._kb_listener.canonical(k)),
            on_release=lambda k: self._hotkey.release(self._kb_listener.canonical(k)),
        )
        self._kb_listener.daemon = True
        self._kb_listener.start()

    def _on_hotkey_fired(self):
        self._toggle_recording()


# ── Entry point ─────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark palette
    from PyQt6.QtGui import QPalette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(T.BG_PRIMARY))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(T.TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Base, QColor(T.BG_SECONDARY))
    palette.setColor(QPalette.ColorRole.Text, QColor(T.TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Button, QColor(T.BG_ELEVATED))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(T.TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(T.ACCENT))
    app.setPalette(palette)

    window = WhisperApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
