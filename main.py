"""
Driver Drowsiness Detection System — Desktop Launcher (FYP I, KFUEIT 2026).

Authors : Muhammad Ashraf Khan Mazari · Muhammad Umair Mukhtar
Supervisor: Dr. Shadab Alam Hashmi

This launcher provides:
  • Tkinter GUI to select the detection mode
  • "Start live detection" — MediaPipe / Haar cascade (no dlib model required)
  • "Dlib classic script" — blinkDetect.py (requires landmark model)
  • "Open web dashboard" — launches Streamlit in the browser
  • "Face detection test" — Haar cascade quick test
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Launch helpers
# ---------------------------------------------------------------------------

def launch_streamlit() -> None:
    """Open the Streamlit web dashboard in the default browser."""
    app = ROOT / "streamlit_app" / "streamlit_app.py"
    subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(app)],
        cwd=str(ROOT),
    )
    messagebox.showinfo(
        "Web dashboard launched",
        "The web dashboard is starting.\n"
        "It will open automatically in your default browser.\n\n"
        "If it does not open, navigate to:\n  http://localhost:8501",
    )


def run_dlib_blink() -> None:
    """Launch the classic Dlib EAR blink-detection script."""
    model = ROOT / "models" / "shape_predictor_68_face_landmarks.dat"
    if not model.is_file():
        messagebox.showerror(
            "Dlib model missing",
            "The file\n  models/shape_predictor_68_face_landmarks.dat\n"
            "was not found.\n\n"
            "Download it from:\n"
            "  https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2\n\n"
            "Extract and place the .dat file in the models/ folder, then try again.\n\n"
            "Tip: Use 'Start live detection' for OpenCV mode (no extra download).",
        )
        return
    script = ROOT / "blinkDetect.py"
    subprocess.Popen([sys.executable, str(script)], cwd=str(ROOT))


def run_face_detection() -> None:
    """Quick Haar-cascade face detection test."""
    if not messagebox.askyesno(
        "Camera access",
        "Open webcam for Haar cascade face-detection test?\n\n"
        "Press Q in the video window to quit.",
    ):
        return
    subprocess.Popen([sys.executable, str(ROOT / "face-try.py")], cwd=str(ROOT))


def run_desktop_detection(status_label: tk.Label) -> None:
    """
    Live OpenCV + Dlib detection loop in the current process.
    Runs in a background thread so the Tkinter GUI remains responsive.
    Updates `status_label` with the current detection state.
    """
    from drowsiness_core import (
        DEFAULT_ALARM_PATH,
        DrowsinessEngine,
        State,
        ALARM_COOLDOWN_SEC,
        play_alarm_async,
        reset_alarm_gate,
    )

    if not messagebox.askyesno(
        "Camera access",
        "Allow this application to use your webcam for drowsiness detection?\n\n"
        "Click Yes to open the camera and start monitoring.\n"
        "Press Q in the video window to stop.",
    ):
        return

    def _run() -> None:
        # ---- Camera init ---------------------------------------------------
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if sys.platform == "win32" else 0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror(
                "Camera error",
                "Could not open webcam.\n\nCheck that:\n"
                "• The camera is connected and not in use by another application.\n"
                "• You have granted camera permission.",
            )
            return

        engine        = DrowsinessEngine()
        last_alert    = 0.0
        window_title  = "Driver Drowsiness — press Q to quit, R to reset"

        # State → alarm cooldown multiplier (higher severity = shorter cooldown)
        _COOLDOWN = {
            State.DROWSY_WARNING:         ALARM_COOLDOWN_SEC,
            State.DROWSY_ALERT:           ALARM_COOLDOWN_SEC / 2,
            State.POSSIBLE_UNRESPONSIVE:  ALARM_COOLDOWN_SEC / 4,
        }

        # Status label colours (Tkinter colour strings)
        _STATUS_FG = {
            State.NORMAL:               "#166534",
            State.EYES_CLOSING:         "#155e75",
            State.DROWSY_WARNING:       "#9a3412",
            State.DROWSY_ALERT:         "#991b1b",
            State.POSSIBLE_UNRESPONSIVE:"#7f1d1d",
            State.RECOVERY:             "#065f46",
            State.NO_FACE:              "#475569",
            State.INITIALIZING:         "#475569",
        }

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                frame = engine.process_bgr_frame(frame)
                current_state = engine.metrics.status

                # Update GUI label from the worker thread
                colour = _STATUS_FG.get(current_state, "#475569")
                status_label.config(
                    text=f"Status: {current_state}",
                    fg=colour,
                )

                # Alarm logic with state-appropriate cooldown
                if engine.is_alarming:
                    now = time.time()
                    cd  = _COOLDOWN.get(current_state, ALARM_COOLDOWN_SEC)
                    if now - last_alert >= cd:
                        last_alert = now
                        play_alarm_async(DEFAULT_ALARM_PATH)
                else:
                    reset_alarm_gate()

                cv2.imshow(window_title, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key in (ord("r"), ord("R")):
                    engine.reset()
                    last_alert = 0.0
                    reset_alarm_gate()

        finally:
            engine.close()
            cap.release()
            cv2.destroyAllWindows()
            status_label.config(text="Status: Stopped", fg="#475569")

    # Import cv2 lazily (only needed in detection mode)
    try:
        import cv2  # noqa: F401
    except ImportError:
        messagebox.showerror(
            "Import error",
            "opencv-python is not installed.\n\nRun:\n  pip install opencv-python",
        )
        return

    threading.Thread(target=_run, daemon=True).start()


# ---------------------------------------------------------------------------
# Tkinter application window
# ---------------------------------------------------------------------------
class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Driver Drowsiness Detection System — FYP I")
        self.geometry("780x560")
        self.minsize(680, 500)
        self.configure(bg="#0f172a")

        # ---- Styles -------------------------------------------------------
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Dark.TFrame",  background="#0f172a")
        style.configure("Card.TFrame",  background="#1e293b")
        style.configure(
            "Title.TLabel",
            background="#0f172a", foreground="#f8fafc",
            font=("Segoe UI", 22, "bold"),
        )
        style.configure(
            "Sub.TLabel",
            background="#0f172a", foreground="#94a3b8",
            font=("Segoe UI", 10),
        )
        style.configure(
            "Card.TLabel",
            background="#1e293b", foreground="#e2e8f0",
            font=("Segoe UI", 10),
        )
        style.configure(
            "Action.TButton",
            font=("Segoe UI", 11, "bold"),
            padding=(14, 10),
        )
        style.configure(
            "Small.TButton",
            font=("Segoe UI", 10),
            padding=(10, 7),
        )

        outer = ttk.Frame(self, style="Dark.TFrame", padding=28)
        outer.pack(fill=tk.BOTH, expand=True)

        # ---- Header -------------------------------------------------------
        ttk.Label(
            outer, text="Driver Drowsiness Detection", style="Title.TLabel"
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text="KFUEIT · FYP I · Department of Data Science & AI · 2026",
            style="Sub.TLabel",
        ).pack(anchor="w", pady=(2, 20))

        # ---- Status indicator ---------------------------------------------
        self._status_label = tk.Label(
            outer,
            text="Status: Ready",
            bg="#0f172a", fg="#94a3b8",
            font=("Segoe UI", 12, "bold"),
        )
        self._status_label.pack(anchor="w", pady=(0, 16))

        # ---- Primary action card -----------------------------------------
        card = tk.Frame(outer, bg="#1e293b", highlightbackground="#334155", highlightthickness=1)
        card.pack(fill=tk.X, pady=(0, 16))
        inner = tk.Frame(card, bg="#1e293b", padx=20, pady=18)
        inner.pack(fill=tk.X)

        tk.Label(
            inner,
            text="Recommended: Start live detection for immediate real-time monitoring.\n"
                 "No additional downloads required — uses the OpenCV backend by default.\n"
                 "If the Dlib landmark model is present in models/, it activates automatically for higher accuracy.",
            bg="#1e293b", fg="#cbd5e1",
            font=("Segoe UI", 10),
            justify=tk.LEFT, wraplength=680,
        ).pack(anchor="w")

        btn_row = tk.Frame(inner, bg="#1e293b")
        btn_row.pack(fill=tk.X, pady=(16, 0))

        ttk.Button(
            btn_row,
            text="▶  Start live detection",
            style="Action.TButton",
            command=lambda: run_desktop_detection(self._status_label),
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(
            btn_row,
            text="🌐  Open web dashboard",
            style="Action.TButton",
            command=launch_streamlit,
        ).pack(side=tk.LEFT, padx=(0, 10))

        # ---- Secondary actions card --------------------------------------
        card2 = tk.Frame(outer, bg="#1e293b", highlightbackground="#334155", highlightthickness=1)
        card2.pack(fill=tk.X, pady=(0, 16))
        inner2 = tk.Frame(card2, bg="#1e293b", padx=20, pady=14)
        inner2.pack(fill=tk.X)

        tk.Label(
            inner2,
            text="Additional modes:",
            bg="#1e293b", fg="#94a3b8",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", pady=(0, 10))

        btn_row2 = tk.Frame(inner2, bg="#1e293b")
        btn_row2.pack(fill=tk.X)

        ttk.Button(
            btn_row2,
            text="Classic Dlib EAR script",
            style="Small.TButton",
            command=run_dlib_blink,
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(
            btn_row2,
            text="Face detection test",
            style="Small.TButton",
            command=run_face_detection,
        ).pack(side=tk.LEFT)

        # ---- Info card ---------------------------------------------------
        info_card = tk.Frame(outer, bg="#0c1a2e", highlightbackground="#1e3a5f", highlightthickness=1)
        info_card.pack(fill=tk.X, pady=(0, 20))
        tk.Label(
            info_card,
            text=(
                "ℹ  The system auto-selects the Dlib backend if models/shape_predictor_68_face_landmarks.dat is present.\n"
                "   Otherwise it uses the built-in Haar cascade (no extra download needed).\n"
                "   Detection states: Normal → Eyes Closing → Drowsy Warning → Drowsy Alert → Possible Unresponsiveness → Recovery"
            ),
            bg="#0c1a2e", fg="#7ea6d8",
            font=("Segoe UI", 9),
            justify=tk.LEFT, wraplength=700, padx=14, pady=10,
        ).pack(anchor="w")

        # ---- Quit ---------------------------------------------------------
        ttk.Label(outer, text="", style="Sub.TLabel").pack()
        ttk.Button(outer, text="Exit application", command=self.destroy).pack(anchor="w")


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()
