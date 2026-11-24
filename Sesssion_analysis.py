"""
Driver Drowsiness Detection System — Session Analysis Viewer.

Authors : Muhammad Ashraf Khan Mazari · Muhammad Umair Mukhtar
Supervisor: Dr. Shadab Alam Hashmi
University: KFUEIT, Rahim Yar Khan — FYP I (2026)

Provides a simple Tkinter tabbed interface for:
  • Live Tracking  — real-time EAR / status display from the running engine
  • Session History — review of saved session summaries
"""

import json
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk, messagebox
from typing import List, Dict

ROOT = Path(__file__).resolve().parent
SESSION_LOG_PATH = ROOT / "session_log.json"


# ---------------------------------------------------------------------------
# Session log helpers
# ---------------------------------------------------------------------------

def _load_sessions() -> List[Dict]:
    if SESSION_LOG_PATH.is_file():
        try:
            with open(SESSION_LOG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, IOError):
            return []
    return []


def _save_session(summary: Dict) -> None:
    sessions = _load_sessions()
    sessions.append(summary)
    try:
        with open(SESSION_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(sessions, f, indent=2, ensure_ascii=False)
    except IOError as exc:
        print(f"[Session] Could not save log: {exc}")


# ---------------------------------------------------------------------------
# Live Tracking tab
# ---------------------------------------------------------------------------
class LiveTrackingFrame(ttk.Frame):
    def __init__(self, parent: ttk.Notebook) -> None:
        super().__init__(parent, padding=16)
        self._engine = None

        # Try to import core; gracefully handle missing deps
        try:
            from drowsiness_core import DrowsinessEngine, State
            self._engine_class = DrowsinessEngine
            self._State = State
        except ImportError:
            self._engine_class = None
            self._State = None

        ttk.Label(
            self, text="Live Detection Status",
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w", pady=(0, 12))

        info = ttk.Frame(self)
        info.pack(fill=tk.X)

        fields = ["Status", "EAR", "MAR", "Blinks", "Alerts", "Face", "Backend"]
        self._vars: Dict[str, tk.StringVar] = {}
        for i, f in enumerate(fields):
            ttk.Label(info, text=f"{f}:", font=("Segoe UI", 10, "bold")).grid(
                row=i, column=0, sticky="w", padx=(0, 12), pady=3
            )
            var = tk.StringVar(value="—")
            ttk.Label(info, textvariable=var, font=("Segoe UI", 10)).grid(
                row=i, column=1, sticky="w", pady=3
            )
            self._vars[f] = var

        ttk.Separator(self, orient="horizontal").pack(fill=tk.X, pady=12)

        btn_row = ttk.Frame(self)
        btn_row.pack(anchor="w")
        ttk.Button(btn_row, text="Start engine", command=self._start).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_row, text="Stop engine",  command=self._stop).pack(side=tk.LEFT)

        self._status_lbl = ttk.Label(self, text="", font=("Segoe UI", 9))
        self._status_lbl.pack(anchor="w", pady=(8, 0))

    def _start(self) -> None:
        if self._engine_class is None:
            messagebox.showerror("Import error", "drowsiness_core could not be imported.")
            return
        if self._engine is not None:
            return
        try:
            self._engine = self._engine_class()
            self._status_lbl.config(text="Engine started (metrics refresh every 2 s).")
            self._refresh()
        except Exception as exc:
            messagebox.showerror("Engine error", str(exc))

    def _stop(self) -> None:
        if self._engine is not None:
            self._engine.close()
            self._engine = None
        self._status_lbl.config(text="Engine stopped.")
        for v in self._vars.values():
            v.set("—")

    def _refresh(self) -> None:
        if self._engine is None:
            return
        m = self._engine.metrics
        self._vars["Status"].set(m.status)
        self._vars["EAR"].set(f"{m.ear:.4f}")
        self._vars["MAR"].set(f"{m.mar:.4f}")
        self._vars["Blinks"].set(str(m.blink_count))
        self._vars["Alerts"].set(str(m.alert_count))
        self._vars["Face"].set("Yes" if m.face_detected else "No")
        self._vars["Backend"].set(m.backend.upper())
        # Recurse every 2000 ms
        self.after(2000, self._refresh)


# ---------------------------------------------------------------------------
# Session History tab
# ---------------------------------------------------------------------------
class SessionHistoryFrame(ttk.Frame):
    def __init__(self, parent: ttk.Notebook) -> None:
        super().__init__(parent, padding=16)

        ttk.Label(
            self, text="Session History",
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w", pady=(0, 12))

        btn_row = ttk.Frame(self)
        btn_row.pack(anchor="w", pady=(0, 10))
        ttk.Button(btn_row, text="🔄 Refresh", command=self._load).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_row, text="🗑 Clear log", command=self._clear).pack(side=tk.LEFT)

        cols = ("Date/Time", "Duration (min)", "Avg EAR", "Blinks", "Alerts")
        self._tree = ttk.Treeview(self, columns=cols, show="headings", height=12)
        for col in cols:
            self._tree.heading(col, text=col)
            self._tree.column(col, width=130, anchor="center")
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=scrollbar.set)
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.LEFT, fill=tk.Y)

        self._load()

    def _load(self) -> None:
        for row in self._tree.get_children():
            self._tree.delete(row)
        sessions = _load_sessions()
        if not sessions:
            self._tree.insert("", "end", values=("No sessions recorded yet.", "", "", "", ""))
            return
        for s in reversed(sessions):
            self._tree.insert("", "end", values=(
                s.get("timestamp",    "—"),
                s.get("duration_min", "—"),
                s.get("avg_ear",      "—"),
                s.get("blinks",       "—"),
                s.get("alerts",       "—"),
            ))

    def _clear(self) -> None:
        if messagebox.askyesno("Clear log", "Delete all session history? This cannot be undone."):
            SESSION_LOG_PATH.unlink(missing_ok=True)
            self._load()


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
class SessionAnalysisApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Driver Drowsiness — Session Analysis")
        self.geometry("720x480")
        self.minsize(600, 420)
        self.configure(bg="#f8fafc")

        style = ttk.Style(self)
        style.theme_use("clam")

        notebook = ttk.Notebook(self)
        live_tab    = LiveTrackingFrame(notebook)
        history_tab = SessionHistoryFrame(notebook)
        notebook.add(live_tab,    text="📡 Live Tracking")
        notebook.add(history_tab, text="📋 Session History")
        notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        footer = ttk.Label(
            self,
            text="KFUEIT · FYP I · Driver Drowsiness Detection System · 2026",
            font=("Segoe UI", 8),
            foreground="#94a3b8",
        )
        footer.pack(side=tk.BOTTOM, pady=4)


def main() -> None:
    SessionAnalysisApp().mainloop()


if __name__ == "__main__":
    main()
