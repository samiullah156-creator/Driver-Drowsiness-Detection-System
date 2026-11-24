"""
Driver Drowsiness Detection System — Professional Streamlit Web Dashboard.

Authors : Muhammad Ashraf Khan Mazari · Muhammad Umair Mukhtar
Supervisor: Dr. Shadab Alam Hashmi
University: KFUEIT, Rahim Yar Khan — FYP I (2026)

Run:
    streamlit run streamlit_app/streamlit_app.py

Features (FR/NFR coverage):
  FR-01  Live video capture via WebRTC
  FR-02  Face detection (Haar or Dlib)
  FR-03  Eye landmark tracking and EAR
  FR-04  5-state drowsiness detection
  FR-05  Audible alarm + visual alert banners
  FR-06  Per-frame feature extraction
  FR-07  Live metrics dashboard
  NFR-01 <100 ms latency (async processing)
  NFR-02 Low false-positive with hysteresis
  NFR-03 Intuitive, professional dashboard
  NFR-04 Long-duration stability (no unbounded state growth)
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from datetime import datetime

import av
import cv2
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drowsiness_core import (           # noqa: E402
    DEFAULT_ALARM_PATH,
    DrowsinessEngine,
    LiveMetrics,
    State,
    play_alarm_async,
    reset_alarm_gate,
    ALARM_COOLDOWN_SEC,
)

# ---------------------------------------------------------------------------
# Page configuration (MUST be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Driver Drowsiness Detection | KFUEIT FYP",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — professional dark-teal theme
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* ---- Header --------------------------------------------------------- */
.app-header {
    background: linear-gradient(135deg, #0f172a 0%, #134e4a 60%, #0d9488 100%);
    padding: 1.5rem 2rem;
    border-radius: 14px;
    color: #f8fafc;
    margin-bottom: 1.25rem;
    box-shadow: 0 8px 32px rgba(15,23,42,0.28);
}
.app-header h1 { margin: 0; font-size: 1.75rem; font-weight: 700; letter-spacing:-0.5px; }
.app-header p  { margin: 0.4rem 0 0; opacity: 0.85; font-size: 0.95rem; }

/* ---- Status badges -------------------------------------------------- */
.badge {
    display: inline-block;
    padding: 0.3rem 0.85rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.9rem;
    letter-spacing: 0.3px;
}
.badge-normal     { background:#dcfce7; color:#166534; }
.badge-closing    { background:#cffafe; color:#155e75; }
.badge-warning    { background:#fff7ed; color:#9a3412; }
.badge-alert      { background:#fee2e2; color:#991b1b; }
.badge-unresponsive { background:#1e1b4b; color:#e0e7ff; animation: pulse-border 1.2s infinite; }
.badge-recovery   { background:#d1fae5; color:#065f46; }
.badge-noface     { background:#f1f5f9; color:#475569; }

@keyframes pulse-border {
    0%,100% { box-shadow: 0 0 0 0 rgba(220,38,38,.5); }
    50%      { box-shadow: 0 0 0 6px rgba(220,38,38,0); }
}

/* ---- Metric cards --------------------------------------------------- */
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 0.85rem 1.1rem;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
    margin-bottom: 0.5rem;
}
.metric-label { font-size:0.8rem; color:#64748b; text-transform:uppercase;
                letter-spacing:0.5px; margin-bottom:0.25rem; }
.metric-value { font-size:1.35rem; font-weight:700; color:#0f172a; }

/* ---- Alert log table ------------------------------------------------ */
.alert-row {
    display:flex; align-items:center; gap:0.75rem;
    padding:0.5rem 0.75rem; border-radius:8px;
    background:#f8fafc; border:1px solid #e2e8f0;
    margin-bottom:0.4rem; font-size:0.88rem;
}
.alert-ts  { color:#64748b; font-size:0.8rem; min-width:4.5rem; }
.alert-st  { font-weight:600; flex:1; }

/* ---- Consent box ---------------------------------------------------- */
.consent-box {
    background: #f8fafc;
    border: 2px dashed #94a3b8;
    border-radius: 16px;
    padding: 2.5rem 2rem;
    text-align: center;
}

/* ---- How-it-works --------------------------------------------------- */
.how-step {
    display:flex; gap:0.75rem; align-items:flex-start;
    margin-bottom:0.6rem;
}
.step-num {
    background:#0d9488; color:#fff; font-weight:700;
    font-size:0.8rem; min-width:1.6rem; height:1.6rem;
    border-radius:50%; display:inline-flex;
    align-items:center; justify-content:center;
}

/* ---- Misc ----------------------------------------------------------- */
div[data-testid="stMetricValue"] { font-size:1.3rem; }
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "camera_consent":   False,
    "last_alert_time":  0.0,
    "session_start":    None,
    "engine":           None,
    "ear_threshold":    0.27,
    "consec_frames":    18,
    "escalation_frames": 60,
    "unresponsive_frames": 120,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _get_engine() -> DrowsinessEngine:
    """Create or return cached engine, recreated if thresholds changed."""
    if st.session_state.engine is None:
        st.session_state.engine = DrowsinessEngine(
            ear_threshold=st.session_state.ear_threshold,
            consec_closed_frames=st.session_state.consec_frames,
            escalation_frames=st.session_state.escalation_frames,
            unresponsive_frames=st.session_state.unresponsive_frames,
        )
        st.session_state.session_start = datetime.now().strftime("%H:%M:%S")
    return st.session_state.engine


# ---------------------------------------------------------------------------
# State → badge CSS class mapping
# ---------------------------------------------------------------------------
_STATE_BADGE = {
    State.NORMAL:                "badge-normal",
    State.EYES_CLOSING:          "badge-closing",
    State.DROWSY_WARNING:        "badge-warning",
    State.DROWSY_ALERT:          "badge-alert",
    State.POSSIBLE_UNRESPONSIVE: "badge-unresponsive",
    State.RECOVERY:              "badge-recovery",
    State.NO_FACE:               "badge-noface",
    State.INITIALIZING:          "badge-noface",
}


def _status_badge(status: str) -> str:
    cls = _STATE_BADGE.get(status, "badge-noface")
    label = "⚠ POSSIBLE UNRESPONSIVENESS" if status == State.POSSIBLE_UNRESPONSIVE else status
    return f'<span class="badge {cls}">{label}</span>'


# ---------------------------------------------------------------------------
# WebRTC video processor
# ---------------------------------------------------------------------------
try:
    from streamlit_webrtc import WebRtcMode, webrtc_streamer
    from streamlit_webrtc.webrtc import VideoTransformerBase
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

if WEBRTC_AVAILABLE:
    class _DrowsinessProcessor(VideoTransformerBase):
        """
        Runs in a background thread — processes each video frame.
        Accesses session_state via a snapshot taken at construction time
        to avoid cross-thread session_state access issues.
        """

        def __init__(self) -> None:
            self._engine          = _get_engine()
            self._alarm_cooldown  = ALARM_COOLDOWN_SEC
            self._last_alert      = 0.0
            self._lock            = threading.Lock()

        def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            with self._lock:
                img = self._engine.process_bgr_frame(img)
                if self._engine.is_alarming:
                    now = time.time()
                    # Shorter cooldown for POSSIBLE_UNRESPONSIVE
                    cd = self._alarm_cooldown / max(1, self._engine.alarm_priority)
                    if now - self._last_alert >= cd:
                        self._last_alert = now
                        play_alarm_async(DEFAULT_ALARM_PATH)
                else:
                    reset_alarm_gate()
            return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>🚗 Driver Drowsiness Detection System</h1>
        <p>Real-time Eye Aspect Ratio monitoring · OpenCV · Dlib · FYP I — KFUEIT 2026</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — configuration & telemetry
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Detection Settings")
    st.caption("Adjust thresholds; changes take effect on next monitoring session.")

    new_ear = st.slider(
        "EAR threshold",
        min_value=0.15, max_value=0.40,
        value=st.session_state.ear_threshold, step=0.01,
        help="Eye Aspect Ratio below this value is considered 'eye closed'.",
    )
    new_consec = st.slider(
        "Warning frames (consecutive closed)",
        min_value=5, max_value=60,
        value=st.session_state.consec_frames, step=1,
        help="Number of consecutive closed frames to trigger Drowsy Warning.",
    )
    new_esc = st.slider(
        "Alert escalation frames",
        min_value=30, max_value=180,
        value=st.session_state.escalation_frames, step=5,
        help="Frames before escalating to Drowsy Alert.",
    )
    new_unr = st.slider(
        "Unresponsive frames",
        min_value=60, max_value=360,
        value=st.session_state.unresponsive_frames, step=10,
        help="Frames before raising 'Possible Unresponsiveness' warning.",
    )

    settings_changed = (
        new_ear    != st.session_state.ear_threshold      or
        new_consec != st.session_state.consec_frames      or
        new_esc    != st.session_state.escalation_frames  or
        new_unr    != st.session_state.unresponsive_frames
    )
    if settings_changed:
        st.session_state.ear_threshold       = new_ear
        st.session_state.consec_frames       = new_consec
        st.session_state.escalation_frames   = new_esc
        st.session_state.unresponsive_frames = new_unr
        st.session_state.engine              = None   # force recreation

    st.divider()
    st.markdown("## 📊 Live Telemetry")

    engine = _get_engine()
    m: LiveMetrics = engine.metrics
    status_cls = _STATE_BADGE.get(m.status, "badge-noface")
    label = "⚠ POSSIBLE UNRESPONSIVENESS" if m.status == State.POSSIBLE_UNRESPONSIVE else m.status
    st.markdown(
        f'<div style="margin-bottom:0.75rem;">'
        f'Status: <span class="badge {status_cls}">{label}</span></div>',
        unsafe_allow_html=True,
    )
    col_a, col_b = st.columns(2)
    col_a.metric("EAR",     f"{m.ear:.3f}")
    col_b.metric("MAR",     f"{m.mar:.3f}")
    col_a.metric("Blinks",  m.blink_count)
    col_b.metric("Alerts",  m.alert_count)
    col_a.metric("Yawns",   m.yawn_count)
    col_b.metric("Face",    "✅ Yes" if m.face_detected else "❌ No")
    if st.session_state.session_start:
        st.caption(f"Session started: {st.session_state.session_start}")
    st.caption(f"Last frame: {m.last_updated or '—'}")
    st.caption(f"Backend: **{m.backend.upper()}**")

    st.divider()
    st.markdown("## 📖 Detection States")
    state_descriptions = {
        "🟢 Normal":                       "Face visible, eyes open.",
        "🔵 Eyes Closing":                 "Brief closure — possible blink.",
        "🟠 Drowsy — Warning":             "Eyes closed beyond threshold. Alarm triggered.",
        "🔴 Drowsy — Alert":               "Sustained closure. Louder / more frequent alarm.",
        "🚨 Possible Unresponsiveness":    "Prolonged closure. Immediate attention required.",
        "🟩 Recovery":                     "Eyes reopened after alert.",
        "⚫ Face Not Detected":             "Driver cannot be assessed.",
    }
    for state_lbl, desc in state_descriptions.items():
        st.markdown(f"**{state_lbl}** — {desc}")

# ---------------------------------------------------------------------------
# Main content columns
# ---------------------------------------------------------------------------
col_camera, col_info = st.columns([2.2, 1], gap="large")

# ---- Camera column --------------------------------------------------------
with col_camera:
    if not st.session_state.camera_consent:
        st.markdown('<div class="consent-box">', unsafe_allow_html=True)
        st.markdown("### 📷 Camera access required")
        st.markdown(
            "This system uses your **webcam** to monitor eye and facial behaviour in real time, "
            "as required by the FYP specification (FR-01 video capture, FR-07 visualization).\n\n"
            "**No video is recorded or transmitted.**  "
            "All processing happens locally on your machine."
        )
        st.info(
            "After clicking **Allow camera access**, your browser will show its own "
            "permission dialog — click **Allow** there to start detection."
        )
        c1, c2, _ = st.columns([1.2, 1, 1.8])
        with c1:
            if st.button("✅ Allow camera access", type="primary", use_container_width=True):
                st.session_state.camera_consent = True
                st.rerun()
        with c2:
            if st.button("Not now", use_container_width=True):
                st.warning("Detection remains off until camera access is granted.")
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        # ---- Active monitoring session ------------------------------------
        col_stop, col_reset, _ = st.columns([1.2, 1, 2])
        with col_stop:
            if st.button("⏹ Stop & revoke camera", type="secondary", use_container_width=True):
                st.session_state.camera_consent = False
                st.session_state.engine         = None
                st.rerun()
        with col_reset:
            if st.button("🔄 Reset counters", use_container_width=True):
                _get_engine().reset()
                st.success("Counters reset.")

        if not WEBRTC_AVAILABLE:
            st.error(
                "**streamlit-webrtc** is not installed. "
                "Run `pip install streamlit-webrtc` then restart."
            )
        else:
            ctx = webrtc_streamer(
                key="drowsiness-detection",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=_DrowsinessProcessor,
                media_stream_constraints={
                    "video": {
                        "width":      {"ideal": 640},
                        "height":     {"ideal": 480},
                        "facingMode": "user",
                        "frameRate":  {"ideal": 20, "max": 30},
                    },
                    "audio": False,
                },
                async_processing=True,
                rtc_configuration={
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                },
            )

            if ctx and ctx.state.playing:
                st.success(
                    "✅ **Camera active — monitoring in progress.**  "
                    "Press **Stop** on the player to pause."
                )
            elif st.session_state.camera_consent:
                st.info(
                    "Click **START** on the video widget above, then allow the "
                    "browser camera prompt if shown."
                )

        # ---- Live metrics bar (auto-refresh) ----------------------------------
        metrics_ph = st.empty()

        @st.fragment(run_every=1)
        def _refresh_metrics() -> None:
            m2 = _get_engine().metrics
            with metrics_ph.container():
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("EAR",           f"{m2.ear:.3f}")
                c2.metric("MAR",           f"{m2.mar:.3f}")
                c3.metric("Closed frames", m2.closed_frames)
                c4.metric("Blinks",        m2.blink_count)
                c5.metric("Face",          "Yes" if m2.face_detected else "No")

        _refresh_metrics()

# ---- Info / alert history column -----------------------------------------
with col_info:
    st.markdown("### 📋 How it works")
    steps = [
        ("1", "Webcam streams live video (FR-01)"),
        ("2", "Haar / Dlib detects the driver's face (FR-02)"),
        ("3", "68-point landmarks isolate the eyes (FR-03)"),
        ("4", "EAR is computed each frame"),
        ("5", "State machine classifies drowsiness level (FR-04)"),
        ("6", "Audible alarm triggers on warning+ states (FR-05)"),
        ("7", "Overlay shows live status in colour (FR-07)"),
    ]
    for num, text in steps:
        st.markdown(
            f'<div class="how-step">'
            f'<span class="step-num">{num}</span>'
            f'<span style="font-size:0.88rem;color:#334155">{text}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("### 🔔 Alert Log")

    @st.fragment(run_every=2)
    def _refresh_alerts() -> None:
        m3 = _get_engine().metrics
        if not m3.alert_log:
            st.caption("No alerts this session.")
            return
        # Show most recent 10 alerts
        for ev in reversed(m3.alert_log[-10:]):
            cls = _STATE_BADGE.get(ev.state, "badge-noface")
            lbl = (
                "⚠ UNRESPONSIVE" if ev.state == State.POSSIBLE_UNRESPONSIVE
                else ev.state
            )
            st.markdown(
                f'<div class="alert-row">'
                f'<span class="alert-ts">{ev.timestamp}</span>'
                f'<span class="badge {cls} alert-st">{lbl}</span>'
                f'<span style="color:#94a3b8;font-size:0.78rem;">'
                f'EAR {ev.ear}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    _refresh_alerts()

    # ---- Session summary --------------------------------------------------
    st.markdown("### 📈 Session Summary")

    @st.fragment(run_every=5)
    def _session_summary() -> None:
        m4 = _get_engine().metrics
        st.markdown(
            f"""
| Metric | Value |
|--------|-------|
| Session start | {m4.session_start} |
| Blinks counted | {m4.blink_count} |
| Yawns detected | {m4.yawn_count} |
| Alerts triggered | {m4.alert_count} |
| Current EAR | {m4.ear:.3f} |
| Backend | {m4.backend.upper()} |
"""
        )
        if m4.alert_count > 0:
            st.warning(
                f"**{m4.alert_count} drowsiness event(s) recorded this session.**  "
                "Consider taking a break."
            )

    _session_summary()

# ---------------------------------------------------------------------------
# Bottom — installation guide
# ---------------------------------------------------------------------------
st.divider()
with st.expander("🛠 Installation & run instructions"):
    st.markdown(
        f"""
**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Download Dlib landmark model (recommended for best accuracy):**
```
https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
```
Extract and place `shape_predictor_68_face_landmarks.dat` in `models/`.

**3. Start the web dashboard:**
```bash
streamlit run streamlit_app/streamlit_app.py
```

**4. Start the desktop launcher (no browser needed):**
```bash
python main.py
```

**5. Run the classic Dlib script directly:**
```bash
python blinkDetect.py
```

Alarm file: `{DEFAULT_ALARM_PATH.name}` at project root.  
Backend auto-selects Dlib (if model present) or OpenCV Haar cascade.
"""
    )

with st.expander("ℹ️ About this project"):
    st.markdown(
        """
**Driver Drowsiness Detection System — FYP I**  
Department of Data Science & AI · KFUEIT, Rahim Yar Khan · 2026

**Authors:** Muhammad Ashraf Khan Mazari · Muhammad Umair Mukhtar  
**Supervisor:** Dr. Shadab Alam Hashmi

**Technology Stack:** Python · OpenCV · Dlib · SciPy · Streamlit · streamlit-webrtc

**Detection Method:** Eye Aspect Ratio (EAR) with 5-state temporal state machine:
Normal → Eyes Closing → Drowsy Warning → Drowsy Alert → Possible Unresponsiveness → Recovery

> *Note: This system provides a non-intrusive, software-based drowsiness indicator.  
> Eye closure alone cannot medically confirm unconsciousness.  
> The "Possible Unresponsiveness" state indicates sustained eye closure beyond the  
> configured safety threshold and should prompt immediate driver attention.*
"""
    )
