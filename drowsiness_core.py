"""
Driver Drowsiness Detection System — Core Detection Engine (FYP I, KFUEIT 2026).

Authors : Muhammad Ashraf Khan Mazari · Muhammad Umair Mukhtar
Supervisor: Dr. Shadab Alam Hashmi

Detection pipeline:
  Camera frame → Face detection (Haar / Dlib)
               → Eye landmark extraction & EAR calculation (FR-03)
               → Mouth landmark extraction & MAR calculation (yawning)
               → State machine:
                    NORMAL → EYES_CLOSING → DROWSY_WARNING
                    → DROWSY_ALERT → POSSIBLE_UNRESPONSIVE → RECOVERY
               → Visual overlay (FR-07)
               → Audible alarm trigger (FR-05)

Non-Functional Requirements addressed:
  NFR-01  low latency   — Haar downsample, Dlib FACE_DOWNSAMPLE
  NFR-02  low FP rate   — temporal hysteresis on every state transition
  NFR-03  intuitive UI  — colour-coded overlays, clear text labels
  NFR-04  long-duration  — no unbounded file I/O, alarm gate prevents spam
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial import distance as dist

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT      = Path(__file__).resolve().parent
DEFAULT_ALARM_PATH = PROJECT_ROOT / "alarm.wav"
DLIB_MODEL_PATH   = PROJECT_ROOT / "models" / "shape_predictor_68_face_landmarks.dat"
FACE_CASCADE_PATH = PROJECT_ROOT / "models" / "haarcascade_frontalface_default.xml"

# ---------------------------------------------------------------------------
# Configurable thresholds (FR-04, NFR-02)
# ---------------------------------------------------------------------------
EAR_THRESHOLD          = 0.27   # Below this → eye considered closed
MAR_YAWN_THRESHOLD     = 0.55   # Above this → yawning
CONSEC_CLOSED_FRAMES   = 18     # ≈ 0.6 s at 30 fps — triggers DROWSY_WARNING
YAWN_CONSEC_FRAMES     = 12     # Persistent yawn → also triggers warning
ESCALATION_FRAMES      = 60     # ≈ 2 s sustained closure → DROWSY_ALERT
UNRESPONSIVE_FRAMES    = 120    # ≈ 4 s sustained closure → POSSIBLE_UNRESPONSIVE
RECOVERY_OPEN_FRAMES   = 6      # Eyes open for N frames → RECOVERY
ALARM_COOLDOWN_SEC     = 4.0    # Min seconds between successive alarm plays

# Dlib eye / mouth landmark indices (68-point model)
LEFT_EYE_IDX  = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_IDX = [42, 43, 44, 45, 46, 47]
MOUTH_IDX     = [48, 54, 51, 57]   # left, right, upper-inner, lower-inner

# ---------------------------------------------------------------------------
# Detection states
# ---------------------------------------------------------------------------
class State:
    NORMAL               = "Normal"
    EYES_CLOSING         = "Eyes Closing"
    DROWSY_WARNING       = "Drowsy — Warning"
    DROWSY_ALERT         = "Drowsy — Alert"
    POSSIBLE_UNRESPONSIVE = "Possible Unresponsiveness"
    RECOVERY             = "Recovery"
    NO_FACE              = "Face Not Detected"
    INITIALIZING         = "Initializing"

# Colour palette for state overlays  (BGR)
STATE_COLOUR: Dict[str, Tuple[int, int, int]] = {
    State.NORMAL:                (0,  200,  60),
    State.EYES_CLOSING:          (0,  200, 200),
    State.DROWSY_WARNING:        (0,  165, 255),
    State.DROWSY_ALERT:          (0,   60, 255),
    State.POSSIBLE_UNRESPONSIVE: (0,    0, 220),
    State.RECOVERY:              (60, 220, 120),
    State.NO_FACE:               (80,  80,  80),
    State.INITIALIZING:          (80,  80,  80),
}

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _euclidean(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return float(dist.euclidean(p1, p2))


def eye_aspect_ratio(eye: List[Tuple[int, int]]) -> float:
    """Standard 6-point EAR formula."""
    a = _euclidean(eye[1], eye[5])
    b = _euclidean(eye[2], eye[4])
    c = _euclidean(eye[0], eye[3])
    if c <= 1e-6:
        return 0.0
    return (a + b) / (2.0 * c)


def mouth_aspect_ratio(landmarks: List[Tuple[int, int]]) -> float:
    """Vertical / horizontal ratio for the inner lip bounding box."""
    left  = landmarks[MOUTH_IDX[0]]
    right = landmarks[MOUTH_IDX[1]]
    upper = landmarks[MOUTH_IDX[2]]
    lower = landmarks[MOUTH_IDX[3]]
    vertical   = _euclidean(upper, lower)
    horizontal = _euclidean(left,  right)
    if horizontal <= 1e-6:
        return 0.0
    return vertical / horizontal

# ---------------------------------------------------------------------------
# Alert event log entry
# ---------------------------------------------------------------------------
@dataclass
class AlertEvent:
    timestamp:  str
    state:      str
    ear:        float
    mar:        float
    closed_frames: int

# ---------------------------------------------------------------------------
# Live metrics dataclass (shared between engine and UI)
# ---------------------------------------------------------------------------
@dataclass
class LiveMetrics:
    ear:           float = 0.0
    mar:           float = 0.0
    status:        str   = State.INITIALIZING
    blink_count:   int   = 0
    alert_count:   int   = 0
    yawn_count:    int   = 0
    closed_frames: int   = 0
    yawn_frames:   int   = 0
    face_detected: bool  = False
    backend:       str   = "opencv"
    session_start: str   = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))
    last_updated:  Optional[str]        = None
    alert_log:     List[AlertEvent]     = field(default_factory=list)

# ---------------------------------------------------------------------------
# Internal engine state (counters etc.)
# ---------------------------------------------------------------------------
@dataclass
class _EngineState:
    closed_counter:  int  = 0
    open_counter:    int  = 0   # consecutive open frames after closure
    yawn_counter:    int  = 0
    was_closed:      bool = False
    detection_state: str  = State.INITIALIZING

# ---------------------------------------------------------------------------
# OpenCV Haar-cascade backend
# ---------------------------------------------------------------------------
class _OpenCvBackend:
    """
    Fallback when Dlib is unavailable.
    Uses a heuristic 'pseudo-EAR' based on the eye ROI pixel distribution.
    NFR-02: histogram-equalised input for better performance under N-03.
    """

    def __init__(self) -> None:
        face_path = str(FACE_CASCADE_PATH)
        self.face_cascade = cv2.CascadeClassifier(face_path)
        self.eye_cascade  = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load face cascade: {face_path}")

    def close(self) -> None:
        pass

    def _pseudo_ear(self, eye_roi: np.ndarray) -> float:
        """
        Heuristic eye openness from the fraction of dark (pupil/iris) pixels.
        Returns a value in [0, 0.45] that approximates EAR.
        NOTE: This is NOT a true EAR — it is only used when Dlib is absent.
        """
        h, w = eye_roi.shape[:2]
        if h < 4 or w < 4:
            return 0.0
        gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY) if eye_roi.ndim == 3 else eye_roi
        # Histogram-equalise to handle varying illumination (N-03)
        gray = cv2.equalizeHist(gray)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        white = float(cv2.countNonZero(thresh))
        ratio = white / float(h * w)
        return min(0.45, ratio * 2.2)

    def process(
        self, frame: np.ndarray, state: _EngineState
    ) -> Tuple[np.ndarray, float, float, bool]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)   # NFR-03 / N-03
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
        if len(faces) == 0:
            return frame, 0.0, 0.0, False

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 128, 0), 2)
        roi_gray   = gray[y: y + h, x: x + w]
        roi_colour = frame[y: y + h, x: x + w]
        eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 8, minSize=(20, 20))

        ears: List[float] = []
        for (ex, ey, ew, eh) in eyes[:2]:
            cv2.rectangle(roi_colour, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)
            ear_val = self._pseudo_ear(roi_colour[ey: ey + eh, ex: ex + ew])
            ears.append(ear_val)

        ear = float(np.mean(ears)) if ears else 0.0

        # Rough MAR from mouth region pixel statistics
        mouth_y1 = y + int(h * 0.62)
        mouth_y2 = y + int(h * 0.92)
        mouth_x1 = x + int(w * 0.25)
        mouth_x2 = x + int(w * 0.75)
        mouth_roi = gray[mouth_y1:mouth_y2, mouth_x1:mouth_x2]
        mar = 0.0
        if mouth_roi.size > 0:
            mar = min(1.0, float(np.std(mouth_roi)) / 40.0)

        return frame, ear, mar, True

# ---------------------------------------------------------------------------
# Dlib 68-point landmark backend
# ---------------------------------------------------------------------------
class _DlibBackend:
    """
    Precise EAR using dlib's 68-point facial landmark predictor.
    Preferred when the .dat model file is present.
    """
    FACE_DOWNSAMPLE = 1.5   # shrink for speed (NFR-01)

    def __init__(self) -> None:
        import dlib   # type: ignore
        self.detector  = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(str(DLIB_MODEL_PATH))

    def close(self) -> None:
        pass

    def _landmarks(self, gray: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        import dlib   # type: ignore
        small = cv2.resize(
            gray, None,
            fx=1.0 / self.FACE_DOWNSAMPLE,
            fy=1.0 / self.FACE_DOWNSAMPLE,
            interpolation=cv2.INTER_LINEAR,
        )
        rects = self.detector(small, 0)
        if not rects:
            return None
        rect = dlib.rectangle(
            int(rects[0].left()   * self.FACE_DOWNSAMPLE),
            int(rects[0].top()    * self.FACE_DOWNSAMPLE),
            int(rects[0].right()  * self.FACE_DOWNSAMPLE),
            int(rects[0].bottom() * self.FACE_DOWNSAMPLE),
        )
        return [(p.x, p.y) for p in self.predictor(gray, rect).parts()]

    def process(
        self, frame: np.ndarray, state: _EngineState
    ) -> Tuple[np.ndarray, float, float, bool]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)   # N-03 lighting robustness
        landmarks = self._landmarks(gray)
        if not landmarks:
            return frame, 0.0, 0.0, False

        left  = [landmarks[i] for i in LEFT_EYE_IDX]
        right = [landmarks[i] for i in RIGHT_EYE_IDX]
        ear   = (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2.0
        mar   = mouth_aspect_ratio(landmarks)

        # Draw eye landmarks (FR-07)
        for idx in LEFT_EYE_IDX + RIGHT_EYE_IDX:
            cv2.circle(frame, landmarks[idx], 2, (0, 255, 255), -1)

        return frame, ear, mar, True

# ---------------------------------------------------------------------------
# Main drowsiness engine
# ---------------------------------------------------------------------------
@dataclass
class DrowsinessEngine:
    """
    Stateful, frame-by-frame drowsiness detection engine (FR-01 – FR-07).

    State machine transitions (hysteresis prevents oscillation — NFR-02):
      INITIALIZING / NO_FACE
        → NORMAL          (face visible, eyes open)
        → EYES_CLOSING    (EAR < threshold, < CONSEC_CLOSED_FRAMES)
        → DROWSY_WARNING  (≥ CONSEC_CLOSED_FRAMES; alarm triggered)
        → DROWSY_ALERT    (≥ ESCALATION_FRAMES; stronger alarm)
        → POSSIBLE_UNRESPONSIVE (≥ UNRESPONSIVE_FRAMES)
        → RECOVERY        (eyes reopen after warning; reset after RECOVERY_OPEN_FRAMES)
    """

    ear_threshold:        float = EAR_THRESHOLD
    mar_yawn_threshold:   float = MAR_YAWN_THRESHOLD
    consec_closed_frames: int   = CONSEC_CLOSED_FRAMES
    yawn_consec_frames:   int   = YAWN_CONSEC_FRAMES
    escalation_frames:    int   = ESCALATION_FRAMES
    unresponsive_frames:  int   = UNRESPONSIVE_FRAMES
    recovery_open_frames: int   = RECOVERY_OPEN_FRAMES

    metrics: LiveMetrics  = field(default_factory=LiveMetrics)
    _state:  _EngineState = field(default_factory=_EngineState, init=False, repr=False)
    _backend: object      = field(default=None,                  init=False, repr=False)

    def __post_init__(self) -> None:
        self._backend = self._create_backend()
        if isinstance(self._backend, _DlibBackend):
            self.metrics.backend = "dlib"
        else:
            self.metrics.backend = "opencv"

    def _create_backend(self) -> object:
        if DLIB_MODEL_PATH.is_file():
            try:
                return _DlibBackend()
            except (ImportError, Exception):
                pass
        return _OpenCvBackend()

    # ------------------------------------------------------------------
    # State machine logic
    # ------------------------------------------------------------------
    def _update_state(self, face_ok: bool, ear: float, mar: float) -> None:
        st = self._state
        m  = self.metrics

        if not face_ok:
            # Reset counters but preserve blink / alert history
            st.closed_counter = 0
            st.open_counter   = 0
            st.yawn_counter   = 0
            st.was_closed     = False
            st.detection_state = State.NO_FACE
            m.face_detected    = False
            m.closed_frames    = 0
            m.yawn_frames      = 0
            return

        m.face_detected = True
        eye_closed      = ear < self.ear_threshold
        yawning         = mar >= self.mar_yawn_threshold

        # ---- Eye closure counter ----------------------------------------
        if eye_closed:
            st.closed_counter += 1
            st.open_counter    = 0
            if not st.was_closed:
                st.was_closed = True
        else:
            # Count a blink only if the closure was short (genuine blink)
            if st.was_closed and st.closed_counter < self.consec_closed_frames:
                m.blink_count += 1
            st.open_counter += 1
            # Only fully reset closed_counter after RECOVERY_OPEN_FRAMES consecutive
            # open frames to prevent oscillation (hysteresis — NFR-02)
            if st.open_counter >= self.recovery_open_frames:
                st.closed_counter = 0
                st.was_closed     = False

        # ---- Yawn counter -----------------------------------------------
        if yawning:
            st.yawn_counter += 1
        else:
            st.yawn_counter = 0

        # ---- State transitions ------------------------------------------
        prev_state = st.detection_state
        cc = st.closed_counter
        yc = st.yawn_counter

        if cc == 0 and yc < self.yawn_consec_frames:
            if st.detection_state == State.RECOVERY:
                # Stay in RECOVERY for a moment, then return to NORMAL
                if st.open_counter >= self.recovery_open_frames:
                    st.detection_state = State.NORMAL
            else:
                st.detection_state = State.NORMAL
        elif 0 < cc < self.consec_closed_frames and yc < self.yawn_consec_frames:
            st.detection_state = State.EYES_CLOSING
        elif cc >= self.unresponsive_frames:
            st.detection_state = State.POSSIBLE_UNRESPONSIVE
        elif cc >= self.escalation_frames:
            st.detection_state = State.DROWSY_ALERT
        elif cc >= self.consec_closed_frames or yc >= self.yawn_consec_frames:
            st.detection_state = State.DROWSY_WARNING
        elif prev_state in (State.DROWSY_WARNING, State.DROWSY_ALERT,
                            State.POSSIBLE_UNRESPONSIVE) and not eye_closed:
            st.detection_state = State.RECOVERY

        # ---- Yawn counter contribution to yawn_count -------------------
        if yc == self.yawn_consec_frames:   # edge trigger
            m.yawn_count += 1

        # ---- Persist to metrics ----------------------------------------
        m.closed_frames    = st.closed_counter
        m.yawn_frames      = st.yawn_counter
        m.status           = st.detection_state

        # ---- Log alert events ------------------------------------------
        if st.detection_state in (
            State.DROWSY_WARNING, State.DROWSY_ALERT, State.POSSIBLE_UNRESPONSIVE
        ) and (
            not m.alert_log
            or m.alert_log[-1].state != st.detection_state
        ):
            m.alert_log.append(AlertEvent(
                timestamp=datetime.now().strftime("%H:%M:%S"),
                state=st.detection_state,
                ear=round(ear, 3),
                mar=round(mar, 3),
                closed_frames=st.closed_counter,
            ))
            m.alert_count += 1

    # ------------------------------------------------------------------
    # Visual overlay (FR-07)
    # ------------------------------------------------------------------
    def _draw_overlay(self, frame: np.ndarray, ear: float, mar: float) -> np.ndarray:
        h, w = frame.shape[:2]
        st_name  = self._state.detection_state
        colour   = STATE_COLOUR.get(st_name, (200, 200, 200))
        m        = self.metrics

        # Colour-coded status border for high-severity states
        if st_name in (State.DROWSY_WARNING, State.DROWSY_ALERT,
                       State.POSSIBLE_UNRESPONSIVE):
            border_thickness = 10 if st_name == State.POSSIBLE_UNRESPONSIVE else 6
            cv2.rectangle(frame, (0, 0), (w, h), colour, border_thickness)

        # Status banner
        banner_h = 50
        overlay  = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (15, 23, 42), -1)
        frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

        status_label = st_name
        if st_name == State.POSSIBLE_UNRESPONSIVE:
            status_label = "⚠ POSSIBLE UNRESPONSIVENESS"

        cv2.putText(
            frame, status_label,
            (10, 34), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, colour, 2, cv2.LINE_AA,
        )

        # EAR / MAR / blink telemetry (right-aligned)
        info_line = f"EAR:{ear:.2f}  MAR:{mar:.2f}  Blinks:{m.blink_count}  Alerts:{m.alert_count}"
        (tw, _), _ = cv2.getTextSize(info_line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(
            frame, info_line,
            (w - tw - 10, 34), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (180, 200, 220), 1, cv2.LINE_AA,
        )

        # Yawning indicator
        if self._state.yawn_counter >= self.yawn_consec_frames:
            cv2.putText(
                frame, "YAWNING DETECTED",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 165, 255), 2, cv2.LINE_AA,
            )

        # Backend label (bottom right, small)
        backend_label = f"Backend: {m.backend.upper()}"
        (bw, _), _ = cv2.getTextSize(backend_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(
            frame, backend_label,
            (w - bw - 6, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (80, 100, 120), 1, cv2.LINE_AA,
        )

        return frame

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_bgr_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process one BGR camera frame.
        Returns annotated BGR frame.  Updates self.metrics in-place.
        Thread-safe: metrics are updated atomically for individual fields.
        """
        frame, ear, mar, face_ok = self._backend.process(frame, self._state)

        self.metrics.ear = round(ear, 4)
        self.metrics.mar = round(mar, 4)
        self._update_state(face_ok, ear, mar)

        frame = self._draw_overlay(frame, ear, mar)
        self.metrics.last_updated = datetime.now().strftime("%H:%M:%S")
        return frame

    def reset(self) -> None:
        """Reset detection counters while preserving cumulative session totals."""
        self._state = _EngineState()
        self.metrics.closed_frames = 0
        self.metrics.yawn_frames   = 0
        self.metrics.status        = State.NORMAL

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()

    @property
    def is_alarming(self) -> bool:
        return self._state.detection_state in (
            State.DROWSY_WARNING,
            State.DROWSY_ALERT,
            State.POSSIBLE_UNRESPONSIVE,
        )

    @property
    def alarm_priority(self) -> int:
        """Higher number = higher urgency (for variable alarm cooldown)."""
        priority_map = {
            State.DROWSY_WARNING:        1,
            State.DROWSY_ALERT:          2,
            State.POSSIBLE_UNRESPONSIVE: 3,
        }
        return priority_map.get(self._state.detection_state, 0)


# ---------------------------------------------------------------------------
# Alarm helpers (FR-05)
# ---------------------------------------------------------------------------
_alarm_lock    = threading.Lock()
_alarm_playing = False


def play_alarm_async(alarm_path: Optional[Path] = None) -> None:
    """
    Play the alarm WAV once in a background daemon thread.
    Prevents duplicate plays via a mutex gate (NFR-04).
    On Windows uses winsound (no extra dependency).
    On other platforms falls back to playsound.
    """
    global _alarm_playing
    path = Path(alarm_path) if alarm_path else DEFAULT_ALARM_PATH
    if not path.is_file():
        return

    with _alarm_lock:
        if _alarm_playing:
            return
        _alarm_playing = True

    def _play() -> None:
        global _alarm_playing
        try:
            import sys
            if sys.platform == "win32":
                import winsound
                winsound.PlaySound(str(path), winsound.SND_FILENAME)   # blocking
            else:
                from playsound import playsound   # type: ignore
                playsound(str(path))
        except Exception:
            pass
        finally:
            with _alarm_lock:
                _alarm_playing = False

    threading.Thread(target=_play, daemon=True).start()


def reset_alarm_gate() -> None:
    """Allow the next alarm to play (call when detection state clears)."""
    global _alarm_playing
    with _alarm_lock:
        _alarm_playing = False
