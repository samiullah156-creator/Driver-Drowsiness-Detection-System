# -*- coding: utf-8 -*-
"""
Driver Drowsiness Detection System — Classic Dlib + EAR blink/drowsiness script.

Authors : Muhammad Ashraf Khan Mazari · Muhammad Umair Mukhtar
Supervisor: Dr. Shadab Alam Hashmi
University: KFUEIT, Rahim Yar Khan — FYP I (2026)

Requirements:
  • shape_predictor_68_face_landmarks.dat in models/
  • dlib, opencv-python, scipy, playsound

Run:  python blinkDetect.py
Keys: R = reset counters | Q / ESC = quit
"""

import os
import sys
import cv2
import time
import queue
import threading
from datetime import datetime
from pathlib import Path

import dlib  # type: ignore
import numpy as np
from scipy.spatial import distance as dist

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = Path(__file__).resolve().parent
MODEL_PATH  = BASE_DIR / "models" / "shape_predictor_68_face_landmarks.dat"
SOUND_PATH  = BASE_DIR / "alarm.wav"

# ---------------------------------------------------------------------------
# Detection thresholds
# ---------------------------------------------------------------------------
EAR_THRESHOLD         = 0.27   # Below → eye closed
BLINK_TIME_SEC        = 0.15   # Normal blink duration (~150 ms)
DROWSY_TIME_SEC       = 1.5    # Prolonged closure → DROWSY_WARNING (~1.5 s)
ESCALATION_TIME_SEC   = 3.0    # Sustained closure → DROWSY_ALERT (~3 s)

FACE_DOWNSAMPLE_RATIO = 1.5
RESIZE_HEIGHT         = 460
GAMMA                 = 1.5
CALIBRATION_FRAMES    = 100   # Frames used to measure per-frame processing time

# Landmark indices
LEFT_EYE_IDX  = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_IDX = [42, 43, 44, 45, 46, 47]

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class DetectionState:
    NORMAL               = "Normal"
    EYES_CLOSING         = "Eyes Closing"
    DROWSY_WARNING       = "Drowsy — Warning"
    DROWSY_ALERT         = "Drowsy — Alert"
    POSSIBLE_UNRESPONSIVE = "Possible Unresponsiveness"
    RECOVERY             = "Recovery"
    NO_FACE              = "Face Not Detected"

# ---------------------------------------------------------------------------
# Gamma look-up table for brightness correction
# ---------------------------------------------------------------------------
_inv_gamma = 1.0 / GAMMA
_lut = np.array(
    [((i / 255.0) ** _inv_gamma) * 255 for i in range(256)], dtype="uint8"
)


def gamma_correction(image: np.ndarray) -> np.ndarray:
    return cv2.LUT(image, _lut)


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)

# ---------------------------------------------------------------------------
# EAR calculation (genuine 6-point formula)
# ---------------------------------------------------------------------------
def eye_aspect_ratio(eye_pts: list) -> float:
    """Standard Eye Aspect Ratio from 6 (x, y) landmark points."""
    A = dist.euclidean(eye_pts[1], eye_pts[5])
    B = dist.euclidean(eye_pts[2], eye_pts[4])
    C = dist.euclidean(eye_pts[0], eye_pts[3])
    if C <= 1e-6:
        return 0.0
    return (A + B) / (2.0 * C)

# ---------------------------------------------------------------------------
# Audio helper
# ---------------------------------------------------------------------------
_sound_queue = queue.Queue()


def _sound_worker(path: str) -> None:
    """Background thread that loops the alarm while queue is empty."""
    while True:
        stop = _sound_queue.get()   # blocks until signal
        if stop:
            return
        try:
            import playsound  # type: ignore
            playsound.playsound(path)
        except Exception as e:
            print(f"[Audio] Error: {e}")
        _sound_queue.task_done()


# ---------------------------------------------------------------------------
# Session tracker
# ---------------------------------------------------------------------------
class SessionTracker:
    def __init__(self) -> None:
        self.start_time: datetime   = datetime.now()
        self.ear_values: list       = []
        self.alerts: list           = []
        self.blink_count: int       = 0
        self._active: bool          = True

    def record_ear(self, ear: float) -> None:
        if self._active:
            self.ear_values.append({"value": round(ear, 4), "ts": datetime.now().isoformat()})

    def record_alert(self, state: str) -> None:
        if self._active:
            self.alerts.append({"state": state, "ts": datetime.now().isoformat()})
            print(f"[Alert #{len(self.alerts)}] {state} — {datetime.now().strftime('%H:%M:%S')}")

    def end(self) -> dict:
        if not self._active:
            return {}
        self._active = False
        end_time  = datetime.now()
        duration  = (end_time - self.start_time).total_seconds() / 60.0
        avg_ear   = (
            sum(e["value"] for e in self.ear_values) / len(self.ear_values)
            if self.ear_values else 0.0
        )
        summary = {
            "duration_min": round(duration, 2),
            "ear_readings":  len(self.ear_values),
            "avg_ear":       round(avg_ear, 4),
            "alerts":        len(self.alerts),
            "blinks":        self.blink_count,
        }
        print("\n=== Session Summary ===")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        return summary

# ---------------------------------------------------------------------------
# Landmark getter (operates on a pre-processed gray image)
# ---------------------------------------------------------------------------

def get_landmarks(
    gray_image: np.ndarray,
    detector: dlib.fhog_object_detector,
    predictor: dlib.shape_predictor,
) -> list:
    """
    Return list of (x, y) tuples for the 68 landmarks, or empty list if no face.
    Downsamples for speed (NFR-01).
    """
    small = cv2.resize(
        gray_image, None,
        fx=1.0 / FACE_DOWNSAMPLE_RATIO,
        fy=1.0 / FACE_DOWNSAMPLE_RATIO,
        interpolation=cv2.INTER_LINEAR,
    )
    rects = detector(small, 0)
    if len(rects) == 0:
        return []

    rect = dlib.rectangle(
        int(rects[0].left()   * FACE_DOWNSAMPLE_RATIO),
        int(rects[0].top()    * FACE_DOWNSAMPLE_RATIO),
        int(rects[0].right()  * FACE_DOWNSAMPLE_RATIO),
        int(rects[0].bottom() * FACE_DOWNSAMPLE_RATIO),
    )
    return [(p.x, p.y) for p in predictor(gray_image, rect).parts()]

# ---------------------------------------------------------------------------
# Main detection loop
# ---------------------------------------------------------------------------

def main() -> None:
    # ---- Model check -------------------------------------------------------
    if not MODEL_PATH.is_file():
        print(f"[ERROR] Landmark model not found at:\n  {MODEL_PATH}")
        print("Download from: https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("Extract and place the .dat file in the models/ folder.")
        sys.exit(1)

    # ---- Dlib setup --------------------------------------------------------
    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(MODEL_PATH))

    # ---- Camera ------------------------------------------------------------
    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW if sys.platform == "win32" else 0)
    if not capture.isOpened():
        capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("[ERROR] Could not open webcam.")
        sys.exit(1)

    # ---- Calibration (measure per-frame processing time) ------------------
    print("Calibration in progress — please face the camera…")
    total_time   = 0.0
    valid_frames = 0
    frame        = None

    while valid_frames < CALIBRATION_FRAMES:
        ret, frame = capture.read()
        if not ret or frame is None:
            print("[Warning] Dropped frame during calibration — retrying…")
            continue

        height, width = frame.shape[:2]
        scale = np.float32(height) / RESIZE_HEIGHT
        frame = cv2.resize(
            frame, None, fx=1 / scale, fy=1 / scale,
            interpolation=cv2.INTER_LINEAR,
        )

        t          = time.time()
        adjusted   = histogram_equalization(frame)
        landmarks  = get_landmarks(adjusted, detector, predictor)
        elapsed    = time.time() - t

        if len(landmarks) == 0:
            cv2.putText(
                frame,
                "No face — check lighting or position",
                (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA,
            )
        else:
            valid_frames += 1
            total_time   += elapsed

        cv2.imshow("Calibration — Blink Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:   # ESC during calibration exits
            capture.release()
            cv2.destroyAllWindows()
            return

    sec_per_frame      = total_time / max(valid_frames, 1)
    drowsy_limit       = int(DROWSY_TIME_SEC      / sec_per_frame)
    escalation_limit   = int(ESCALATION_TIME_SEC  / sec_per_frame)
    false_blink_limit  = int(BLINK_TIME_SEC       / sec_per_frame)

    print(f"Calibration complete — SPF: {sec_per_frame * 1000:.1f} ms")
    print(f"  False-blink limit: {false_blink_limit} frames")
    print(f"  Drowsy limit:      {drowsy_limit} frames")
    print(f"  Escalation limit:  {escalation_limit} frames")

    # ---- Session start -----------------------------------------------------
    session = SessionTracker()

    # ---- Detection state ---------------------------------------------------
    state_counter     = 0   # consecutive closed frames
    recovery_counter  = 0   # consecutive open frames after closure
    was_closed        = False
    blink_count       = 0
    detection_state   = DetectionState.NORMAL
    alarm_on          = False
    last_alert_state  = None

    # ---- Audio thread -------------------------------------------------------
    if SOUND_PATH.is_file():
        _sound_thread = threading.Thread(
            target=_sound_worker, args=(str(SOUND_PATH),), daemon=True
        )
        _sound_thread.start()
    else:
        print(f"[Warning] Alarm file not found: {SOUND_PATH}")

    # ---- Colour map per state -----------------------------------------------
    STATE_COLOUR = {
        DetectionState.NORMAL:               (0,  200,  60),
        DetectionState.EYES_CLOSING:         (0,  200, 200),
        DetectionState.DROWSY_WARNING:       (0,  165, 255),
        DetectionState.DROWSY_ALERT:         (0,   60, 255),
        DetectionState.POSSIBLE_UNRESPONSIVE:(0,    0, 200),
        DetectionState.RECOVERY:             (60, 220, 120),
        DetectionState.NO_FACE:              (80,  80,  80),
    }

    print("Detection running — press R to reset, Q or ESC to quit.\n")

    while True:
        try:
            ret, frame = capture.read()
            if not ret or frame is None:
                print("[Warning] Frame read failed — continuing…")
                continue

            # ---- Resize ----------------------------------------------------------------
            height, width = frame.shape[:2]
            scale = np.float32(height) / RESIZE_HEIGHT
            frame = cv2.resize(
                frame, None, fx=1 / scale, fy=1 / scale,
                interpolation=cv2.INTER_LINEAR,
            )

            adjusted  = histogram_equalization(frame)
            landmarks = get_landmarks(adjusted, detector, predictor)

            if len(landmarks) == 0:
                detection_state = DetectionState.NO_FACE
                state_counter   = 0
                recovery_counter = 0
                alarm_on        = False
                cv2.putText(
                    frame,
                    "No face — check lighting or camera position",
                    (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.55, (80, 80, 80), 1, cv2.LINE_AA,
                )
            else:
                # ---- Extract EAR -------------------------------------------------------
                left_eye  = [landmarks[i] for i in LEFT_EYE_IDX]
                right_eye = [landmarks[i] for i in RIGHT_EYE_IDX]
                ear       = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

                session.record_ear(ear)

                # ---- Draw landmarks ----------------------------------------------------
                for i in LEFT_EYE_IDX + RIGHT_EYE_IDX:
                    cv2.circle(frame, landmarks[i], 1, (0, 0, 255), -1, cv2.LINE_AA)

                # ---- Update counters ---------------------------------------------------
                if ear < EAR_THRESHOLD:
                    state_counter  += 1
                    recovery_counter = 0
                    if not was_closed:
                        was_closed = True
                else:
                    if was_closed and state_counter < false_blink_limit:
                        blink_count += 1
                        session.blink_count += 1
                    recovery_counter += 1
                    # Hysteresis: only reset after 6 consecutive open frames
                    if recovery_counter >= 6:
                        state_counter = 0
                        was_closed    = False

                # ---- State machine -----------------------------------------------------
                if state_counter == 0:
                    if detection_state in (
                        DetectionState.DROWSY_WARNING,
                        DetectionState.DROWSY_ALERT,
                        DetectionState.POSSIBLE_UNRESPONSIVE,
                    ):
                        detection_state = DetectionState.RECOVERY
                    elif detection_state == DetectionState.RECOVERY and recovery_counter >= 6:
                        detection_state = DetectionState.NORMAL
                    else:
                        detection_state = DetectionState.NORMAL
                elif 0 < state_counter < false_blink_limit:
                    detection_state = DetectionState.EYES_CLOSING
                elif state_counter >= escalation_limit:
                    detection_state = DetectionState.POSSIBLE_UNRESPONSIVE
                elif state_counter >= drowsy_limit * 2:
                    detection_state = DetectionState.DROWSY_ALERT
                elif state_counter >= drowsy_limit:
                    detection_state = DetectionState.DROWSY_WARNING

                # ---- Alarm logic -------------------------------------------------------
                if detection_state in (
                    DetectionState.DROWSY_WARNING,
                    DetectionState.DROWSY_ALERT,
                    DetectionState.POSSIBLE_UNRESPONSIVE,
                ):
                    if not alarm_on:
                        alarm_on = True
                        _sound_queue.put(False)   # signal to play
                    if detection_state != last_alert_state:
                        session.record_alert(detection_state)
                        last_alert_state = detection_state
                else:
                    alarm_on         = False
                    last_alert_state = None

                # ---- Display EAR, blink count ------------------------------------------
                colour = STATE_COLOUR.get(detection_state, (200, 200, 200))
                cv2.putText(
                    frame,
                    f"EAR: {ear:.3f}  Blinks: {blink_count}  Frames: {state_counter}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2, cv2.LINE_AA,
                )
                if detection_state == DetectionState.POSSIBLE_UNRESPONSIVE:
                    cv2.putText(
                        frame, "!!! POSSIBLE UNRESPONSIVENESS !!!",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.85, colour, 3, cv2.LINE_AA,
                    )
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), colour, 8)
                elif detection_state in (DetectionState.DROWSY_ALERT, DetectionState.DROWSY_WARNING):
                    cv2.putText(
                        frame, f"! {detection_state.upper()} !",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.85, colour, 2, cv2.LINE_AA,
                    )
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), colour, 5)
                elif detection_state == DetectionState.RECOVERY:
                    cv2.putText(
                        frame, "Recovery — Eyes Reopened",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2, cv2.LINE_AA,
                    )

            cv2.imshow("Blink Detection — Driver Drowsiness System", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R")):
                state_counter    = 0
                recovery_counter = 0
                was_closed       = False
                blink_count      = 0
                detection_state  = DetectionState.NORMAL
                alarm_on         = False
                last_alert_state = None
                print("[Reset] Counters cleared.")

        except Exception as exc:
            print(f"[Error] Frame processing exception: {exc}")

    # ---- Cleanup -----------------------------------------------------------
    if SOUND_PATH.is_file():
        _sound_queue.put(True)   # stop worker thread

    session.end()
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
