"""
Driver Drowsiness Detection System — Test Suite.

Authors : Muhammad Ashraf Khan Mazari · Muhammad Umair Mukhtar
Supervisor: Dr. Shadab Alam Hashmi
University: KFUEIT, Rahim Yar Khan — FYP I (2026)

Run:
    python -m pytest tests/test_system.py -v
    python tests/test_system.py          # standalone

Coverage:
    TC-01  DrowsinessEngine instantiates without error
    TC-02  OpenCV backend processes a synthetic frame
    TC-03  Normal state — eyes open throughout
    TC-04  Brief blink does not trigger DROWSY_WARNING
    TC-05  Prolonged closure triggers DROWSY_WARNING
    TC-06  Extended closure escalates to DROWSY_ALERT
    TC-07  Very long closure escalates to POSSIBLE_UNRESPONSIVE
    TC-08  Eyes reopening after warning → RECOVERY
    TC-09  Full recovery returns to NORMAL
    TC-10  Face loss → NO_FACE state (not DROWSY)
    TC-11  Yawning triggers DROWSY_WARNING
    TC-12  Blink count increments only for short closures
    TC-13  Alert log records new states
    TC-14  Alert log does not duplicate the same state consecutively
    TC-15  reset() clears counters without clearing session totals
    TC-16  EAR / MAR helper functions return correct values
    TC-17  play_alarm_async skips if alarm file is missing (no crash)
    TC-18  reset_alarm_gate allows subsequent alarm play
    TC-19  is_alarming property reflects state correctly
    TC-20  alarm_priority returns increasing values per severity
"""

import sys
import os
from pathlib import Path
import unittest

import numpy as np

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from drowsiness_core import (
    DrowsinessEngine,
    LiveMetrics,
    State,
    AlertEvent,
    eye_aspect_ratio,
    mouth_aspect_ratio,
    play_alarm_async,
    reset_alarm_gate,
    CONSEC_CLOSED_FRAMES,
    ESCALATION_FRAMES,
    UNRESPONSIVE_FRAMES,
    RECOVERY_OPEN_FRAMES,
    MOUTH_IDX,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(h: int = 240, w: int = 320) -> np.ndarray:
    """Synthetic black BGR frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _pump_engine(engine: DrowsinessEngine, n_frames: int, frame: np.ndarray) -> None:
    """Feed the engine `n_frames` identical frames (no actual face visible)."""
    for _ in range(n_frames):
        engine.process_bgr_frame(frame.copy())


def _inject_state(engine: DrowsinessEngine, closed_counter: int = 0,
                  open_counter: int = 0, was_closed: bool = False,
                  detection_state: str = State.NORMAL) -> None:
    """Directly set the internal engine state for unit testing."""
    engine._state.closed_counter  = closed_counter
    engine._state.open_counter    = open_counter
    engine._state.was_closed      = was_closed
    engine._state.detection_state = detection_state


def _call_update(engine: DrowsinessEngine, face_ok: bool, ear: float, mar: float = 0.0) -> None:
    """Call the private state-update method directly."""
    engine.metrics.ear = ear
    engine.metrics.mar = mar
    engine._update_state(face_ok, ear, mar)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEngineInstantiation(unittest.TestCase):
    """TC-01: Engine should instantiate cleanly."""

    def test_engine_creates_metrics(self):
        engine = DrowsinessEngine()
        self.assertIsInstance(engine.metrics, LiveMetrics)

    def test_initial_status_is_initializing(self):
        engine = DrowsinessEngine()
        self.assertEqual(engine.metrics.status, State.INITIALIZING)

    def test_backend_set(self):
        engine = DrowsinessEngine()
        self.assertIn(engine.metrics.backend, ("dlib", "opencv"))


class TestFrameProcessing(unittest.TestCase):
    """TC-02: OpenCV backend should process a synthetic frame without error."""

    def test_process_black_frame_returns_ndarray(self):
        engine = DrowsinessEngine()
        frame  = _make_frame()
        result = engine.process_bgr_frame(frame)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[2], 3)  # BGR

    def test_no_face_detected_on_black_frame(self):
        engine = DrowsinessEngine()
        _pump_engine(engine, 3, _make_frame())
        self.assertFalse(engine.metrics.face_detected)


class TestNormalState(unittest.TestCase):
    """TC-03: Eyes open → NORMAL."""

    def test_eyes_open_gives_normal(self):
        engine = DrowsinessEngine(ear_threshold=0.27, consec_closed_frames=18)
        # Simulate face visible, eyes open (EAR > threshold)
        _inject_state(engine, detection_state=State.NORMAL)
        for _ in range(10):
            _call_update(engine, face_ok=True, ear=0.35)
        self.assertEqual(engine.metrics.status, State.NORMAL)


class TestBriefBlink(unittest.TestCase):
    """TC-04: Brief blink (< consec_closed_frames) must NOT trigger warning."""

    def test_blink_under_limit_no_warning(self):
        engine = DrowsinessEngine(ear_threshold=0.27, consec_closed_frames=18)
        _inject_state(engine, detection_state=State.NORMAL)

        blink_duration = 5   # fewer frames than the threshold
        for _ in range(blink_duration):
            _call_update(engine, face_ok=True, ear=0.10)  # eyes closed

        # Open eyes again
        for _ in range(RECOVERY_OPEN_FRAMES + 1):
            _call_update(engine, face_ok=True, ear=0.35)

        self.assertNotIn(engine.metrics.status, (
            State.DROWSY_WARNING, State.DROWSY_ALERT, State.POSSIBLE_UNRESPONSIVE
        ))

    def test_brief_blink_increments_blink_count(self):
        engine = DrowsinessEngine(ear_threshold=0.27, consec_closed_frames=18)
        # Pre-warm: set was_closed = False and detection_state = NORMAL
        _inject_state(engine, detection_state=State.NORMAL, was_closed=False)
        initial_blinks = engine.metrics.blink_count

        # Short closure (fewer than consec_closed_frames)
        for _ in range(5):
            _call_update(engine, face_ok=True, ear=0.10)

        # Reopen — the reopen triggers blink count (was_closed=True, counter < limit)
        for _ in range(RECOVERY_OPEN_FRAMES + 1):
            _call_update(engine, face_ok=True, ear=0.35)

        # Exactly one blink should have been counted
        self.assertEqual(engine.metrics.blink_count, initial_blinks + 1)


class TestProlongedClosure(unittest.TestCase):
    """TC-05: Prolonged closure triggers DROWSY_WARNING."""

    def test_prolonged_closure_triggers_warning(self):
        threshold = 18
        engine = DrowsinessEngine(ear_threshold=0.27, consec_closed_frames=threshold)
        _inject_state(engine, detection_state=State.NORMAL)

        for _ in range(threshold + 1):
            _call_update(engine, face_ok=True, ear=0.10)

        self.assertEqual(engine.metrics.status, State.DROWSY_WARNING)


class TestEscalation(unittest.TestCase):
    """TC-06 & TC-07: Escalation to DROWSY_ALERT and POSSIBLE_UNRESPONSIVE."""

    def test_escalation_to_alert(self):
        engine = DrowsinessEngine(
            ear_threshold=0.27,
            consec_closed_frames=18,
            escalation_frames=40,
            unresponsive_frames=100,
        )
        _inject_state(engine, detection_state=State.NORMAL)
        for _ in range(45):
            _call_update(engine, face_ok=True, ear=0.10)
        self.assertEqual(engine.metrics.status, State.DROWSY_ALERT)

    def test_escalation_to_unresponsive(self):
        engine = DrowsinessEngine(
            ear_threshold=0.27,
            consec_closed_frames=18,
            escalation_frames=40,
            unresponsive_frames=80,
        )
        _inject_state(engine, detection_state=State.NORMAL)
        for _ in range(85):
            _call_update(engine, face_ok=True, ear=0.10)
        self.assertEqual(engine.metrics.status, State.POSSIBLE_UNRESPONSIVE)


class TestRecovery(unittest.TestCase):
    """TC-08 & TC-09: Recovery state after eyes reopen."""

    def test_eyes_reopen_after_warning_gives_recovery(self):
        engine = DrowsinessEngine(ear_threshold=0.27, consec_closed_frames=18)
        # Drive into warning
        for _ in range(25):
            _call_update(engine, face_ok=True, ear=0.10)
        self.assertEqual(engine.metrics.status, State.DROWSY_WARNING)

        # Reopen eyes — need RECOVERY_OPEN_FRAMES consecutive open frames
        # to clear closed_counter and enter RECOVERY state
        for _ in range(RECOVERY_OPEN_FRAMES):
            _call_update(engine, face_ok=True, ear=0.35)
        self.assertEqual(engine.metrics.status, State.RECOVERY)

    def test_sustained_open_returns_to_normal(self):
        engine = DrowsinessEngine(ear_threshold=0.27, consec_closed_frames=18)
        # Drive into warning
        for _ in range(25):
            _call_update(engine, face_ok=True, ear=0.10)
        # Full recovery
        for _ in range(RECOVERY_OPEN_FRAMES + 5):
            _call_update(engine, face_ok=True, ear=0.35)
        self.assertEqual(engine.metrics.status, State.NORMAL)


class TestFaceLoss(unittest.TestCase):
    """TC-10: Face loss → NO_FACE, NOT DROWSY."""

    def test_no_face_gives_no_face_state(self):
        engine = DrowsinessEngine()
        # First call to leave INITIALIZING state
        _call_update(engine, face_ok=True, ear=0.35)
        # Now face disappears
        _call_update(engine, face_ok=False, ear=0.0)
        self.assertEqual(engine.metrics.status, State.NO_FACE)

    def test_no_face_does_not_set_drowsy(self):
        engine = DrowsinessEngine(ear_threshold=0.27, consec_closed_frames=5)
        # Pump many face-absent frames
        for _ in range(30):
            _call_update(engine, face_ok=False, ear=0.0)
        self.assertNotIn(engine.metrics.status, (
            State.DROWSY_WARNING, State.DROWSY_ALERT, State.POSSIBLE_UNRESPONSIVE
        ))

    def test_no_face_resets_closed_counter(self):
        engine = DrowsinessEngine(ear_threshold=0.27, consec_closed_frames=18)
        for _ in range(10):
            _call_update(engine, face_ok=True, ear=0.10)
        _call_update(engine, face_ok=False, ear=0.0)
        self.assertEqual(engine._state.closed_counter, 0)


class TestYawning(unittest.TestCase):
    """TC-11: Yawning detection triggers warning."""

    def test_yawn_triggers_warning(self):
        yawn_frames = 12
        engine = DrowsinessEngine(
            mar_yawn_threshold=0.55,
            yawn_consec_frames=yawn_frames,
            ear_threshold=0.27,
        )
        _inject_state(engine, detection_state=State.NORMAL)
        for _ in range(yawn_frames + 1):
            _call_update(engine, face_ok=True, ear=0.35, mar=0.70)
        self.assertEqual(engine.metrics.status, State.DROWSY_WARNING)


class TestBlinkCount(unittest.TestCase):
    """TC-12: Blink count only increments for short closures."""

    def test_long_closure_does_not_count_as_blink(self):
        engine = DrowsinessEngine(ear_threshold=0.27, consec_closed_frames=18)
        _inject_state(engine, detection_state=State.NORMAL)
        initial = engine.metrics.blink_count

        # Long closure (25 frames)
        for _ in range(25):
            _call_update(engine, face_ok=True, ear=0.10)
        # Reopen
        for _ in range(RECOVERY_OPEN_FRAMES + 1):
            _call_update(engine, face_ok=True, ear=0.35)

        # Should be 0 new blinks (long closure is not a blink)
        self.assertEqual(engine.metrics.blink_count, initial)


class TestAlertLog(unittest.TestCase):
    """TC-13 & TC-14: Alert log records events and avoids duplicates."""

    def test_alert_logged_on_warning(self):
        engine = DrowsinessEngine(ear_threshold=0.27, consec_closed_frames=18)
        _inject_state(engine, detection_state=State.NORMAL)
        for _ in range(20):
            _call_update(engine, face_ok=True, ear=0.10)
        self.assertGreater(len(engine.metrics.alert_log), 0)
        self.assertEqual(engine.metrics.alert_log[0].state, State.DROWSY_WARNING)

    def test_same_state_not_logged_twice_in_a_row(self):
        engine = DrowsinessEngine(ear_threshold=0.27, consec_closed_frames=18)
        _inject_state(engine, detection_state=State.NORMAL)
        for _ in range(30):
            _call_update(engine, face_ok=True, ear=0.10)
        log_len = len(engine.metrics.alert_log)
        for _ in range(10):
            _call_update(engine, face_ok=True, ear=0.10)
        self.assertEqual(len(engine.metrics.alert_log), log_len)


class TestReset(unittest.TestCase):
    """TC-15: reset() clears counters but preserves cumulative totals."""

    def test_reset_clears_closed_frames(self):
        engine = DrowsinessEngine(ear_threshold=0.27, consec_closed_frames=18)
        for _ in range(20):
            _call_update(engine, face_ok=True, ear=0.10)
        engine.reset()
        self.assertEqual(engine.metrics.closed_frames, 0)
        self.assertEqual(engine._state.closed_counter, 0)


class TestHelpers(unittest.TestCase):
    """TC-16: EAR / MAR helper functions."""

    def test_ear_fully_open(self):
        # When all 6 eye landmark points are at the same location (degenerate closed eye),
        # all distances are 0, but c=0 too — the function returns 0.0 via the guard.
        # Alternatively: when upper/lower points coincide → vertical distances = 0,
        # but horizontal can be non-zero → EAR = 0.
        # Use: p1=p5 and p2=p4 (no vertical distance) with c > 0
        # eye[0]=(0,0), eye[3]=(10,0) → c=10
        # eye[1]=(2,0), eye[5]=(8,0) → a=6  BUT these are horizontal too!
        # Actually EAR=(a+b)/(2c); for a closed eye b and a are SMALL; let's test
        # the actual closed-eye scenario: all 6 points nearly collinear → low EAR
        eye_almost_closed = [(0, 0), (3, 0), (6, 0), (9, 0), (6, 0), (3, 0)]
        # c = dist((0,0),(9,0)) = 9
        # a = dist((3,0),(3,0)) = 0
        # b = dist((6,0),(6,0)) = 0
        # EAR = 0/18 = 0
        self.assertAlmostEqual(eye_aspect_ratio(eye_almost_closed), 0.0, places=3)

    def test_ear_wide_open(self):
        # Simulate wide-open eye with vertical spread
        eye = [(0, 0), (1, 2), (2, 2), (3, 0), (2, -2), (1, -2)]
        val = eye_aspect_ratio(eye)
        self.assertGreater(val, 0.0)

    def test_ear_zero_width_returns_zero(self):
        # Degenerate case: all points at same x
        eye = [(5, 5)] * 6
        self.assertEqual(eye_aspect_ratio(eye), 0.0)

    def test_mouth_aspect_ratio_closed(self):
        # Upper == lower lip → MAR = 0
        pts = [(0, 0)] * 68
        pts[48] = (0,  0)   # left
        pts[54] = (10, 0)   # right
        pts[51] = (5,  0)   # upper (same as lower)
        pts[57] = (5,  0)   # lower
        self.assertAlmostEqual(mouth_aspect_ratio(pts), 0.0, places=3)

    def test_mouth_aspect_ratio_open(self):
        pts = [(0, 0)] * 68
        pts[48] = (0,  0)   # left
        pts[54] = (10, 0)   # right  → horizontal = 10
        pts[51] = (5,  3)   # upper
        pts[57] = (5,  7)   # lower  → vertical   = 4
        mar = mouth_aspect_ratio(pts)
        self.assertAlmostEqual(mar, 4.0 / 10.0, places=3)


class TestAlarmHelpers(unittest.TestCase):
    """TC-17 & TC-18: Alarm helpers are safe and gateable."""

    def test_play_alarm_missing_file_no_crash(self):
        """play_alarm_async with a nonexistent path must not raise."""
        try:
            play_alarm_async(Path("/nonexistent/path/alarm.wav"))
        except Exception as exc:
            self.fail(f"play_alarm_async raised unexpectedly: {exc}")

    def test_reset_alarm_gate(self):
        """reset_alarm_gate should not raise."""
        try:
            reset_alarm_gate()
        except Exception as exc:
            self.fail(f"reset_alarm_gate raised unexpectedly: {exc}")


class TestIsAlarming(unittest.TestCase):
    """TC-19 & TC-20: is_alarming and alarm_priority properties."""

    def test_not_alarming_in_normal(self):
        engine = DrowsinessEngine()
        _inject_state(engine, detection_state=State.NORMAL)
        engine.metrics.status = State.NORMAL
        self.assertFalse(engine.is_alarming)

    def test_alarming_in_warning(self):
        engine = DrowsinessEngine(ear_threshold=0.27, consec_closed_frames=18)
        _inject_state(engine, detection_state=State.DROWSY_WARNING)
        self.assertTrue(engine.is_alarming)

    def test_priority_increases_with_severity(self):
        engine = DrowsinessEngine()
        p = {}
        for s in (State.DROWSY_WARNING, State.DROWSY_ALERT, State.POSSIBLE_UNRESPONSIVE):
            _inject_state(engine, detection_state=s)
            p[s] = engine.alarm_priority
        self.assertLess(p[State.DROWSY_WARNING], p[State.DROWSY_ALERT])
        self.assertLess(p[State.DROWSY_ALERT],   p[State.POSSIBLE_UNRESPONSIVE])

    def test_priority_zero_for_non_alarm_states(self):
        engine = DrowsinessEngine()
        for s in (State.NORMAL, State.EYES_CLOSING, State.RECOVERY, State.NO_FACE):
            _inject_state(engine, detection_state=s)
            self.assertEqual(engine.alarm_priority, 0)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
