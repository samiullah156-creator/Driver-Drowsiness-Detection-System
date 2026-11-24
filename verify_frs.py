# verify_frs.py
"""Simple sanity‑check script for functional requirements FR‑01 – FR‑07.
It does **not** require a webcam – it just verifies that the expected
functions/classes are importable and that the required dependencies are
present.  Run it with:

    python verify_frs.py

The script will exit with status 0 if all checks pass, otherwise it prints
which FR is missing.
"""
import importlib
import sys

# Mapping of FR IDs to the expected import path / attribute
FR_MAP = {
    "FR-01": ("cv2", None),                # video capture module
    "FR-02": ("drowsiness_core", "DrowsinessEngine"),  # uses dlib internally
    "FR-03": ("drowsiness_core", "DrowsinessEngine"),  # eye detection inside engine
    "FR-04": ("drowsiness_core", "DrowsinessEngine"),  # drowsiness detection logic
    "FR-05": ("drowsiness_core", "play_alarm_async"),   # alarm function
    "FR-06": ("drowsiness_core", "DrowsinessEngine"),  # model processing pipeline
    "FR-07": ("main", "run_desktop_detection"),         # visualization (Tkinter GUI)
}

missing = []
for fr, (module_name, attr) in FR_MAP.items():
    try:
        mod = importlib.import_module(module_name)
        if attr:
            getattr(mod, attr)
    except Exception as e:
        missing.append((fr, f"{module_name}{'.' + attr if attr else ''}: {e}"))

if not missing:
    print("All functional requirements (FR-01 to FR-07) are present and importable.")
    sys.exit(0)
else:
    print("Missing or broken functional requirements:")
    for fr, reason in missing:
        print(f" - {fr}: {reason}")
    sys.exit(1)
