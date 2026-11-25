<!-- styled:v2 -->
<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1d1726,50:167dfe,100:9533fa&height=180&section=header&text=Driver%20Drowsiness%20Detection%20System&fontSize=31&fontColor=ffffff&fontAlignY=36&desc=Python%20%E2%80%A2%20OpenCV%20%E2%80%A2%20dlib&descAlignY=58&descSize=16&animation=fadeIn" width="100%" alt="Driver Drowsiness Detection System" />

**Real-time computer vision that watches a driver's eyes and mouth through a webcam and sounds an alarm before fatigue turns into an accident.**

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV" />
  <img src="https://img.shields.io/badge/dlib-008000?style=for-the-badge" alt="dlib" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/PWA-5A0FC8?style=for-the-badge&logo=pwa&logoColor=white" alt="PWA" />
</p>

<p>
  <a href="#overview">Overview</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#key-features">Key features</a> ·
  <a href="#tech-stack">Tech stack</a> ·
  <a href="#project-structure">Project structure</a> ·
  <a href="#getting-started">Getting started</a>
</p>

</div>

---

## Overview

Drivers on long trips often don't notice their own drowsiness in time. This system tracks facial landmarks on every camera frame, measures how open the eyes are (eye aspect ratio, EAR) and whether the driver is yawning (mouth aspect ratio, MAR), and raises an audible warning when the eyes stay closed for too long. It runs as a desktop app or as a Streamlit web app that can be installed like a native app (PWA).

## How it works

```mermaid
flowchart LR
    A([Webcam frame]) --> B[Face + 68 landmarks<br/>dlib / Haar fallback]
    B --> C[EAR and MAR<br/>per frame]
    C --> D{Eyes closed<br/>18+ frames<br/>or yawning?}
    D -->|yes| E([Alarm + on-screen<br/>DROWSY warning])
    D -->|no| A

    classDef box fill:#1d1726,stroke:#167dfe,color:#e5e7eb
    classDef ends fill:#2e1065,stroke:#9533fa,color:#ffffff
    class B,C,D box
    class A,E ends
```

| Setting | Value | Meaning |
|:--|:--|:--|
| `EAR_THRESHOLD` | 0.27 | Below this, the eye counts as closed |
| `CONSEC_CLOSED_FRAMES` | 18 | About 0.6 s at 30 fps before a drowsiness warning |
| `MAR_YAWN_THRESHOLD` | 0.55 | Above this, the mouth counts as yawning |
| `YAWN_CONSEC_FRAMES` | 12 | A sustained yawn also triggers the warning |

Thresholds live in `drowsiness_core.py` and can be tuned per driver and camera.

## Key features

- Real-time eye tracking with the standard 6-point EAR formula
- Yawn detection from mouth landmarks
- Audible alarm and colour-coded on-screen overlays
- Falls back to Haar cascades if dlib is not installed
- Lane detection module (`lanedetection.py`)
- Streamlit web dashboard with live camera through WebRTC
- Installable PWA with a service worker and offline page
- Session analysis script and automated tests

## Tech stack

| Area | Tools |
|:--|:--|
| Computer vision | OpenCV, dlib (68-point landmarks), SciPy, NumPy |
| Web app | Streamlit, streamlit-webrtc, PyAV |
| PWA | Web manifest, service worker, generated icons (Pillow) |
| Alarm | winsound on Windows, playsound on Linux and macOS |

## Project structure

```
├── main.py                 # Desktop entry point
├── drowsiness_core.py      # Landmarks, EAR/MAR and alert logic
├── blinkDetect.py          # Blink detection
├── lanedetection.py        # Lane detection module
├── Sesssion_analysis.py    # Session statistics
├── streamlit_app/          # Web dashboard and PWA (manifest, sw.js, icons)
├── models/                 # Place the dlib landmark model here
└── tests/                  # Automated tests
```

## Getting started

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

Download the 68-point landmark model from [dlib.net](https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) (about 100 MB, not stored in this repo), extract it and put `shape_predictor_68_face_landmarks.dat` in `models/`.

```bash
python main.py                                   # desktop app
cd streamlit_app && streamlit run streamlit_app_pwa.py   # web app / PWA on http://localhost:8501
```

If dlib fails to install on Windows, run `pip install cmake` first. See [DEPLOYMENT.md](DEPLOYMENT.md) for Streamlit Cloud and other hosting options.

---

<div align="center">

**Built by [Ch Muhammad Samiullah](https://github.com/samiullah156-creator)** · AI Automation Engineer and Full-Stack Developer at [Lexvaxa](https://lexvaxa.com)

<a href="https://github.com/samiullah156-creator"><img src="https://img.shields.io/badge/More%20projects-181717?style=for-the-badge&logo=github&logoColor=white" alt="More projects" /></a>
<a href="https://lexvaxa.com/contact"><img src="https://img.shields.io/badge/Work%20with%20me-167dfe?style=for-the-badge&logo=googlecalendar&logoColor=white" alt="Work with me" /></a>
<a href="https://www.linkedin.com/in/sami-lexvaxa/"><img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" /></a>

<sub>⭐ Star this repo if you found it useful</sub>

</div>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:9533fa,50:167dfe,100:1d1726&height=110&section=footer" width="100%" alt="" />
