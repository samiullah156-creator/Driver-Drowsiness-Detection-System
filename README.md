# Driver Drowsiness Detection Key Features
- Real-time monitoring of driver's eyes using a camera
- Fatigue and drowsiness detection algorithm
- Timely alerts to prevent the driver from falling asleep
- Non-intrusive and easy to set up
- Progressive Web App (PWA) support - Install like a native app!
- Offline functionality with caching
- Fast loading with optimized performance on System


## Project Overview
The Driver Drowsiness Detection System is a non-intrusive solution designed to monitor and detect signs of fatigue in drivers. By analyzing eye states through a camera feed, the system can identify early symptoms of drowsiness and issue timely warnings, helping to prevent accidents caused by driver fatigue. This project aims to enhance road safety, especially for those driving long distances who may not recognize their own drowsiness in time.


## Key Features
- Real-time monitoring of driver’s eyes using a camera
- Fatigue and drowsiness detection algorithm
- Timely alerts to prevent the driver from falling asleep
- Non-intrusive and easy to set up

## Technologies Used
- Python
- OpenCV
- Dlib or Mediapipe (for facial landmark detection)
- NumPy, Pandas

  
 ## Optional: create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


## Setup Instructions
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download the model:**

    The dlib facial landmark model is required for blink detection. Download it from the link below, unzip it, and place the `shape_predictor_68_face_landmarks.dat` file inside the `/models` directory.
    * **Download Link**: [shape_predictor_68_face_landmarks.dat](https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

4. **Run the application:**
   ```bash
   python main.py
   ```
   (Replace `main.py` with the actual entry point if different.)

## Usage
- Ensure your webcam is connected.
- Run the application as shown above.
- The system will start monitoring your eyes and alert you if signs of drowsiness are detected.


## Model file

The 68-point face landmark model is not stored in this repo (it is about 100 MB). Download it from [dlib.net](https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2), extract it and place `shape_predictor_68_face_landmarks.dat` in `models/`.
