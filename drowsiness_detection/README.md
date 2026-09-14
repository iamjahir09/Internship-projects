# Drowsiness Detection

This project detects drowsiness and face-related conditions using a camera feed and trained models.

## What is included
- dataset/: training data
- model/: Keras and YOLO model files
- src/gui.py: main user interface
- src/drowsiness.py, face_detector.py, person_detector.py: detection logic

## Run
```bash
python src/gui.py
```

## Notes
- The system checks whether a person appears drowsy or inattentive.
- It uses face detection and eye/alert tracking logic.
