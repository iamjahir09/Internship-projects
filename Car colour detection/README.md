# Car Colour Detection

This project identifies the colour of a car from an image or camera feed using a trained classifier.

## What is included
- dataset/: training, validation, and test data
- model/: saved model files
- src/detector.py: detection logic
- src/color_classifier.py: colour classification logic
- src/main.py: main program entry point

## Run
```bash
python src/main.py
```

## Notes
- The model predicts car colours such as common vehicle colours.
- You can use the system on images or live detection pipelines.
