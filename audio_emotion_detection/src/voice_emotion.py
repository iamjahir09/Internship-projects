import os
import json
import librosa
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "voice_emotion_model.keras"
)

LABELS_PATH = os.path.join(
    BASE_DIR,
    "model",
    "voice_emotion_labels.json"
)

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    labels = json.load(f)


def extract_mfcc(file_path, max_pad_len=130):
    audio, sr = librosa.load(file_path, sr=22050)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(
            mfcc,
            pad_width=((0, 0), (0, pad_width)),
            mode="constant"
        )
    else:
        mfcc = mfcc[:, :max_pad_len]

    return mfcc


def check_female_voice(file_path):
    audio, sr = librosa.load(file_path, sr=22050)

    if len(audio) < sr * 0.5:
        return False

    f0 = librosa.yin(
        audio,
        fmin=75,
        fmax=300,
        sr=sr
    )

    valid_f0 = f0[np.isfinite(f0)]

    if len(valid_f0) == 0:
        return False

    median_f0 = float(np.median(valid_f0))

    return median_f0 >= 140


def predict_emotion(file_path):
    if not check_female_voice(file_path):
        return "female_only", 0.0

    mfcc = extract_mfcc(file_path)
    mfcc = mfcc[np.newaxis, ..., np.newaxis]

    prediction = model.predict(
        mfcc,
        verbose=0
    )

    predicted_index = int(np.argmax(prediction[0]))
    emotion = labels[str(predicted_index)]
    confidence = float(prediction[0][predicted_index])

    return emotion, confidence