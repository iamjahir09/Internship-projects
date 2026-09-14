import os

import numpy as np
import tensorflow as tf
import cv2


class DrowsinessDetector:

    def __init__(self, model_path=None):

        src_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        project_dir = os.path.dirname(src_dir)

        if model_path is None:

            model_path = os.path.join(
                project_dir,
                "model",
                "drowsiness_model.keras"
            )

        model_path = os.path.abspath(model_path)

        if not os.path.exists(model_path):

            raise FileNotFoundError(
                f"Drowsiness model not found:\n{model_path}"
            )

        self.model = tf.keras.models.load_model(
            model_path
        )

        self.class_names = [
            "Closed",
            "Open",
            "no_yawn",
            "yawn"
        ]

        self.drowsiness_map = {
            "Closed": "Sleeping",
            "Open": "Awake",
            "no_yawn": "Awake",
            "yawn": "Sleeping"
        }

    def predict(self, face_image):

        if face_image is None or face_image.size == 0:
            return None

        image = cv2.resize(
            face_image,
            (224, 224)
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = image.astype(
            np.float32
        ) / 255.0

        image = np.expand_dims(
            image,
            axis=0
        )

        predictions = self.model.predict(
            image,
            verbose=0
        )[0]

        class_index = np.argmax(
            predictions
        )

        predicted_class = (
            self.class_names[class_index]
        )

        confidence = float(
            predictions[class_index] * 100
        )

        state = self.drowsiness_map[
            predicted_class
        ]

        return {
            "class": predicted_class,
            "state": state,
            "confidence": confidence
        }