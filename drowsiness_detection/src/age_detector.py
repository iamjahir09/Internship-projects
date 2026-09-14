import os

import numpy as np
import tensorflow as tf
import cv2


class AgeDetector:

    def __init__(self, model_path=None):

        src_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        project_dir = os.path.dirname(src_dir)

        if model_path is None:

            model_path = os.path.join(
                project_dir,
                "model",
                "age_detector_final.keras"
            )

        model_path = os.path.abspath(model_path)

        if not os.path.exists(model_path):

            raise FileNotFoundError(
                f"Age model not found:\n{model_path}"
            )

        self.model = tf.keras.models.load_model(
            model_path
        )

    def predict(self, face_image):

        if face_image is None or face_image.size == 0:
            return None

        image = cv2.resize(
            face_image,
            (160, 160)
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = image.astype(
            np.float32
        )

        image = np.expand_dims(
            image,
            axis=0
        )

        prediction = self.model.predict(
            image,
            verbose=0
        )

        age = float(
            prediction[0][0]
        )

        age = max(
            0,
            min(100, age)
        )

        return round(age)