import os
import sys
import cv2
import numpy as np
import tensorflow as tf


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")

NATIONALITY_MODEL_PATH = os.path.join(MODEL_DIR, "nationality_detector_final.keras")
AGE_MODEL_PATH = os.path.join(MODEL_DIR, "age_detector_final.keras")
DRESS_MODEL_PATH = os.path.join(MODEL_DIR, "dress_color_detector_final.keras")
EMOTION_WEIGHTS_PATH = os.path.join(MODEL_DIR, "model_weights1.h5")
CASCADE_PATH = os.path.join(MODEL_DIR, "haarcascade_frontalface_default.xml")

NATIONALITY_LABELS = {0: "United States", 1: "African", 2: "Other", 3: "Indian"}
DRESS_LABELS = {0:"black",1:"blue",2:"brown",3:"green",4:"pink",5:"red",6:"silver",7:"white",8:"yellow"}
EMOTIONS_LIST = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]


def build_emotion_model():
    inputs = tf.keras.Input(shape=(48, 48, 1), name="input_1")

    x = tf.keras.layers.Conv2D(64, (3,3), padding="same", activation="linear", name="conv2d")(inputs)
    x = tf.keras.layers.BatchNormalization(axis=3, name="batch_normalization")(x)
    x = tf.keras.layers.Activation("relu", name="activation")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), name="max_pooling2d")(x)
    x = tf.keras.layers.Dropout(0.25, name="dropout")(x)

    x = tf.keras.layers.Conv2D(128, (5,5), padding="same", activation="linear", name="conv2d_1")(x)
    x = tf.keras.layers.BatchNormalization(axis=3, name="batch_normalization_1")(x)
    x = tf.keras.layers.Activation("relu", name="activation_1")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), name="max_pooling2d_1")(x)
    x = tf.keras.layers.Dropout(0.25, name="dropout_1")(x)

    x = tf.keras.layers.Conv2D(512, (3,3), padding="same", activation="linear", name="conv2d_2")(x)
    x = tf.keras.layers.BatchNormalization(axis=3, name="batch_normalization_2")(x)
    x = tf.keras.layers.Activation("relu", name="activation_2")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), name="max_pooling2d_2")(x)
    x = tf.keras.layers.Dropout(0.25, name="dropout_2")(x)

    x = tf.keras.layers.Conv2D(512, (3,3), padding="same", activation="linear", name="conv2d_3")(x)
    x = tf.keras.layers.BatchNormalization(axis=3, name="batch_normalization_3")(x)
    x = tf.keras.layers.Activation("relu", name="activation_3")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), name="max_pooling2d_3")(x)
    x = tf.keras.layers.Dropout(0.25, name="dropout_3")(x)

    x = tf.keras.layers.Flatten(name="flatten")(x)

    x = tf.keras.layers.Dense(256, activation="linear", name="dense")(x)
    x = tf.keras.layers.BatchNormalization(axis=1, name="batch_normalization_4")(x)
    x = tf.keras.layers.Activation("relu", name="activation_4")(x)
    x = tf.keras.layers.Dropout(0.25, name="dropout_4")(x)

    x = tf.keras.layers.Dense(512, activation="linear", name="dense_1")(x)
    x = tf.keras.layers.BatchNormalization(axis=1, name="batch_normalization_5")(x)
    x = tf.keras.layers.Activation("relu", name="activation_5")(x)
    x = tf.keras.layers.Dropout(0.25, name="dropout_5")(x)

    outputs = tf.keras.layers.Dense(7, activation="softmax", name="dense_2")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="model")


class Predictor:

    def __init__(self):
        required_files = [
            NATIONALITY_MODEL_PATH, AGE_MODEL_PATH, DRESS_MODEL_PATH,
            EMOTION_WEIGHTS_PATH, CASCADE_PATH
        ]
        for path in required_files:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Required file not found:\n{path}")

        self.nationality_model = tf.keras.models.load_model(NATIONALITY_MODEL_PATH, compile=False)
        self.age_model = tf.keras.models.load_model(AGE_MODEL_PATH, compile=False)
        self.dress_model = tf.keras.models.load_model(DRESS_MODEL_PATH, compile=False)

        self.emotion_model = build_emotion_model()
        self.emotion_model.load_weights(EMOTION_WEIGHTS_PATH)

        self.face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        if self.face_cascade.empty():
            raise RuntimeError("Unable to load Haar Cascade.")

        print()
        print("Models loaded successfully.")
        print("Nationality output:", self.nationality_model.output_shape)
        print("Age output:", self.age_model.output_shape)
        print("Dress output:", self.dress_model.output_shape)
        print()

    def load_image(self, file_path):
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError("Unable to read image.")
        return image

    def detect_largest_face(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
        return image[y:y + h, x:x + w]

    def prepare_color_image(self, image, size):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, size)
        image = image.astype(np.float32)
        image = np.expand_dims(image, axis=0)
        return image

    def predict_nationality(self, image):
        face = self.detect_largest_face(image)
        if face is None:
            face = image

        input_image = self.prepare_color_image(face, (160, 160))
        prediction = self.nationality_model.predict(input_image, verbose=0)[0]

        print()
        print("Nationality probabilities:")
        for i, probability in enumerate(prediction):
            print(f"{NATIONALITY_LABELS[i]}: {probability * 100:.2f}%")

        index = int(np.argmax(prediction))
        return NATIONALITY_LABELS[index], float(prediction[index])

    def predict_age(self, image):
        face = self.detect_largest_face(image)
        if face is None:
            face = image

        input_image = self.prepare_color_image(face, (160, 160))
        prediction = self.age_model.predict(input_image, verbose=0)
        age = float(np.squeeze(prediction))
        age = max(0, min(116, age))
        return round(age)

    def predict_emotion(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            return "Unable to detect", 0.0

        x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (48, 48))

        input_image = face[np.newaxis, :, :, np.newaxis]

        prediction = self.emotion_model.predict(input_image, verbose=0)[0]
        index = int(np.argmax(prediction))
        return EMOTIONS_LIST[index], float(prediction[index])
    
    def predict_dress_color(self, image):
        input_image = self.prepare_color_image(image, (128, 128))
        prediction = self.dress_model.predict(input_image, verbose=0)[0]

        print()
        print("Dress colour probabilities:")
        for i, probability in enumerate(prediction):
            print(f"{DRESS_LABELS[i]}: {probability * 100:.2f}%")

        index = int(np.argmax(prediction))
        return DRESS_LABELS[index], float(prediction[index])

    def predict(self, file_path):
        image = self.load_image(file_path)

        nationality, nationality_confidence = self.predict_nationality(image)
        emotion, emotion_confidence = self.predict_emotion(image)

        result = {
            "nationality": nationality,
            "nationality_confidence": nationality_confidence,
            "emotion": emotion,
            "emotion_confidence": emotion_confidence
        }

        if nationality == "Indian":
            result["age"] = self.predict_age(image)
            dress_color, dress_confidence = self.predict_dress_color(image)
            result["dress_color"] = dress_color
            result["dress_confidence"] = dress_confidence

        elif nationality == "United States":
            result["age"] = self.predict_age(image)

        elif nationality == "African":
            dress_color, dress_confidence = self.predict_dress_color(image)
            result["dress_color"] = dress_color
            result["dress_confidence"] = dress_confidence

        return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predictor.py image_path")
        sys.exit(1)

    image_path = sys.argv[1]
    predictor = Predictor()
    result = predictor.predict(image_path)

    print()
    print("========== FINAL RESULT ==========")
    print("Nationality:", result["nationality"])
    print("Nationality Confidence:", f"{result['nationality_confidence'] * 100:.2f}%")
    print("Emotion:", result["emotion"])
    print("Emotion Confidence:", f"{result['emotion_confidence'] * 100:.2f}%")

    if "age" in result:
        print("Age:", result["age"])

    if "dress_color" in result:
        print("Dress Colour:", result["dress_color"])
        print("Dress Confidence:", f"{result['dress_confidence'] * 100:.2f}%")

    print("=================================")