import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(
    BASE_DIR,
    "model",
    "vehicle_color_model.h5"
)

class_names_path = os.path.join(
    BASE_DIR,
    "tasks",
    "class_names.json"
)

model = keras.models.load_model(model_path)

with open(class_names_path, "r") as file:
    class_names = json.load(file)


def predict_color(car_image):
    image = tf.image.resize(car_image, (224, 224))
    image = image.numpy()
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image, verbose=0)

    predicted_index = np.argmax(prediction[0])

    predicted_color = class_names[predicted_index]
    confidence = prediction[0][predicted_index]

    return predicted_color, confidence