import os
import subprocess
from pathlib import Path
import tempfile

import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "model"
MODEL_FILENAME = "FINAL_animal_detector.keras"
MODEL_PATH = MODEL_DIR / MODEL_FILENAME

SAVED_MODEL_SIBLING = MODEL_DIR / "animal_detector_best_saved_model"

IMG_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

CLASSES = [
    "Bear",
    "Deer",
    "Duck",
    "Fox",
    "Parrot",
    "Rabbit",
    "Raccoon",
    "Red Panda",
    "Squirrel",
    "Tiger"
]

CARNIVORES = {"Bear", "Fox", "Tiger"}

st.set_page_config(
    page_title="Animal Detection",
    layout="wide"
)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model not found: {MODEL_PATH}")
        st.stop()

    if not SAVED_MODEL_SIBLING.exists():
        st.error(
            "The '.keras' file needs its SavedModel sibling folder "
            f"next to it: {SAVED_MODEL_SIBLING} was not found. "
            "Download 'animal_detector_best_saved_model' from Kaggle "
            "and place it in the model/ folder alongside the .keras file."
        )
        st.stop()

    original_cwd = os.getcwd()

    try:
        os.chdir(MODEL_DIR)
        loaded_model = keras.models.load_model(
            MODEL_FILENAME,
            compile=False,
            safe_mode=False
        )
    finally:
        os.chdir(original_cwd)

    return loaded_model

model = load_model()

def get_prediction_output(prediction):
    if isinstance(prediction, dict):
        prediction = next(iter(prediction.values()))

    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]

    if hasattr(prediction, "numpy"):
        prediction = prediction.numpy()

    return np.asarray(prediction)

def preprocess_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))
    resized = resized.astype(np.float32) / 255.0
    return np.expand_dims(resized, axis=0)

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)
    intersection = intersection_width * intersection_height

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union <= 0:
        return 0

    return intersection / union

def classwise_nms(detections):
    final_detections = []

    for class_id in range(len(CLASSES)):
        class_detections = [
            detection
            for detection in detections
            if detection["class_id"] == class_id
        ]

        class_detections.sort(
            key=lambda x: x["confidence"],
            reverse=True
        )

        while class_detections:
            best = class_detections.pop(0)
            final_detections.append(best)

            remaining = []

            for detection in class_detections:
                iou = calculate_iou(
                    best["box"],
                    detection["box"]
                )

                if iou < IOU_THRESHOLD:
                    remaining.append(detection)

            class_detections = remaining

    return final_detections

def decode_predictions(prediction, original_width, original_height):
    prediction = get_prediction_output(prediction)

    if prediction.ndim == 3:
        prediction = prediction[0]

    if prediction.shape[0] < prediction.shape[1]:
        prediction = prediction.T

    if prediction.shape[1] < 4 + len(CLASSES):
        return []

    boxes = prediction[:, :4]
    class_scores = prediction[:, 4:4 + len(CLASSES)]

    class_ids = np.argmax(class_scores, axis=1)
    confidences = np.max(class_scores, axis=1)

    detections = []

    scale_x = original_width / IMG_SIZE
    scale_y = original_height / IMG_SIZE

    for box, class_id, confidence in zip(
        boxes,
        class_ids,
        confidences
    ):
        confidence = float(confidence)

        if confidence < CONF_THRESHOLD:
            continue

        x_center, y_center, width, height = box

        x1 = int((x_center - width / 2) * scale_x)
        y1 = int((y_center - height / 2) * scale_y)
        x2 = int((x_center + width / 2) * scale_x)
        y2 = int((y_center + height / 2) * scale_y)

        x1 = max(0, min(x1, original_width - 1))
        y1 = max(0, min(y1, original_height - 1))
        x2 = max(0, min(x2, original_width - 1))
        y2 = max(0, min(y2, original_height - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        detections.append({
            "box": [x1, y1, x2, y2],
            "class_id": int(class_id),
            "class_name": CLASSES[int(class_id)],
            "confidence": confidence
        })

    return classwise_nms(detections)

def detect_animals(image):
    height, width = image.shape[:2]

    input_tensor = preprocess_image(image)
    prediction = model(
        input_tensor,
        training=False
    )

    detections = decode_predictions(
        prediction,
        width,
        height
    )

    result = image.copy()
    carnivore_count = 0

    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        class_name = detection["class_name"]
        confidence = detection["confidence"]

        if class_name in CARNIVORES:
            color = (0, 0, 255)
            carnivore_count += 1
        else:
            color = (0, 255, 0)

        cv2.rectangle(
            result,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        label = f"{class_name} {confidence:.2f}"

        text_y = max(y1 - 10, 25)

        cv2.putText(
            result,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    return result, detections, carnivore_count

def process_video(input_path):
    capture = cv2.VideoCapture(input_path)

    if not capture.isOpened():
        return None, 0, 0, {}

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    output_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    output_path = output_file.name
    output_file.close()

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    frame_count = 0
    max_carnivores = 0
    animal_counts = {}

    while True:
        success, frame = capture.read()

        if not success:
            break

        result, detections, carnivore_count = detect_animals(frame)

        writer.write(result)

        for detection in detections:
            class_name = detection["class_name"]
            animal_counts[class_name] = animal_counts.get(class_name, 0) + 1

        frame_count += 1
        max_carnivores = max(
            max_carnivores,
            carnivore_count
        )

    capture.release()
    writer.release()

    browser_friendly_path = reencode_for_browser(output_path)

    return browser_friendly_path, frame_count, max_carnivores, animal_counts

def reencode_for_browser(input_path):
    """
    Re-encodes the 'mp4v' output to H.264 using ffmpeg so it plays
    in the browser via st.video(). Falls back to the original file
    if ffmpeg is unavailable or the conversion fails.
    """
    output_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )
    output_path = output_file.name
    output_file.close()

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", input_path,
                "-vcodec", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                output_path
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return output_path
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        st.warning(
            "ffmpeg re-encode failed, showing original video "
            f"(it may not play in the browser). Error: {error}"
        )
        return input_path

st.title("Animal Detection System")
st.write("Detect animals using the custom-trained animal detection model.")

mode = st.radio(
    "Select Input",
    ["Image", "Video"],
    horizontal=True
)

if mode == "Image":

    uploaded_file = st.file_uploader(
        "Upload Animal Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        file_bytes = np.asarray(
            bytearray(uploaded_file.read()),
            dtype=np.uint8
        )

        image = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        if image is None:
            st.error("Unable to read the image.")
        else:

            result, detections, carnivore_count = detect_animals(image)

            original_rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            result_rgb = cv2.cvtColor(
                result,
                cv2.COLOR_BGR2RGB
            )

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Image")
                st.image(
                    original_rgb,
                    use_container_width=True
                )

            with col2:
                st.subheader("Detection Result")
                st.image(
                    result_rgb,
                    use_container_width=True
                )

            st.divider()

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Total Animals",
                    len(detections)
                )

            with col2:
                st.metric(
                    "Carnivorous Animals",
                    carnivore_count
                )

            if carnivore_count > 0:
                st.error(
                    f"{carnivore_count} carnivorous animal(s) detected."
                )
                st.toast(
                    f"{carnivore_count} carnivorous animal(s) detected."
                )
            else:
                st.success(
                    "No carnivorous animals detected."
                )

            if detections:

                st.subheader("Detected Animals")

                for detection in detections:

                    class_name = detection["class_name"]
                    confidence = detection["confidence"]

                    if class_name in CARNIVORES:
                        st.error(
                            f"{class_name} — {confidence:.2%}"
                        )
                    else:
                        st.success(
                            f"{class_name} — {confidence:.2%}"
                        )

            else:
                st.warning("No animals detected.")

if mode == "Video":

    uploaded_video = st.file_uploader(
        "Upload Animal Video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:

        if st.button(
            "Start Detection",
            type="primary"
        ):

            input_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=os.path.splitext(
                    uploaded_video.name
                )[1]
            )

            input_file.write(
                uploaded_video.read()
            )

            input_file.close()

            with st.spinner(
                "Processing video..."
            ):

                output_path, frame_count, max_carnivores, animal_counts = process_video(
                    input_file.name
                )

            if output_path is None:
                st.error(
                    "Unable to process the video."
                )
            else:

                st.subheader("Detection Result")

                with open(
                    output_path,
                    "rb"
                ) as video_file:

                    video_bytes = video_file.read()

                st.video(video_bytes)

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Processed Frames",
                        frame_count
                    )

                with col2:
                    st.metric(
                        "Maximum Carnivores in One Frame",
                        max_carnivores
                    )

                if max_carnivores > 0:
                    st.error(
                        f"Carnivorous animals detected. Maximum in one frame: {max_carnivores}"
                    )
                    st.toast(
                        f"Carnivorous animal detected: {max_carnivores}"
                    )
                else:
                    st.success(
                        "No carnivorous animals detected."
                    )

                if animal_counts:

                    st.subheader("Animals Detected in Video")

                    sorted_animals = sorted(
                        animal_counts.items(),
                        key=lambda item: item[1],
                        reverse=True
                    )

                    for class_name, count in sorted_animals:

                        if class_name in CARNIVORES:
                            st.error(
                                f"{class_name} — seen in {count} frame(s)"
                            )
                        else:
                            st.success(
                                f"{class_name} — seen in {count} frame(s)"
                            )

                else:
                    st.warning("No animals detected in the video.")