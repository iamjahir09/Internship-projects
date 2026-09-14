import streamlit as st
import datetime
import json
import os
import cv2
import numpy as np

from tensorflow.keras.models import model_from_json
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

JSON_PATH = os.path.join(
    BASE_DIR,
    "model",
    "sign_model.json"
)

WEIGHTS_PATH = os.path.join(
    BASE_DIR,
    "model",
    "sign_model.weights.h5"
)

CLASS_INDICES_PATH = os.path.join(
    BASE_DIR,
    "model",
    "class_indices.json"
)

IMG_SIZE = 64

START_HOUR = 18
END_HOUR = 23

KNOWN_WORDS = {
    "HI": "HI",
    "BYE": "BYE",
    "YES": "YES",
    "NO": "NO",
    "LOVE": "LOVE",
}

ROI_TOP = 100
ROI_BOTTOM = 400
ROI_LEFT = 350
ROI_RIGHT = 650


def SignLanguageModel(json_file, weights_file):

    with open(json_file, "r") as file:
        loaded_model_json = file.read()

    model = model_from_json(
        loaded_model_json
    )

    model.load_weights(
        weights_file
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def load_labels(path):

    with open(path, "r") as f:
        class_indices = json.load(f)

    return {
        v: k
        for k, v in class_indices.items()
    }


@st.cache_resource
def load_model():

    return SignLanguageModel(
        JSON_PATH,
        WEIGHTS_PATH
    )


@st.cache_data
def load_class_labels():

    return load_labels(
        CLASS_INDICES_PATH
    )


model = load_model()

LABELS = load_class_labels()


def is_within_operating_hours():

    now = datetime.datetime.now().time()

    return (
        datetime.time(START_HOUR, 0)
        <= now
        <= datetime.time(END_HOUR, 0)
    )


def preprocess(roi_gray):

    roi = cv2.resize(
        roi_gray,
        (IMG_SIZE, IMG_SIZE)
    )

    roi = roi.astype(
        "float32"
    ) / 255.0

    roi = roi.reshape(
        1,
        IMG_SIZE,
        IMG_SIZE,
        1
    )

    return roi


def predict_letter(gray_roi):

    roi = preprocess(
        gray_roi
    )

    prediction = model.predict(
        roi,
        verbose=0
    )

    pred_idx = int(
        np.argmax(prediction)
    )

    return LABELS.get(
        pred_idx,
        "?"
    )


def update_word_buffer(letter):

    if letter in (
        "nothing",
        "?",
        "del"
    ):
        return

    if letter == "space":

        st.session_state.word_buffer.append(
            " "
        )

    else:

        st.session_state.word_buffer.append(
            letter
        )


def get_recognized_word():

    current = "".join(
        st.session_state.word_buffer[-8:]
    ).upper()

    for word in KNOWN_WORDS:

        if word in current:
            return word

    return None


class VideoProcessor(VideoProcessorBase):

    def recv(self, frame):

        image = frame.to_ndarray(
            format="bgr24"
        )

        image = cv2.flip(
            image,
            1
        )

        height, width = image.shape[:2]

        left = min(
            ROI_LEFT,
            width
        )

        right = min(
            ROI_RIGHT,
            width
        )

        top = min(
            ROI_TOP,
            height
        )

        bottom = min(
            ROI_BOTTOM,
            height
        )

        if (
            right > left
            and bottom > top
        ):

            cv2.rectangle(
                image,
                (left, top),
                (right, bottom),
                (0, 255, 0),
                2
            )

            roi = image[
                top:bottom,
                left:right
            ]

            gray_roi = cv2.cvtColor(
                roi,
                cv2.COLOR_BGR2GRAY
            )

            try:

                pred = predict_letter(
                    gray_roi
                )

                cv2.putText(
                    image,
                    f"Sign: {pred}",
                    (
                        left,
                        max(
                            top - 10,
                            30
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

            except Exception:

                cv2.putText(
                    image,
                    "Unable to detect",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2
                )

        return frame.from_ndarray(
            np.asarray(
                image,
                dtype=np.uint8
            ),
            format="bgr24"
        )


st.set_page_config(
    page_title="Sign Language Detector",
    page_icon="",
    layout="wide"
)


st.title(
    "Sign Language Detector"
)


st.info(
    f"Operating Hours: "
    f"{START_HOUR}:00 - {END_HOUR}:00 "
    f"(6 PM - 10 PM)"
)


if "word_buffer" not in st.session_state:

    st.session_state.word_buffer = []


if "uploaded_prediction" not in st.session_state:

    st.session_state.uploaded_prediction = ""


tab1, tab2, tab3 = st.tabs(
    [
        "Upload Image",
        "Real-Time Video",
        "Word Builder"
    ]
)


with tab1:

    st.subheader(
        "Upload Sign Image"
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        )

        st.image(
            image,
            caption="Uploaded Image",
            width=400
        )

        if st.button(
            "Detect Sign",
            key="detect_image"
        ):

            if not is_within_operating_hours():

                st.warning(
                    f"This model only operates "
                    f"between {START_HOUR}:00 "
                    f"and {END_HOUR}:00 "
                    f"(6 PM - 10 PM)."
                )

            else:

                try:

                    image_array = np.array(
                        image
                    )

                    if len(
                        image_array.shape
                    ) == 3:

                        if image_array.shape[2] == 4:

                            gray_image = cv2.cvtColor(
                                image_array,
                                cv2.COLOR_RGBA2GRAY
                            )

                        else:

                            gray_image = cv2.cvtColor(
                                image_array,
                                cv2.COLOR_RGB2GRAY
                            )

                    else:

                        gray_image = image_array

                    prediction = predict_letter(
                        gray_image
                    )

                    st.session_state.uploaded_prediction = prediction

                    st.success(
                        f"Predicted Sign: "
                        f"{prediction}"
                    )

                except Exception as e:

                    st.error(
                        f"Unable to detect: {e}"
                    )

        if st.session_state.uploaded_prediction:

            if st.button(
                "Add Detected Letter",
                key="add_uploaded"
            ):

                update_word_buffer(
                    st.session_state.uploaded_prediction
                )

                st.success(
                    f"Added: "
                    f"{st.session_state.uploaded_prediction}"
                )


with tab2:

    st.subheader(
        "Real-Time Sign Detection"
    )

    if not is_within_operating_hours():

        st.warning(
            f"Real-time detection is available "
            f"only between {START_HOUR}:00 "
            f"and {END_HOUR}:00 "
            f"(6 PM - 10 PM)."
        )

    else:

        st.write(
            "Place your hand inside "
            "the green rectangle."
        )

        webrtc_streamer(
            key="sign-language-camera",

            video_processor_factory=VideoProcessor,

            media_stream_constraints={
                "video": True,
                "audio": False
            },

            async_processing=True
        )


with tab3:

    st.subheader(
        "Word Builder"
    )

    current_buffer = "".join(
        st.session_state.word_buffer[-20:]
    )

    recognized_word = get_recognized_word()

    if recognized_word:

        st.success(
            f"Recognized Word: "
            f"{recognized_word}"
        )

    else:

        st.write(
            f"Buffer: {current_buffer}"
        )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "Clear Word",
            key="clear_word"
        ):

            st.session_state.word_buffer.clear()

            st.session_state.uploaded_prediction = ""

            st.rerun()

    with col2:

        if st.button(
            "Add Space",
            key="add_space"
        ):

            update_word_buffer(
                "space"
            )

            st.rerun()

    st.divider()

    st.write(
        "Known Words"
    )

    for word in KNOWN_WORDS:

        st.write(
            f"- {word}"
        )