import os
import sys
import tempfile
import streamlit as st
from PIL import Image


CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)


from predictor import Predictor


st.set_page_config(
    page_title="Nationality Detection System",
    layout="centered"
)


@st.cache_resource
def load_predictor():
    return Predictor()


try:
    predictor = load_predictor()

except Exception as e:
    st.error("Model loading failed.")
    st.exception(e)
    st.stop()


st.title("Nationality Detection System")

st.write(
    "Upload an image to predict nationality, "
    "age, emotion and dress colour."
)


uploaded_file = st.file_uploader(
    "Drag and drop an image here",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )


    if st.button(
        "🔍 Predict",
        use_container_width=True
    ):

        temp_path = None

        try:

            suffix = os.path.splitext(
                uploaded_file.name
            )[1]

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getbuffer()
                )

                temp_path = temp_file.name


            with st.spinner(
                "Analyzing image..."
            ):

                result = predictor.predict(
                    temp_path
                )


            st.success(
                "Prediction completed!"
            )

            st.subheader(
                "Prediction Results"
            )


            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Nationality",
                    result["nationality"]
                )

                st.metric(
                    "Nationality Confidence",
                    f"{result['nationality_confidence'] * 100:.2f}%"
                )


            with col2:

                st.metric(
                    "Emotion",
                    result["emotion"]
                )

                if result["emotion"] != "Unable to detect":

                    st.metric(
                        "Emotion Confidence",
                        f"{result['emotion_confidence'] * 100:.2f}%"
                    )


            if "age" in result:

                st.divider()

                st.subheader(
                    "Age Prediction"
                )

                st.metric(
                    "Predicted Age",
                    f"{result['age']} years"
                )


            if "dress_color" in result:

                st.divider()

                st.subheader(
                    "Dress Colour"
                )

                col3, col4 = st.columns(2)

                with col3:

                    st.metric(
                        "Dress Colour",
                        result["dress_color"].title()
                    )

                with col4:

                    st.metric(
                        "Dress Confidence",
                        f"{result['dress_confidence'] * 100:.2f}%"
                    )


        except Exception as e:

            st.error(
                "Prediction failed."
            )

            st.exception(e)


        finally:

            if (
                temp_path is not None
                and os.path.exists(temp_path)
            ):

                os.remove(
                    temp_path
                )