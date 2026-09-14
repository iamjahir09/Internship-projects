import os
import sys
import tempfile

import cv2
import numpy as np
import streamlit as st



SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


from processor import DrowsinessProcessor
from video_processor import VideoProcessor



st.set_page_config(
    page_title="Drowsiness Detection",
    page_icon="D",
    layout="wide"
)



st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .sub-title {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .sleeping-box {
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #ff0000;
        text-align: center;
        margin-bottom: 10px;
    }

    .awake-box {
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #00aa00;
        text-align: center;
        margin-bottom: 10px;
    }

    .unknown-box {
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #cccc00;
        text-align: center;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)



st.markdown(
    '<div class="main-title">Drowsiness Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Detect people, sleeping state and predicted age'
    '</div>',
    unsafe_allow_html=True
)



@st.cache_resource
def load_image_processor():
    return DrowsinessProcessor()


@st.cache_resource
def load_video_processor():
    return VideoProcessor()



st.sidebar.title("Options")

mode = st.sidebar.radio(
    "Select Input Type",
    ["Image", "Video"]
)



if mode == "Image":

    st.header("Image Detection")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:

        file_bytes = uploaded_file.read()

        image_array = cv2.imdecode(
            np.frombuffer(
                file_bytes,
                dtype=np.uint8
            ),
            cv2.IMREAD_COLOR
        )

        if image_array is None:

            st.error("Could not read the image.")

        else:

            col1, col2 = st.columns(2)


            with col1:

                st.subheader("Input Image")

                st.image(
                    cv2.cvtColor(
                        image_array,
                        cv2.COLOR_BGR2RGB
                    ),
                    use_container_width=True
                )

            process_button = st.button(
                "Detect Drowsiness",
                type="primary",
                use_container_width=True
            )


            if process_button:

                with st.spinner(
                    "Detecting people and analyzing faces..."
                ):

                    processor = load_image_processor()

                    result = processor.process_image(
                        image_array
                    )

                output_image = result["image"]


                with col2:

                    st.subheader("Detection Result")

                    st.image(
                        cv2.cvtColor(
                            output_image,
                            cv2.COLOR_BGR2RGB
                        ),
                        use_container_width=True
                    )


                st.divider()

                total_people = result["total_people"]

                sleeping_count = result["sleeping_count"]

                awake_count = result["awake_count"]

                unknown_count = result["unknown_count"]

                col1, col2, col3, col4 = st.columns(4)

                with col1:

                    st.metric(
                        "Total People",
                        total_people
                    )

                with col2:

                    st.metric(
                        "Sleeping",
                        sleeping_count
                    )

                with col3:

                    st.metric(
                        "Awake",
                        awake_count
                    )

                with col4:

                    st.metric(
                        "Unknown",
                        unknown_count
                    )


                st.subheader("Person Details")

                people = result["people"]

                if len(people) == 0:

                    st.warning(
                        "No person detected in the image."
                    )

                else:

                    for i, person in enumerate(
                        people,
                        start=1
                    ):

                        state = person["state"]
                        age = person["age"]
                        confidence = person["confidence"]

                        age_text = (
                            age if age is not None else "Unknown"
                        )

                        if state == "Sleeping":

                            st.markdown(
                                f"""
                                <div class="sleeping-box">
                                    <h3>Person {i}</h3>
                                    <p>
                                        <b>Status:</b> Sleeping
                                    </p>
                                    <p>
                                        <b>Age:</b> {age_text}
                                    </p>
                                    <p>
                                        <b>Confidence:</b>
                                        {confidence:.2f}%
                                    </p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        elif state == "Awake":

                            st.markdown(
                                f"""
                                <div class="awake-box">
                                    <h3>Person {i}</h3>
                                    <p>
                                        <b>Status:</b> Awake
                                    </p>
                                    <p>
                                        <b>Age:</b> {age_text}
                                    </p>
                                    <p>
                                        <b>Confidence:</b>
                                        {confidence:.2f}%
                                    </p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        else:

                            st.markdown(
                                f"""
                                <div class="unknown-box">
                                    <h3>Person {i}</h3>
                                    <p>
                                        <b>Status:</b>
                                        Face Not Detected
                                    </p>
                                    <p>
                                        <b>Age:</b> {age_text}
                                    </p>
                                    <p>
                                        <b>Detection Confidence:</b>
                                        {confidence:.2f}%
                                    </p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )


                if sleeping_count > 0:

                    sleeping_ages = result["sleeping_ages"]

                    age_text = ", ".join(
                        str(age)
                        for age in sleeping_ages
                    )

                    st.warning(
                        f"Sleeping People: {sleeping_count} | "
                        f"Sleeping Ages: {age_text}"
                    )

                else:

                    st.success(
                        "No sleeping person detected."
                    )

                if unknown_count > 0:

                    st.info(
                        f"{unknown_count} person(s) could not be "
                        f"analyzed because the face was not "
                        f"detected."
                    )



else:

    st.header("Video Detection")

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:


        input_extension = os.path.splitext(
            uploaded_video.name
        )[1]

        input_temp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=input_extension
        )

        input_temp.write(
            uploaded_video.read()
        )

        input_temp.close()


        st.subheader("Input Video")

        st.video(
            input_temp.name
        )

        process_button = st.button(
            "Process Video",
            type="primary",
            use_container_width=True
        )


        if process_button:

            output_temp = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            output_path = output_temp.name

            output_temp.close()

            with st.spinner(
                "Processing video... This may take some time."
            ):

                video_processor = load_video_processor()

                result = video_processor.process_video(
                    input_temp.name,
                    output_path
                )

            st.success(
                "Video processing completed."
            )


            st.subheader("Detection Result")

            st.video(
                output_path
            )


            st.divider()

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Processed Frames",
                    result["total_frames"]
                )

            with col2:

                st.metric(
                    "Maximum Sleeping",
                    result["max_sleeping_count"]
                )

            with col3:

                if result["sleeping_ages"]:

                    ages = ", ".join(
                        str(age)
                        for age in result["sleeping_ages"]
                    )

                else:

                    ages = "None"

                st.metric(
                    "Sleeping Ages",
                    ages
                )


            if result["max_sleeping_count"] > 0:

                st.warning(
                    f"Sleeping People: "
                    f"{result['max_sleeping_count']}"
                )

                if result["sleeping_ages"]:

                    st.write(
                        "Predicted ages of sleeping people:",
                        result["sleeping_ages"]
                    )

            else:

                st.success(
                    "No sleeping people detected."
                )


            with open(
                output_path,
                "rb"
            ) as video_file:

                st.download_button(
                    label="Download Result Video",
                    data=video_file,
                    file_name="drowsiness_result.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )