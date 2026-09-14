import os
import tempfile

import cv2
import numpy as np
import streamlit as st

from detector import detect


st.set_page_config(
    page_title="Traffic Vision",
    layout="wide"
)


st.markdown("""
<style>

.stApp {
    background-color: #0b0f19;
}

.block-container {
    max-width: 1450px;
    padding: 2rem 3rem 4rem 3rem;
}

.title {
    font-size: 44px;
    font-weight: 800;
    color: #f8fafc;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 16px;
    color: #94a3b8;
    margin-bottom: 30px;
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #f8fafc;
    margin-top: 25px;
    margin-bottom: 15px;
}

.metric-card {
    background-color: #111827;
    border: 1px solid #1f2937;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
}

.metric-number {
    font-size: 34px;
    font-weight: 800;
    color: #f8fafc;
}

.metric-label {
    font-size: 14px;
    color: #94a3b8;
    margin-top: 5px;
}

.legend {
    background-color: #111827;
    border: 1px solid #1f2937;
    border-radius: 14px;
    padding: 18px;
    margin-top: 20px;
    color: #cbd5e1;
}

.red-box {
    color: #ef4444;
    font-weight: 700;
}

.blue-box {
    color: #3b82f6;
    font-weight: 700;
}

.green-box {
    color: #22c55e;
    font-weight: 700;
}

.stButton > button {
    width: 100%;
    min-height: 48px;
    border-radius: 10px;
    background-color: #2563eb;
    border: 1px solid #2563eb;
    color: white;
    font-size: 15px;
    font-weight: 700;
}

.stButton > button:hover {
    background-color: #1d4ed8;
    border-color: #1d4ed8;
}

[data-testid="stFileUploader"] {
    background-color: #111827;
    border: 1px solid #1f2937;
    border-radius: 14px;
    padding: 10px;
}

[data-testid="stFileUploaderDropzone"] {
    background-color: #0f172a;
    border: 1px dashed #334155;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


st.markdown(
    '<div class="title">Traffic Vision</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Vehicle color detection and traffic object counting using YOLO11s '
    'and Custom CNN.'
    '</div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="section-title">Upload Traffic Image</div>',
    unsafe_allow_html=True
)


uploaded_file = st.file_uploader(
    "Choose a traffic image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
        "bmp",
        "tif",
        "tiff",
        "avif"
    ]
)


if uploaded_file is not None:

    file_bytes = uploaded_file.read()

    original_image = cv2.imdecode(
        np.frombuffer(file_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )

    if original_image is None:
        st.error("Unable to read this image format.")
        st.stop()

    st.markdown(
        '<div class="section-title">Image Analysis</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:

        st.subheader("Original Image")

        st.image(
            cv2.cvtColor(
                original_image,
                cv2.COLOR_BGR2RGB
            ),
            use_container_width=True
        )

    with col2:

        st.subheader("AI Detection Result")

        analyze = st.button(
            "Analyze Image",
            use_container_width=True
        )

        if analyze:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".jpg"
            ) as temp_file:

                temp_file.write(file_bytes)
                image_path = temp_file.name

            try:

                with st.spinner(
                    "Analyzing traffic image..."
                ):

                    result_image, cars, people = detect(
                        image_path
                    )

            finally:

                if os.path.exists(image_path):
                    os.remove(image_path)

            st.image(
                cv2.cvtColor(
                    result_image,
                    cv2.COLOR_BGR2RGB
                ),
                use_container_width=True
            )

            st.success(
                "Analysis completed successfully."
            )

            st.markdown(
                '<div class="section-title">Detection Summary</div>',
                unsafe_allow_html=True
            )

            metric1, metric2 = st.columns(2, gap="medium")

            with metric1:

                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-number">{cars}</div>
                        <div class="metric-label">
                            Cars Detected
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with metric2:

                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-number">{people}</div>
                        <div class="metric-label">
                            People Detected
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("""
                <div class="legend">
                    <b>Detection Legend</b><br><br>
                    <span class="red-box">■ Red Box</span> - Blue Car<br>
                    <span class="blue-box">■ Blue Box</span> - Other Color Car<br>
                    <span class="green-box">■ Green Box</span> - Person
                </div>
""", unsafe_allow_html=True)