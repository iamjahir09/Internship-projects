import os
import sys
import tempfile

import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.voice_emotion import predict_emotion

st.set_page_config(
    page_title="Voice Emotion Detection",
    page_icon="🎤",
    layout="centered"
)

st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #ddd;
        margin-top: 20px;
    }

    .emotion {
        font-size: 30px;
        font-weight: bold;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">🎤 Voice Emotion Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Upload a female voice note or record your voice to detect emotion.</div>',
    unsafe_allow_html=True
)

st.header("📁 Upload Voice Note")

uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=["wav", "mp3", "ogg", "flac", "m4a"]
)

if uploaded_file is not None:
    st.audio(uploaded_file, format=uploaded_file.type)

    if st.button(
        "🔍 Detect Emotion from Uploaded Voice",
        use_container_width=True
    ):
        temp_path = None

        try:
            suffix = os.path.splitext(uploaded_file.name)[1]

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_path = temp_file.name

            emotion, confidence = predict_emotion(temp_path)

            if emotion == "female_only":
                st.warning(
                    "Non-female voice detected. Please upload a female voice."
                )
            else:
                st.success("Emotion detected successfully!")

                st.markdown(
                    f"""
                    <div class="result-box">
                        <div>Detected Emotion</div>
                        <div class="emotion">{emotion.capitalize()}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%"
                )

        except Exception as e:
            st.error(f"Could not process the audio file.\n\nError: {e}")

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

st.divider()

st.header("🎙️ Record Your Voice")

st.write("Record your voice below and then click the detection button.")

try:
    from audio_recorder_streamlit import audio_recorder

    audio_bytes = audio_recorder(
        text="Click to record",
        recording_color="#ff4b4b",
        neutral_color="#6c757d",
        icon_name="microphone",
        icon_size="2x"
    )

    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")

        if st.button(
            "🔍 Detect Emotion from Recording",
            use_container_width=True
        ):
            temp_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".wav"
                ) as temp_file:
                    temp_file.write(audio_bytes)
                    temp_path = temp_file.name

                emotion, confidence = predict_emotion(temp_path)

                if emotion == "female_only":
                    st.warning(
                        "Non-female voice detected. Please record a female voice."
                    )
                else:
                    st.success("Emotion detected successfully!")

                    st.markdown(
                        f"""
                        <div class="result-box">
                            <div>Detected Emotion</div>
                            <div class="emotion">{emotion.capitalize()}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.metric(
                        "Confidence",
                        f"{confidence * 100:.2f}%"
                    )

            except Exception as e:
                st.error(
                    f"Could not process the recording.\n\nError: {e}"
                )

            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

except ImportError:
    st.warning(
        "Recording feature requires the audio-recorder-streamlit package."
    )

st.divider()

st.caption(
    "Voice Emotion Detection using MFCC features and a CNN model."
)