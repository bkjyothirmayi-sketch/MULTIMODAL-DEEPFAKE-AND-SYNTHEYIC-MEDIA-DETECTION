
import streamlit as st
import os
import sys
import tempfile
import importlib.util
import traceback

# =====================================================
# PROJECT PATHS
# =====================================================

PROJECT_DIR = r"D:\AI_Projects\multimodal-deepfake-detection"

IMAGE_DETECTOR_PATH = os.path.join(
    r"D:\AI_Projects\intermediate fusion\image detector.py"
)

AUDIO_DETECTOR_PATH = os.path.join(
    r"D:\AI_Projects\intermediate fusion\audio detector.py"
)

VIDEO_DETECTOR_PATH = os.path.join(
    r"D:\AI_Projects\intermediate fusion\video detector.py"
)

# =====================================================
# STREAMLIT PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Multimodal Deepfake Detection",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Multimodal Deepfake Detection System")

st.write(
    "An AI-based application for detecting potentially "
    "fake or manipulated images, audio, and videos."
)

st.info(
    "Select a modality below and upload a file to run its "
    "independently trained deepfake detector."
)

# =====================================================
# LOAD DETECTOR MODULE
# =====================================================

@st.cache_resource
def load_detector_module(module_path, module_name):

    if not os.path.isfile(module_path):
        raise FileNotFoundError(
            f"Detector file not found:\n{module_path}"
        )

    spec = importlib.util.spec_from_file_location(
        module_name, module_path
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def get_detector(module, function_name):

    detector = getattr(module, function_name, None)

    if detector is None or not callable(detector):
        raise AttributeError(
            f"Function '{function_name}' was not found "
            f"in the detector file."
        )

    return detector


# =====================================================
# SAVE UPLOADED FILE TEMPORARILY
# =====================================================

def save_uploaded_file(uploaded_file):

    suffix = os.path.splitext(uploaded_file.name)[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp_file:

        temp_file.write(uploaded_file.getbuffer())
        return temp_file.name


# =====================================================
# DISPLAY DETECTION RESULT
# =====================================================

def display_result(result):

    if not isinstance(result, dict):
        st.error(
            "Detector must return a dictionary containing "
            "prediction and confidence."
        )
        st.write("Returned value:", result)
        return

    prediction = result.get(
        "prediction",
        result.get("label", "Unknown")
    )

    confidence = result.get("confidence", None)

    fake_probability = result.get(
        "fake_probability", None
    )

    real_probability = result.get(
        "real_probability", None
    )

    st.subheader("Detection Result")

    if str(prediction).upper() == "FAKE":
        st.error("⚠️ Prediction: FAKE")
    elif str(prediction).upper() == "REAL":
        st.success("✅ Prediction: REAL")
    else:
        st.info(f"Prediction: {prediction}")

    if confidence is not None:
        st.metric(
            "Confidence",
            f"{float(confidence):.2f}%"
        )

    col1, col2 = st.columns(2)

    with col1:
        if fake_probability is not None:
            st.metric(
                "Fake Probability",
                f"{float(fake_probability):.2f}%"
            )

    with col2:
        if real_probability is not None:
            st.metric(
                "Real Probability",
                f"{float(real_probability):.2f}%"
            )

    st.caption(
        "Confidence and probabilities are model outputs, "
        "not guarantees that a file is genuine or manipulated."
    )


# =====================================================
# DETECTION EXECUTION
# =====================================================

def run_detection(
    uploaded_file,
    detector_path,
    module_name,
    function_name
):

    if uploaded_file is None:
        st.warning("Please upload a file first.")
        return

    temp_path = None

    try:
        with st.spinner("Analyzing file..."):

            temp_path = save_uploaded_file(uploaded_file)

            module = load_detector_module(
                detector_path,
                module_name
            )

            detector_function = get_detector(
                module,
                function_name
            )

            result = detector_function(temp_path)

        display_result(result)

    except Exception as error:

        st.error("Detection failed.")
        st.exception(error)

    finally:

        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


# =====================================================
# THREE DETECTION TABS
# =====================================================

image_tab, audio_tab, video_tab = st.tabs(
    ["🖼️ Image Detection", "🎵 Audio Detection", "🎬 Video Detection"]
)

# =====================================================
# IMAGE DETECTION
# =====================================================

with image_tab:

    st.header("Image Deepfake Detection")

    st.write(
        "Upload an image to analyze whether it is "
        "potentially real or AI-generated/manipulated."
    )

    image_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"],
        key="image_upload"
    )

    if image_file is not None:
        st.image(
            image_file,
            caption="Uploaded Image",
            use_container_width=True
        )

    if st.button("Detect Image", key="image_button"):

        run_detection(
            image_file,
            IMAGE_DETECTOR_PATH,
            "image_detector_module",
            "predict_image"
        )


# =====================================================
# AUDIO DETECTION
# =====================================================

with audio_tab:

    st.header("Audio Deepfake Detection")

    st.write(
        "Upload an audio file to analyze whether it is "
        "potentially genuine or synthetically generated."
    )

    audio_file = st.file_uploader(
        "Upload Audio",
        type=["wav", "mp3", "flac", "ogg"],
        key="audio_upload"
    )

    if audio_file is not None:
        st.audio(audio_file)

    if st.button("Detect Audio", key="audio_button"):

        run_detection(
            audio_file,
            AUDIO_DETECTOR_PATH,
            "audio_detector_module",
            "predict_audio"
        )


# =====================================================
# VIDEO DETECTION
# =====================================================

with video_tab:

    st.header("Video Deepfake Detection")

    st.write(
        "Upload a video to analyze sampled frames "
        "for potentially manipulated visual content."
    )

    video_file = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload"
    )

    if video_file is not None:
        st.video(video_file)

    if st.button("Detect Video", key="video_button"):

        run_detection(
            video_file,
            VIDEO_DETECTOR_PATH,
            "video_detector_module",
            "predict_video"
        )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "Multimodal Deepfake Detection | "
    "Image: MobileNetV2 | Audio: MobileNetV2 | "
    "Video: ShuffleNetV2"
)

st.caption(
    "This application integrates three independent "
    "modality-specific detectors. It does not perform "
    "joint multimodal feature fusion."
)
