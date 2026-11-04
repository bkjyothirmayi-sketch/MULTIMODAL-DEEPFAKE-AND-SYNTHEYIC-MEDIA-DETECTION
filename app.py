
import streamlit as st
import os
import sys
import tempfile
import importlib.util
import inspect

# =====================================================
# PROJECT PATHS
# =====================================================

BASE_DIR = r"D:\AI_Projects\intermediate fusion"

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
# STREAMLIT CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Multimodal Deepfake Detection",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Multimodal Deepfake Detection System")

st.markdown(
    """
    Upload an image, audio file, or video to analyze it
    using the corresponding independently trained AI model.
    """
)

# =====================================================
# IMPORT DETECTOR FILE
# =====================================================

@st.cache_resource
def import_detector(path, module_name):

    if not os.path.isfile(path):
        raise FileNotFoundError(
            "Detector file not found: " + path
        )

    spec = importlib.util.spec_from_file_location(
        module_name, path
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# =====================================================
# SAVE UPLOADED FILE
# =====================================================

def save_uploaded_file(uploaded_file):

    extension = os.path.splitext(
        uploaded_file.name
    )[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=extension
    ) as temp:

        temp.write(uploaded_file.getbuffer())

        return temp.name


# =====================================================
# DISPLAY DETECTION RESULT
# Supports both lowercase and capitalized keys
# =====================================================

def display_result(result):

    if not isinstance(result, dict):
        st.write(result)
        return

    prediction = result.get(
        "prediction",
        result.get(
            "Prediction",
            result.get("label", "Unknown")
        )
    )

    confidence = result.get(
        "confidence",
        result.get("Confidence")
    )

    fake_probability = result.get(
        "fake_probability",
        result.get("Fake Probability")
    )

    real_probability = result.get(
        "real_probability",
        result.get("Real Probability")
    )

    modality = result.get(
        "modality",
        result.get("Modality", "Detection")
    )

    st.subheader(f"{modality} Result")

    if str(prediction).upper() == "FAKE":
        st.error("⚠️ Prediction: FAKE")

    elif str(prediction).upper() == "REAL":
        st.success("✅ Prediction: REAL")

    else:
        st.info(f"Prediction: {prediction}")

    col1, col2, col3 = st.columns(3)

    with col1:
        if confidence is not None:
            st.metric(
                "Confidence",
                f"{float(confidence):.2f}%"
            )

    with col2:
        if fake_probability is not None:
            st.metric(
                "Fake Probability",
                f"{float(fake_probability):.2f}%"
            )

    with col3:
        if real_probability is not None:
            st.metric(
                "Real Probability",
                f"{float(real_probability):.2f}%"
            )

    if "Frames Analysed" in result:
        st.write(
            "Frames Analysed:",
            result["Frames Analysed"]
        )

    if "frames_analysed" in result:
        st.write(
            "Frames Analysed:",
            result["frames_analysed"]
        )

    st.caption(
        "Confidence is a model output and is not a guarantee "
        "that the media is authentic or manipulated."
    )


# =====================================================
# GENERIC DETECTION EXECUTION
# =====================================================

def run_detection(
    uploaded_file,
    detector_path,
    module_name,
    function_name,
    loader_name
):

    if uploaded_file is None:
        st.warning("Please upload a file first.")
        return

    temp_path = None

    try:

        with st.spinner("Loading model and analyzing..."):

            temp_path = save_uploaded_file(uploaded_file)

            module = import_detector(
                detector_path,
                module_name
            )

            detector_function = getattr(
                module,
                function_name
            )

            # -----------------------------------------
            # LOAD MODEL USING ACTUAL LOADER FUNCTION
            # -----------------------------------------

            loader = getattr(
                module,
                loader_name,
                None
            )

            if not callable(loader):
                raise AttributeError(
                    f"Model loader '{loader_name}' "
                    f"not found in {detector_path}"
                )

            loaded = loader()

            # Image detector returns model, idx_to_class.
            if isinstance(loaded, tuple):

                model, idx_to_class = loaded

                result = detector_function(
                    temp_path,
                    model,
                    idx_to_class
                )

            else:

                # For a detector whose prediction function
                # accepts only the file path.
                parameters = inspect.signature(
                    detector_function
                ).parameters

                if len(parameters) == 1:

                    result = detector_function(temp_path)

                else:

                    raise RuntimeError(
                        "Unexpected detector loader output "
                        "or prediction function arguments."
                    )

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
# APPLICATION TABS
# =====================================================

image_tab, audio_tab, video_tab = st.tabs(
    [
        "🖼️ Image Detection",
        "🎵 Audio Detection",
        "🎬 Video Detection"
    ]
)


# =====================================================
# IMAGE DETECTION
# =====================================================

with image_tab:

    st.header("Image Deepfake Detection")

    st.write(
        "Model: MobileNetV2"
    )

    image_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        key="image_upload"
    )

    if image_file is not None:

        st.image(
            image_file,
            caption="Uploaded Image",
            use_container_width=True
        )

    if st.button(
        "Detect Image",
        key="detect_image"
    ):

        run_detection(
            image_file,
            IMAGE_DETECTOR_PATH,
            "image_detector_module",
            "predict_image",
            "load_image_detector"
        )


# =====================================================
# AUDIO DETECTION
# =====================================================

with audio_tab:

    st.header("Audio Deepfake Detection")

    st.write(
        "Model: MobileNetV2"
    )

    audio_file = st.file_uploader(
        "Upload an audio file",
        type=["wav", "mp3", "flac", "ogg"],
        key="audio_upload"
    )

    if audio_file is not None:
        st.audio(audio_file)

    if st.button(
        "Detect Audio",
        key="detect_audio"
    ):

        run_detection(
            audio_file,
            AUDIO_DETECTOR_PATH,
            "audio_detector_module",
            "predict_audio",
            "load_audio_detector"
        )


# =====================================================
# VIDEO DETECTION
# =====================================================

with video_tab:

    st.header("Video Deepfake Detection")

    st.write(
        "Model: ShuffleNetV2"
    )

    video_file = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload"
    )

    if video_file is not None:
        st.video(video_file)

    if st.button(
        "Detect Video",
        key="detect_video"
    ):

        run_detection(
            video_file,
            VIDEO_DETECTOR_PATH,
            "video_detector_module",
            "predict_video",
            "load_video_detector"
        )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "Image: MobileNetV2 | "
    "Audio: MobileNetV2 | "
    "Video: ShuffleNetV2"
)

st.caption(
    "Three independently trained modality-specific "
    "deepfake detectors integrated into one application."
)
