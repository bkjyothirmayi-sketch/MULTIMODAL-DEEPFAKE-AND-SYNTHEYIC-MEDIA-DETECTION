
import os
import cv2
import importlib.util
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# =====================================================
# 1. DEVICE
# =====================================================

DEVICE = torch.device("cpu")

# =====================================================
# 2. FILE PATHS
# =====================================================

ARCHITECTURE_PATH = (
    r"D:\AI_Projects\intermediate fusion\model architecture.py"
)

# IMPORTANT:
# Replace this with your actual ShuffleNetV2 checkpoint path.
VIDEO_MODEL_PATH = (
    r"D:\AI_Projects\archive (2)\saved_models\shufflenet_video_best_v3.pth"
)

# =====================================================
# 3. IMPORT MODEL ARCHITECTURE
# =====================================================

if not os.path.isfile(ARCHITECTURE_PATH):
    raise FileNotFoundError(
        "Architecture file not found: " + ARCHITECTURE_PATH
    )

spec = importlib.util.spec_from_file_location(
    "model_architecture",
    ARCHITECTURE_PATH
)

architecture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(architecture)

create_video_model = architecture.create_video_model
load_checkpoint = architecture.load_checkpoint

print("Video model architecture imported successfully!")

# =====================================================
# 4. VIDEO FRAME PREPROCESSING
# =====================================================

IMAGE_SIZE = 224
MAX_FRAMES = 16

video_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================================================
# 5. LOAD TRAINED VIDEO MODEL
# =====================================================

def load_video_detector():

    if not os.path.isfile(VIDEO_MODEL_PATH):
        raise FileNotFoundError(
            "Video checkpoint not found: " + VIDEO_MODEL_PATH
        )

    model = create_video_model()

    model, checkpoint = load_checkpoint(
        model,
        VIDEO_MODEL_PATH
    )

    model = model.to(DEVICE)
    model.eval()

    class_mapping = checkpoint.get(
        "class_mapping",
        {"fake": 0, "real": 1}
    )

    idx_to_class = {
        int(index): label
        for label, index in class_mapping.items()
    }

    print("\nVIDEO MODEL LOADED SUCCESSFULLY")
    print("Architecture: ShuffleNetV2")
    print("Classes:", idx_to_class)

    return model, idx_to_class


# =====================================================
# 6. EXTRACT FRAMES FROM VIDEO
# =====================================================

def extract_video_frames(video_path):

    if not os.path.isfile(video_path):
        raise FileNotFoundError(
            "Video file not found: " + video_path
        )

    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        raise ValueError("Unable to open video file.")

    total_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:
        capture.release()
        raise ValueError("No frames found in video.")

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        min(MAX_FRAMES, total_frames),
        dtype=int
    )

    frames = []

    for frame_index in frame_indices:

        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_index)
        )

        success, frame = capture.read()

        if not success:
            continue

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frame_image = Image.fromarray(frame)

        frame_tensor = video_transform(frame_image)

        frames.append(frame_tensor)

    capture.release()

    if len(frames) == 0:
        raise ValueError("Could not extract video frames.")

    return frames


# =====================================================
# 7. VIDEO PREDICTION
# =====================================================

def predict_video(video_path, model, idx_to_class):

    frames = extract_video_frames(video_path)

    predictions = []

    with torch.no_grad():

        for frame_tensor in frames:

            frame_tensor = frame_tensor.unsqueeze(0).to(DEVICE)

            outputs = model(frame_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predictions.append(
                probabilities.squeeze(0).cpu().numpy()
            )

    # Average frame-level probabilities
    average_probabilities = np.mean(
        predictions,
        axis=0
    )

    predicted_index = int(
        np.argmax(average_probabilities)
    )

    prediction = idx_to_class[predicted_index]

    confidence = (
        average_probabilities[predicted_index] * 100
    )

    fake_probability = (
        average_probabilities[0] * 100
    )

    real_probability = (
        average_probabilities[1] * 100
    )

    return {
        "Modality": "Video",
        "Prediction": prediction.upper(),
        "Confidence": round(float(confidence), 2),
        "Fake Probability": round(float(fake_probability), 2),
        "Real Probability": round(float(real_probability), 2),
        "Frames Analysed": len(frames)
    }


# =====================================================
# 8. TEST VIDEO DETECTOR
# =====================================================

if __name__ == "__main__":

    model, idx_to_class = load_video_detector()

    video_path = input(
        "\nEnter video file path: "
    ).strip().strip('"')

    result = predict_video(
        video_path,
        model,
        idx_to_class
    )

    print("\n========== VIDEO DETECTION RESULT ==========")

    for key, value in result.items():
        print(f"{key}: {value}")

    print("============================================")




def extract_video_frames(video_path):

    if not os.path.isfile(video_path):
        raise FileNotFoundError(
            "Video file not found: " + video_path
        )

    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        raise ValueError("Unable to open video file.")

    reported_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = capture.get(cv2.CAP_PROP_FPS)

    print("Reported frame count:", reported_frames)
    print("Video FPS:", fps)

    frames = []
    total_read = 0

    rng = np.random.default_rng(42)

    while True:

        success, frame = capture.read()

        if not success:
            break

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frame_image = Image.fromarray(frame)
        frame_tensor = video_transform(frame_image)

        # Reservoir sampling: retain up to 16
        # approximately uniformly sampled frames.
        if len(frames) < MAX_FRAMES:
            frames.append(frame_tensor)

        else:
            replacement_index = int(
                rng.integers(0, total_read + 1)
            )

            if replacement_index < MAX_FRAMES:
                frames[replacement_index] = frame_tensor

        total_read += 1

    capture.release()

    print("Actually decoded frames:", total_read)
    print("Frames selected:", len(frames))

    if len(frames) == 0:
        raise ValueError("No video frames could be decoded.")

    return frames
