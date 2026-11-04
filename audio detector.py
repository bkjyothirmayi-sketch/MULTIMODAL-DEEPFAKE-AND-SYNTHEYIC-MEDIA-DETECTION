
import os
import importlib.util

import torch
import librosa
import numpy as np

from PIL import Image
from torchvision import transforms


# ============================================================
# PATHS AND SETTINGS
# ============================================================

DEVICE = torch.device("cpu")

ARCHITECTURE_PATH = (
    r"D:\AI_Projects\intermediate fusion\model architecture.py"
)

AUDIO_MODEL_PATH = (
    r"D:\AI_Projects\audio dataset\model_save_path\best_audio_model.pth"
)

SAMPLE_RATE = 22050
IMAGE_SIZE = 224


# ============================================================
# LOAD MODEL ARCHITECTURE
# ============================================================

spec = importlib.util.spec_from_file_location(
    "model_architecture",
    ARCHITECTURE_PATH
)

architecture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(architecture)

create_audio_model = architecture.create_audio_model
load_checkpoint = architecture.load_checkpoint


# ============================================================
# AUDIO TRANSFORMATION
# ============================================================

audio_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# CONVERT AUDIO TO MEL SPECTROGRAM
# ============================================================

def audio_to_spectrogram(audio_path):

    if not os.path.isfile(audio_path):
        raise FileNotFoundError(
            "Audio file not found: " + audio_path
        )

    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError("The audio file is empty.")

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    mel_db = np.clip(mel_db, -80, 0)

    mel_normalized = (
        (mel_db + 80) / 80 * 255
    ).astype(np.uint8)

    spectrogram_image = Image.fromarray(
        mel_normalized
    ).convert("RGB")

    return spectrogram_image


# ============================================================
# LOAD AUDIO DETECTOR
# ============================================================

def load_audio_detector():

    if not os.path.isfile(AUDIO_MODEL_PATH):
        raise FileNotFoundError(
            "Audio checkpoint not found: "
            + AUDIO_MODEL_PATH
        )

    model = create_audio_model()

    model, checkpoint = load_checkpoint(
        model,
        AUDIO_MODEL_PATH
    )

    model = model.to(DEVICE)
    model.eval()

    class_mapping = checkpoint.get(
        "class_mapping",
        {"fake": 0, "real": 1}
    )

    idx_to_class = {
        int(index): str(label).lower()
        for label, index in class_mapping.items()
    }

    if set(idx_to_class.values()) != {"fake", "real"}:
        raise ValueError(
            "Unexpected audio class mapping: "
            + str(class_mapping)
        )

    print("\nAUDIO MODEL LOADED")
    print("Class mapping:", class_mapping)

    return model, idx_to_class


# ============================================================
# PREDICT AUDIO
# ============================================================

def predict_audio(audio_path, model, idx_to_class):

    spectrogram = audio_to_spectrogram(
        audio_path
    )

    image_tensor = audio_transform(
        spectrogram
    ).unsqueeze(0).to(DEVICE)

    model.eval()

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

    predicted_index = int(
        torch.argmax(probabilities).item()
    )

    prediction = idx_to_class[predicted_index]

    fake_index = next(
        index for index, label in idx_to_class.items()
        if label == "fake"
    )

    real_index = next(
        index for index, label in idx_to_class.items()
        if label == "real"
    )

    fake_probability = float(
        probabilities[fake_index].item()
    )

    real_probability = float(
        probabilities[real_index].item()
    )

    confidence = float(
        probabilities[predicted_index].item()
    )

    print("\n========== AUDIO PREDICTION ==========")
    print("Predicted class:", prediction)
    print("Confidence:", round(confidence * 100, 2), "%")
    print("Fake probability:", round(fake_probability * 100, 2), "%")
    print("Real probability:", round(real_probability * 100, 2), "%")

    return {
        "Modality": "Audio",
        "Prediction": prediction,
        "Confidence": confidence,
        "Fake Probability": fake_probability,
        "Real Probability": real_probability
    }


# ============================================================
# TEST AUDIO DETECTOR
# ============================================================

if __name__ == "__main__":

    model, idx_to_class = load_audio_detector()

    audio_path = input(
        "\nEnter audio file path: "
    ).strip().strip('"')

    result = predict_audio(
        audio_path,
        model,
        idx_to_class
    )

    print("\nFINAL RESULT")
    for key, value in result.items():
        print(key, ":", value)
