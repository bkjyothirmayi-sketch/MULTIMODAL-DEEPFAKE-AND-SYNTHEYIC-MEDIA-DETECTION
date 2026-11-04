
# ============================================================
# AUDIO FEATURE EXTRACTION
# Raw waveform -> Mel spectrogram -> MobileNetV2 -> 1280 features
# ============================================================

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio

from torch.utils.data import DataLoader
from torchvision import models


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATASET_PATH = (
    r"D:\AI_Projects\audio dataset\deepfake_audio"
)

CHECKPOINT_PATH = (
    r"D:\AI_Projects\audio dataset\model_save_path\best_audio_model.pth"
)

OUTPUT_DIR = r"D:\AI_Projects\extracted_features"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "audio_features.npz"
)

BATCH_SIZE = 8
NUM_WORKERS = 0

SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128

IMAGE_SIZE = 224

DEVICE = torch.device("cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. CHECK PATHS
# ============================================================

if not os.path.isdir(DATASET_PATH):
    raise FileNotFoundError(DATASET_PATH)

if not os.path.isfile(CHECKPOINT_PATH):
    raise FileNotFoundError(CHECKPOINT_PATH)


# ============================================================
# 3. IMPORT YOUR AUDIO DATASET
# ============================================================

from dataset import AudioDeepfakeDataset

print("Loading audio dataset...")

dataset = AudioDeepfakeDataset(
    root=DATASET_PATH
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

print("Total audio files:", len(dataset))


# ============================================================
# 4. LOAD TRAINED CHECKPOINT
# ============================================================

print("\nLoading trained MobileNetV2...")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

state_dict = checkpoint["model_state_dict"]

class_mapping = checkpoint.get(
    "class_mapping",
    {"fake": 0, "real": 1}
)

print("Class mapping:", class_mapping)


# ============================================================
# 5. REBUILD MOBILENETV2
# ============================================================

model = models.mobilenet_v2(weights=None)

model.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(1280, 256),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.3),
    nn.Linear(256, 2)
)

model.load_state_dict(
    state_dict,
    strict=True
)

model.to(DEVICE)
model.eval()

print("Trained model loaded successfully.")


# ============================================================
# 6. MEL SPECTROGRAM TRANSFORM
# ============================================================

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    power=2.0
)

db_transform = torchaudio.transforms.AmplitudeToDB(
    stype="power"
)


# ============================================================
# 7. AUDIO PREPROCESSING
# ============================================================

def waveform_to_image(audio):

    # Audio shape: (batch, 48000)

    audio = audio.float()

    # Convert to Mel spectrogram
    mel = mel_transform(audio)

    # Convert to decibel scale
    mel = db_transform(mel)

    # Normalize each spectrogram
    mean = mel.mean(dim=(1, 2), keepdim=True)
    std = mel.std(dim=(1, 2), keepdim=True)

    mel = (mel - mean) / (std + 1e-6)

    # Shape: (batch, 1, n_mels, time)

    # Resize to 224 x 224
    mel = F.interpolate(
        mel.unsqueeze(1),
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False
    )

    # Convert grayscale to 3 channels
    mel = mel.repeat(1, 3, 1, 1)

    return mel


# ============================================================
# 8. FEATURE EXTRACTOR
# ============================================================

class AudioFeatureExtractor(nn.Module):

    def __init__(self, trained_model):

        super().__init__()

        self.features = trained_model.features

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):

        x = self.features(x)

        x = self.pool(x)

        x = torch.flatten(x, 1)

        return x


feature_model = AudioFeatureExtractor(model)

feature_model.to(DEVICE)
feature_model.eval()

print("Feature extractor ready.")


# ============================================================
# 9. EXTRACT FEATURES
# ============================================================

all_features = []
all_labels = []
all_paths = []

print("\nStarting audio feature extraction...")

with torch.inference_mode():

    for batch_index, batch in enumerate(loader):

        # Dataset returns dictionary
        audio = batch["audio"]
        labels = batch["label"]
        paths = batch["path"]

        audio = audio.to(DEVICE)

        # Convert raw waveform into Mel spectrogram
        images = waveform_to_image(audio)

        images = images.to(DEVICE)

        # Extract 1280-dimensional features
        features = feature_model(images)

        all_features.append(
            features.cpu().numpy()
        )

        all_labels.append(
            labels.cpu().numpy()
        )

        all_paths.extend(list(paths))

        if (batch_index + 1) % 20 == 0:

            print(
                f"Processed {batch_index + 1}/{len(loader)} batches"
            )


# ============================================================
# 10. COMBINE FEATURES
# ============================================================

features_array = np.concatenate(
    all_features,
    axis=0
)

labels_array = np.concatenate(
    all_labels,
    axis=0
)

paths_array = np.asarray(
    all_paths,
    dtype=str
)


# ============================================================
# 11. VALIDATE
# ============================================================

print("\nValidating features...")

print("Features:", features_array.shape)
print("Labels:", labels_array.shape)
print("Paths:", paths_array.shape)

if features_array.shape[1] != 1280:
    raise ValueError("Unexpected audio feature dimension.")

if not (
    len(features_array)
    == len(labels_array)
    == len(paths_array)
):
    raise ValueError("Feature count mismatch.")


# ============================================================
# 12. SAVE
# ============================================================

np.savez_compressed(
    OUTPUT_FILE,
    features=features_array,
    labels=labels_array,
    paths=paths_array,
    class_mapping=np.array(class_mapping, dtype=object)
)

print("\n========================================")
print("AUDIO FEATURE EXTRACTION COMPLETED")
print("========================================")

print("Total samples:", len(features_array))
print("Feature dimensions:", features_array.shape)
print("Saved location:", OUTPUT_FILE)
