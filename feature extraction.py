
import os
import sys
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2
from tqdm import tqdm

# ==========================================================
# 1. CONFIGURATION
# ==========================================================

DEVICE = torch.device("cpu")
BATCH_SIZE = 2

PROJECT_DIR = r"D:\AI_Projects"

IMAGE_DATASET_PATH = r"D:\AI_Projects\archive\ddata"

AUDIO_DATASET_PATH = (
    r"D:\AI_Projects\audio dataset\deepfake_audio"
)

VIDEO_DATASET_PATH = (
    r"D:\AI_Projects\archive (2)\deepfake_database"
)

IMAGE_MODEL_PATH = (
    r"D:\AI_Projects\mobilenet_results\best_mobilenet.pth"
)

AUDIO_MODEL_PATH = (
    r"D:\AI_Projects\audio dataset\model_save_path\best_audio_model.pth"
)

VIDEO_MODEL_PATH = (
    r"D:\AI_Projects\archive (2)\saved_models\shufflenet_video_best_v3.pth"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "extracted_features"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# 2. IMPORT YOUR DATASET CLASSES
# ==========================================================

sys.path.insert(0, PROJECT_DIR)

from dataset import (
    ImageDeepfakeDataset,
    AudioDeepfakeDataset,
    VideoFrameDeepfakeDataset
)

# ==========================================================
# 3. LOAD CHECKPOINT STATE DICTIONARY
# ==========================================================

def get_state_dict(checkpoint):

    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Unsupported checkpoint format: "
            + str(type(checkpoint))
        )

    for key in [
        "model_state_dict",
        "state_dict",
        "model_weights",
        "weights"
    ]:
        if key in checkpoint:
            if isinstance(checkpoint[key], nn.Module):
                return checkpoint[key].state_dict()

            if isinstance(checkpoint[key], dict):
                return checkpoint[key]

    return checkpoint


def clean_state_dict(state_dict):

    cleaned = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key.replace("module.", "", 1)

        cleaned[key] = value

    return cleaned


def read_checkpoint(path):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    state_dict = get_state_dict(checkpoint)

    return clean_state_dict(state_dict), checkpoint


# ==========================================================
# 4. BUILD TRAINED MOBILENETV2
# ==========================================================

def build_image_model():

    print("\nLoading trained MobileNetV2...")

    model = mobilenet_v2(weights=None)

    # Exact classifier confirmed by your checkpoint
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(1280, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, 2)
    )

    state_dict, checkpoint = read_checkpoint(
        IMAGE_MODEL_PATH
    )

    # Load complete trained model first
    model.load_state_dict(
        state_dict,
        strict=True
    )

    # Remove classification head for feature extraction
    model.classifier = nn.Identity()

    model.to(DEVICE)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    print("MobileNetV2 loaded successfully.")
    print("Feature dimension: 1280")

    return model


# ==========================================================
# 5. LOAD IMAGE DATASET
# ==========================================================

def load_image_dataset():

    print("\nLoading image dataset...")

    dataset = ImageDeepfakeDataset(
        root=IMAGE_DATASET_PATH
    )

    print("Image samples:", len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return loader


# ==========================================================
# 6. IMAGE FEATURE EXTRACTION
# ==========================================================

def extract_image_features(model, loader):

    print("\nExtracting image features...")

    all_features = []
    all_labels = []
    all_paths = []

    with torch.no_grad():

        for batch in tqdm(loader):

            # Expected dataset output:
            # (images, labels, paths)

            if isinstance(batch, dict):

                images = batch.get("image")
                labels = batch.get("label")
                paths = batch.get("path")

                if images is None:
                    images = batch.get("images")

                if paths is None:
                    paths = batch.get("image_path")

            else:

                images, labels, paths = batch

            images = images.to(DEVICE)

            features = model(images)

            features = features.cpu().numpy()

            all_features.append(features)

            all_labels.extend(
                labels.cpu().numpy().tolist()
            )

            if isinstance(paths, (list, tuple)):
                all_paths.extend(
                    [str(p) for p in paths]
                )
            else:
                all_paths.extend(
                    [str(p) for p in paths]
                )

    all_features = np.concatenate(
        all_features,
        axis=0
    )

    all_labels = np.asarray(all_labels)

    all_paths = np.asarray(
        all_paths,
        dtype=str
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        "image_features.npz"
    )

    np.savez_compressed(
        output_path,
        features=all_features,
        labels=all_labels,
        paths=all_paths
    )

    print("\nImage extraction completed.")

    print("Feature shape:", all_features.shape)
    print("Label shape:", all_labels.shape)
    print("Path count:", len(all_paths))

    print("Saved:", output_path)

    return all_features, all_labels, all_paths


# ==========================================================
# 7. INSPECT AUDIO AND VIDEO CHECKPOINTS
# ==========================================================

def inspect_checkpoint(name, path):

    print("\n" + "=" * 60)
    print(name, "CHECKPOINT INSPECTION")
    print("=" * 60)

    state_dict, checkpoint = read_checkpoint(path)

    if isinstance(checkpoint, nn.Module):

        print("Complete model object found:")
        print(checkpoint)

    else:

        print("Checkpoint tensor keys and shapes:")

        for key, value in state_dict.items():

            if torch.is_tensor(value):

                print(
                    key,
                    tuple(value.shape)
                )

    print("\nTotal parameter tensors:", len(state_dict))


# ==========================================================
# 8. CHECK DATASET SIZES
# ==========================================================

def inspect_datasets():

    print("\nChecking audio and video datasets...")

    audio_dataset = AudioDeepfakeDataset(
        root=AUDIO_DATASET_PATH
    )

    video_dataset = VideoFrameDeepfakeDataset(
        root=VIDEO_DATASET_PATH
    )

    print("Audio samples:", len(audio_dataset))
    print("Video frame samples:", len(video_dataset))

    print(
        "\nNote: Video dataset contains extracted frames."
    )

    return audio_dataset, video_dataset


# ==========================================================
# 9. MAIN
# ==========================================================

def main():

    print("=" * 60)
    print("MULTIMODAL DEEPFAKE FEATURE EXTRACTION")
    print("=" * 60)

    # Check dataset classes
    inspect_datasets()

    # Extract image features
    image_loader = load_image_dataset()

    image_model = build_image_model()

    extract_image_features(
        image_model,
        image_loader
    )

    # Inspect remaining trained models
    inspect_checkpoint(
        "AUDIO",
        AUDIO_MODEL_PATH
    )

    inspect_checkpoint(
        "VIDEO SHUFFLENET",
        VIDEO_MODEL_PATH
    )

    print("\n" + "=" * 60)
    print("IMAGE FEATURE EXTRACTION FINISHED")
    print("Audio/video architecture inspection finished.")
    print("Output folder:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
