
import os
import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import shufflenet_v2_x1_0


# =========================================================
# 1. CONFIGURATION
# =========================================================

DATASET_PATH = (
    r"D:\AI_Projects\archive (2)\deepfake_database"
)

CHECKPOINT_PATH = (
    r"D:\AI_Projects\archive (2)\saved_models\shufflenet_video_best_v3.pth"
)

OUTPUT_DIR = r"D:\AI_Projects\extracted_features"

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0

DEVICE = torch.device("cpu")

SPLITS = ["train", "test", "validation"]

CLASS_MAPPING = {
    "fake": 0,
    "real": 1
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# 2. PREPROCESSING
# =========================================================
# Provisional preprocessing: resize + ToTensor.
# Confirm against the original training pipeline if possible.

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


# =========================================================
# 3. VIDEO FRAME DATASET
# =========================================================

class VideoFrameDataset(Dataset):

    def __init__(self, root, split, transform=None):

        self.root = root
        self.split = split
        self.transform = transform
        self.samples = []

        split_dir = os.path.join(root, split)

        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"Dataset split not found: {split_dir}"
            )

        for class_name, label in CLASS_MAPPING.items():

            class_dir = os.path.join(
                split_dir, class_name
            )

            if not os.path.isdir(class_dir):
                raise FileNotFoundError(
                    f"Class folder not found: {class_dir}"
                )

            for current_root, _, files in os.walk(class_dir):

                for filename in files:

                    if filename.lower().endswith(
                        (".jpg", ".jpeg", ".png", ".bmp")
                    ):

                        image_path = os.path.join(
                            current_root, filename
                        )

                        self.samples.append(
                            (image_path, label)
                        )

        if not self.samples:
            raise RuntimeError(
                f"No image frames found in {split_dir}"
            )

        fake_count = sum(
            label == 0 for _, label in self.samples
        )

        real_count = sum(
            label == 1 for _, label in self.samples
        )

        print(f"\n{split.upper()} DATASET")
        print("Total frames:", len(self.samples))
        print("Fake frames:", fake_count)
        print("Real frames:", real_count)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        try:

            image = Image.open(image_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

        except Exception as error:

            raise RuntimeError(
                f"Unable to load frame: {image_path}"
            ) from error

        return image, label, image_path


# =========================================================
# 4. LOAD TRAINED SHUFFLENETV2 MODEL
# =========================================================

def load_video_model():

    print("\nLoading trained ShuffleNetV2...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False
    )

    model = shufflenet_v2_x1_0(weights=None)

    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(1024, 2)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model.to(DEVICE)
    model.eval()

    print("Video model loaded successfully.")

    print("Checkpoint class mapping:",
          checkpoint.get("class_mapping"))

    return model


# =========================================================
# 5. EXTRACT 1024-D FEATURES
# =========================================================

def extract_features(model, loader, split):

    all_features = []
    all_labels = []
    all_paths = []

    print(f"\nExtracting {split} features...")

    with torch.inference_mode():

        for batch_index, (images, labels, paths) in enumerate(loader):

            images = images.to(DEVICE)

            # ShuffleNetV2 feature extraction
            x = model.conv1(images)
            x = model.maxpool(x)
            x = model.stage2(x)
            x = model.stage3(x)
            x = model.stage4(x)
            x = model.conv5(x)

            # Global average pooling
            features = torch.mean(
                x, dim=(2, 3)
            )

            all_features.append(
                features.cpu().numpy()
            )

            all_labels.append(
                labels.numpy()
            )

            all_paths.extend(paths)

            if (batch_index + 1) % 50 == 0:

                print(
                    f"{split}: Processed "
                    f"{batch_index + 1}/{len(loader)} batches"
                )

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    paths = np.asarray(all_paths)

    return features, labels, paths


# =========================================================
# 6. SAVE FEATURES
# =========================================================

def save_features(features, labels, paths, split):

    output_file = os.path.join(
        OUTPUT_DIR,
        f"video_{split}_features.npz"
    )

    np.savez_compressed(
        output_file,
        features=features,
        labels=labels,
        paths=paths
    )

    print(f"\n{split.upper()} FEATURES SAVED")
    print("Feature shape:", features.shape)
    print("Label shape:", labels.shape)
    print("Path count:", len(paths))
    print("File:", output_file)


# =========================================================
# 7. MAIN
# =========================================================

def main():

    print("=" * 60)
    print("VIDEO FEATURE EXTRACTION")
    print("MODEL: ShuffleNetV2")
    print("FEATURE DIMENSION: 1024")
    print("=" * 60)

    # Load model only once
    model = load_video_model()

    total_frames = 0

    for split in SPLITS:

        print("\n" + "=" * 60)
        print("PROCESSING:", split.upper())
        print("=" * 60)

        dataset = VideoFrameDataset(
            root=DATASET_PATH,
            split=split,
            transform=transform
        )

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=False
        )

        features, labels, paths = extract_features(
            model,
            loader,
            split
        )

        save_features(
            features,
            labels,
            paths,
            split
        )

        total_frames += len(labels)

        del dataset, loader, features, labels, paths

    print("\n" + "=" * 60)
    print("ALL VIDEO FEATURE EXTRACTION COMPLETED")
    print("=" * 60)

    print("Total frames processed:", total_frames)
    print("Output directory:", OUTPUT_DIR)

    print("\nGenerated files:")
    for split in SPLITS:
        print(f"video_{split}_features.npz")


if __name__ == "__main__":
    main()
