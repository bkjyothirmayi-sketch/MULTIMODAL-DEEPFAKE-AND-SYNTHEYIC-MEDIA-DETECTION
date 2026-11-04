
import torch
import numpy as np
import librosa

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ==========================================================
# 1. DATASET PATHS
# ==========================================================

IMAGE_DATASET_PATH = Path(
    r"D:\AI_Projects\archive\ddata"
)

AUDIO_DATASET_PATH = Path(
    r"D:\AI_Projects\audio dataset\deepfake_audio"
)

VIDEO_FRAME_DATASET_PATH = Path(
    r"D:\AI_Projects\archive (2)\deepfake_database"
)


# ==========================================================
# 2. CONFIGURATION
# ==========================================================

IMAGE_SIZE = 224

AUDIO_SAMPLE_RATE = 16000
AUDIO_DURATION = 3
AUDIO_LENGTH = AUDIO_SAMPLE_RATE * AUDIO_DURATION

BATCH_SIZE = 2
NUM_WORKERS = 0

LABEL_MAP = {
    "real": 0,
    "fake": 1
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


# ==========================================================
# 3. IMAGE TRANSFORMATION
# ==========================================================

image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def preprocess_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image = image_transform(image)

    return image


# ==========================================================
# 4. AUDIO PREPROCESSING
# ==========================================================

def preprocess_audio(audio_path):

    audio, sr = librosa.load(
        str(audio_path),
        sr=AUDIO_SAMPLE_RATE,
        mono=True
    )

    # Fixed audio length
    if len(audio) < AUDIO_LENGTH:

        padding = AUDIO_LENGTH - len(audio)

        audio = np.pad(
            audio,
            (0, padding),
            mode="constant"
        )

    else:

        audio = audio[:AUDIO_LENGTH]

    # Normalize audio
    max_value = np.max(np.abs(audio))

    if max_value > 0:
        audio = audio / max_value

    return torch.tensor(
        audio,
        dtype=torch.float32
    )


# ==========================================================
# 5. LABEL DETECTION
# ==========================================================

def get_label(file_path):

    file_path = Path(file_path)

    # Check parent folder names, including nested folders
    for folder in reversed(file_path.parts[:-1]):

        folder = folder.lower().strip()

        if folder in LABEL_MAP:
            return LABEL_MAP[folder]

    raise ValueError(
        f"Cannot identify REAL/FAKE label for: {file_path}\n"
        "Make sure the image/audio file is inside a folder "
        "named real or fake."
    )


# ==========================================================
# 6. IMAGE DEEPFAKE DATASET
# ==========================================================

class ImageDeepfakeDataset(Dataset):

    def __init__(self, root):

        self.root = Path(root)

        if not self.root.exists():
            raise FileNotFoundError(
                f"Image dataset folder not found: {self.root}"
            )

        self.samples = []

        for file_path in self.root.rglob("*"):

            if (
                file_path.is_file()
                and file_path.suffix.lower() in IMAGE_EXTENSIONS
            ):

                label = get_label(file_path)

                self.samples.append({
                    "path": file_path,
                    "label": label
                })

        if len(self.samples) == 0:
            raise ValueError(
                f"No image files found in {self.root}"
            )

        print("\n========== IMAGE DATASET ==========")
        print("Total images:", len(self.samples))
        print(
            "Real images:",
            sum(x["label"] == 0 for x in self.samples)
        )
        print(
            "Fake images:",
            sum(x["label"] == 1 for x in self.samples)
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        image = preprocess_image(sample["path"])

        label = torch.tensor(
            sample["label"],
            dtype=torch.long
        )

        return {
            "image": image,
            "label": label,
            "path": str(sample["path"])
        }


# ==========================================================
# 7. AUDIO DEEPFAKE DATASET
# ==========================================================

class AudioDeepfakeDataset(Dataset):

    def __init__(self, root):

        self.root = Path(root)

        if not self.root.exists():
            raise FileNotFoundError(
                f"Audio dataset folder not found: {self.root}"
            )

        self.samples = []

        for file_path in self.root.rglob("*"):

            if (
                file_path.is_file()
                and file_path.suffix.lower() in AUDIO_EXTENSIONS
            ):

                label = get_label(file_path)

                self.samples.append({
                    "path": file_path,
                    "label": label
                })

        if len(self.samples) == 0:
            raise ValueError(
                f"No audio files found in {self.root}"
            )

        print("\n========== AUDIO DATASET ==========")
        print("Total audio files:", len(self.samples))
        print(
            "Real audio:",
            sum(x["label"] == 0 for x in self.samples)
        )
        print(
            "Fake audio:",
            sum(x["label"] == 1 for x in self.samples)
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        audio = preprocess_audio(sample["path"])

        label = torch.tensor(
            sample["label"],
            dtype=torch.long
        )

        return {
            "audio": audio,
            "label": label,
            "path": str(sample["path"])
        }


# ==========================================================
# 8. VIDEO FRAME DATASET
# IMPORTANT:
# Every extracted frame is treated as an individual image.
# No video grouping or original video files are required.
# ==========================================================

class VideoFrameDeepfakeDataset(Dataset):

    def __init__(self, root):

        self.root = Path(root)

        if not self.root.exists():
            raise FileNotFoundError(
                f"Video frame dataset folder not found: {self.root}"
            )

        self.samples = []

        # Recursively find every extracted frame
        for file_path in self.root.rglob("*"):

            if (
                file_path.is_file()
                and file_path.suffix.lower() in IMAGE_EXTENSIONS
            ):

                # Label comes from a parent folder named real/fake
                label = get_label(file_path)

                self.samples.append({
                    "path": file_path,
                    "label": label
                })

        if len(self.samples) == 0:
            raise ValueError(
                f"No extracted image frames found in: {self.root}"
            )

        print("\n========== VIDEO FRAME DATASET ==========")
        print("Total extracted frames:", len(self.samples))
        print(
            "Real frames:",
            sum(x["label"] == 0 for x in self.samples)
        )
        print(
            "Fake frames:",
            sum(x["label"] == 1 for x in self.samples)
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        # Each extracted frame is loaded as an image
        frame = preprocess_image(sample["path"])

        label = torch.tensor(
            sample["label"],
            dtype=torch.long
        )

        return {
            "video": frame,
            "label": label,
            "frame_path": str(sample["path"])
        }


# ==========================================================
# 9. CREATE THREE INDEPENDENT DATASETS
# ==========================================================

if __name__ == "__main__":

    image_dataset = ImageDeepfakeDataset(
        IMAGE_DATASET_PATH
    )

    audio_dataset = AudioDeepfakeDataset(
        AUDIO_DATASET_PATH
    )

    video_dataset = VideoFrameDeepfakeDataset(
        VIDEO_FRAME_DATASET_PATH
    )


    # ======================================================
    # 10. THREE INDEPENDENT DATALOADERS
    # ======================================================

    image_loader = DataLoader(
        image_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    audio_loader = DataLoader(
        audio_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    video_loader = DataLoader(
        video_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )


    # ======================================================
    # 11. TEST IMAGE DATALOADER
    # ======================================================

    image_batch = next(iter(image_loader))

    print("\n========== IMAGE LOADER TEST ==========")
    print("Image shape:", image_batch["image"].shape)
    print("Image labels:", image_batch["label"].shape)


    # ======================================================
    # 12. TEST AUDIO DATALOADER
    # ======================================================

    audio_batch = next(iter(audio_loader))

    print("\n========== AUDIO LOADER TEST ==========")
    print("Audio shape:", audio_batch["audio"].shape)
    print("Audio labels:", audio_batch["label"].shape)


    # ======================================================
    # 13. TEST VIDEO FRAME DATALOADER
    # ======================================================

    video_batch = next(iter(video_loader))

    print("\n========== VIDEO FRAME LOADER TEST ==========")
    print("Video frame shape:", video_batch["video"].shape)
    print("Video labels:", video_batch["label"].shape)

    print("\nFirst frame path:")
    print(video_batch["frame_path"][0])


    print("\n========== ALL DATASETS LOADED SUCCESSFULLY ==========")
