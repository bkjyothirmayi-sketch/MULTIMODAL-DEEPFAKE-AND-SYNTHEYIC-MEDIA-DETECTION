
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ==========================================
# 1. DATASET CONFIGURATION
# ==========================================

DATASET_ROOT = Path(
    r"D:\AI_Projects\archive\ddata"
)

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0  # Windows compatibility

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


# ==========================================
# 2. IMAGE TRANSFORMATIONS
# ==========================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(degrees=10),

    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# 3. VALIDATE DATASET FOLDERS
# ==========================================

def validate_dataset_folder(folder_path):

    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {folder_path}"
        )

    required_classes = {"real", "fake"}

    available_classes = {
        folder.name.lower()
        for folder in folder_path.iterdir()
        if folder.is_dir()
    }

    if not required_classes.issubset(available_classes):
        raise ValueError(
            f"{folder_path} must contain real and fake folders."
        )

    for class_name in required_classes:

        class_path = folder_path / class_name

        images = [
            file for file in class_path.rglob("*")
            if file.is_file()
            and file.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        if len(images) == 0:
            raise ValueError(
                f"No images found in {class_path}"
            )


# ==========================================
# 4. CREATE IMAGE DATASETS
# ==========================================

def create_datasets():

    train_path = DATASET_ROOT / "train"
    test_path = DATASET_ROOT / "test"

    validate_dataset_folder(train_path)
    validate_dataset_folder(test_path)

    train_dataset = datasets.ImageFolder(
        root=str(train_path),
        transform=train_transform
    )

    test_dataset = datasets.ImageFolder(
        root=str(test_path),
        transform=test_transform
    )

    if train_dataset.class_to_idx != test_dataset.class_to_idx:
        raise ValueError(
            "Train and test class mappings do not match."
        )

    print("\n========== DATASET INFORMATION ==========")

    print("Classes:", train_dataset.classes)

    print(
        "Class mapping:",
        train_dataset.class_to_idx
    )

    print(
        "Training images:",
        len(train_dataset)
    )

    print(
        "Testing images:",
        len(test_dataset)
    )

    print("\nTraining class distribution:")

    for class_name, class_index in train_dataset.class_to_idx.items():

        count = sum(
            1 for _, label in train_dataset.samples
            if label == class_index
        )

        print(f"{class_name}: {count}")

    print("\nTesting class distribution:")

    for class_name, class_index in test_dataset.class_to_idx.items():

        count = sum(
            1 for _, label in test_dataset.samples
            if label == class_index
        )

        print(f"{class_name}: {count}")

    return train_dataset, test_dataset


# ==========================================
# 5. CREATE DATALOADERS
# ==========================================

def create_dataloaders():

    train_dataset, test_dataset = create_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, test_loader


# ==========================================
# 6. TEST DATASET LOADING
# ==========================================

if __name__ == "__main__":

    print("Loading Deepfake Image Dataset...")

    train_loader, test_loader = create_dataloaders()

    print("\nDataset loaded successfully!")

    images, labels = next(iter(train_loader))

    print("\nTraining batch information:")

    print("Image shape:", images.shape)
    print("Label shape:", labels.shape)

    print("\nLabel mapping:")
    print("Real = 0")
    print("Fake = 1")

    print("\nDataset loading completed.")
