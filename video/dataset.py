
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==========================================
# 1. CONFIGURATION
# ==========================================

DATASET_DIR = Path(
    r"D:\AI_Projects\archive (2)\deepfake_database"
)

IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 0

SEED = 42

# Label mapping
CLASS_NAMES = {
    "fake": 0,
    "real": 1
}

SUPPORTED_FORMATS = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".webp"
}


# ==========================================
# 2. REPRODUCIBILITY
# ==========================================

def set_seed():

    random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


# ==========================================
# 3. IMAGE TRANSFORMATIONS
# ==========================================

train_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(5),

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


val_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# 4. FIND SPLIT FOLDERS
# ==========================================

def find_split_folder(split_name):

    # Support validation and val naming
    alternatives = {
        "train": ["train"],
        "validation": ["validation", "val"],
        "test": ["test"]
    }

    for name in alternatives[split_name]:

        folder = DATASET_DIR / name

        if folder.is_dir():
            return folder

    raise FileNotFoundError(
        f"Cannot find {split_name} folder in {DATASET_DIR}"
    )


# ==========================================
# 5. COLLECT IMAGES
# ==========================================

def collect_images(split_folder):

    records = []

    for class_name, label in CLASS_NAMES.items():

        class_folder = split_folder / class_name

        if not class_folder.exists():

            raise FileNotFoundError(
                f"Missing class folder: {class_folder}"
            )

        image_paths = sorted([

            p for p in class_folder.rglob("*")

            if p.is_file()
            and p.suffix.lower() in SUPPORTED_FORMATS

        ])

        for image_path in image_paths:

            records.append({

                "image": image_path,
                "label": label,
                "class_name": class_name

            })

    return records


# ==========================================
# 6. PYTORCH DATASET CLASS
# ==========================================

class DeepfakeFrameDataset(Dataset):

    def __init__(self, records, transform=None):

        self.records = records
        self.transform = transform

    def __len__(self):

        return len(self.records)

    def __getitem__(self, index):

        record = self.records[index]

        image_path = record["image"]

        label = record["label"]

        try:

            image = Image.open(image_path).convert("RGB")

            if self.transform:

                image = self.transform(image)

            return (
                image,
                torch.tensor(label, dtype=torch.long)
            )

        except Exception as error:

            raise RuntimeError(
                f"Image loading failed: {image_path}\n{error}"
            )


# ==========================================
# 7. PRINT DATASET INFORMATION
# ==========================================

def print_dataset_summary(name, records):

    fake_count = sum(
        r["label"] == 0 for r in records
    )

    real_count = sum(
        r["label"] == 1 for r in records
    )

    print(f"\n========== {name.upper()} DATASET ==========")

    print("Total images :", len(records))
    print("Fake images  :", fake_count)
    print("Real images  :", real_count)

    if fake_count == 0 or real_count == 0:

        raise ValueError(
            f"{name} must contain both fake and real images."
        )


# ==========================================
# 8. CREATE DATALOADERS
# ==========================================

def get_dataloaders():

    set_seed()

    print("\n========== DEEPFAKE DATASET LOADING ==========")

    print("Dataset path:", DATASET_DIR)

    if not DATASET_DIR.exists():

        raise FileNotFoundError(
            f"Dataset directory does not exist:\n{DATASET_DIR}\n"
            "Check the exact spelling of your Windows folder path."
        )

    # Locate existing splits
    train_folder = find_split_folder("train")

    val_folder = find_split_folder("validation")

    test_folder = find_split_folder("test")

    # Collect images
    train_records = collect_images(train_folder)

    val_records = collect_images(val_folder)

    test_records = collect_images(test_folder)

    if not train_records:
        raise RuntimeError("No training images found.")

    if not val_records:
        raise RuntimeError("No validation images found.")

    if not test_records:
        raise RuntimeError("No testing images found.")

    # Print statistics
    print_dataset_summary("Train", train_records)

    print_dataset_summary("Validation", val_records)

    print_dataset_summary("Test", test_records)

    # Create datasets
    train_dataset = DeepfakeFrameDataset(
        train_records,
        transform=train_transform
    )

    val_dataset = DeepfakeFrameDataset(
        val_records,
        transform=val_transform
    )

    test_dataset = DeepfakeFrameDataset(
        test_records,
        transform=val_transform
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    print("\n========== DATALOADER SUMMARY ==========")

    print("Training batches  :", len(train_loader))

    print("Validation batches:", len(val_loader))

    print("Testing batches   :", len(test_loader))

    print("\nDataset loaded successfully!")

    return train_loader, val_loader, test_loader


# ==========================================
# 9. TEST DATASET PIPELINE
# ==========================================

if __name__ == "__main__":

    train_loader, val_loader, test_loader = (
        get_dataloaders()
    )

    images, labels = next(iter(train_loader))

    print("\n========== BATCH VERIFICATION ==========")

    print("Image shape:", images.shape)

    print("Label shape:", labels.shape)

    print("Labels:", labels.tolist())

    print("Class mapping:", CLASS_NAMES)

    print("\nDataset pipeline working successfully!")
