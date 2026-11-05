
import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(r"D:\AI_Projects\audio dataset\deepfake_audio"))

DATASET_PATH = r"D:\AI_Projects\audio dataset\output_path"

IMAGE_SIZE = 224

CLASS_TO_IDX = {
    "fake": 0,
    "real": 1
}


# ==========================================
# 2. IMAGE TRANSFORMATION
# ==========================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])


# ==========================================
# 3. AUDIO SPECTROGRAM DATASET
# ==========================================

class AudioDataset(Dataset):

    def __init__(
        self,
        root_dir=DATASET_PATH,
        split="train",
        transform=transform
    ):

        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.samples = []

        split_path = os.path.join(
            self.root_dir,
            self.split
        )

        if not os.path.isdir(split_path):

            raise FileNotFoundError(
                "Dataset folder not found: "
                + split_path
            )

        # Read fake and real spectrogram images

        for class_name, label in CLASS_TO_IDX.items():

            class_path = os.path.join(
                split_path,
                class_name
            )

            if not os.path.isdir(class_path):

                raise FileNotFoundError(
                    "Class folder not found: "
                    + class_path
                )

            for filename in sorted(os.listdir(class_path)):

                if filename.lower().endswith(
                    (".png", ".jpg", ".jpeg")
                ):

                    image_path = os.path.join(
                        class_path,
                        filename
                    )

                    self.samples.append(
                        (image_path, label)
                    )

        if len(self.samples) == 0:

            raise ValueError(
                "No spectrogram images found in "
                + split_path
            )

        print(
            f"{self.split.upper()} dataset loaded: "
            f"{len(self.samples)} images"
        )


    # ======================================
    # 4. DATASET LENGTH
    # ======================================

    def __len__(self):

        return len(self.samples)


    # ======================================
    # 5. LOAD IMAGE AND LABEL
    # ======================================

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:

            image = self.transform(image)

        return (
            image,
            torch.tensor(
                label,
                dtype=torch.long
            )
        )


# ==========================================
# 6. TEST DATASET LOADING
# ==========================================

if __name__ == "__main__":

    train_dataset = AudioDataset(
        split="train"
    )

    test_dataset = AudioDataset(
        split="test"
    )

    print("\nAudio Dataset Summary")
    print("---------------------")

    print("Training images:", len(train_dataset))
    print("Testing images:", len(test_dataset))

    image, label = train_dataset[0]

    print("Image shape:", image.shape)
    print("Label:", label.item())

    print("\nClass mapping:", CLASS_TO_IDX)
