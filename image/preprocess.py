
import os
import hashlib
from pathlib import Path
from PIL import Image, ImageFile

from tqdm import tqdm


# ==========================================
# 1. CONFIGURATION
# ==========================================

INPUT_DIR = Path(
    r"D:\AI_Projects\archive\ddata"
)

OUTPUT_DIR = Path(
    r"D:\AI_Projects\output _dir"
)

IMAGE_SIZE = (224, 224)

CLASSES = ["real", "fake"]

SUPPORTED_FORMATS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

ImageFile.LOAD_TRUNCATED_IMAGES = False


# ==========================================
# 2. CREATE OUTPUT DIRECTORIES
# ==========================================

def create_output_directories():

    for split in ["train", "test"]:

        for class_name in CLASSES:

            folder = OUTPUT_DIR / split / class_name

            folder.mkdir(
                parents=True,
                exist_ok=True
            )

    print("Output directories created successfully.")


# ==========================================
# 3. CHECK IMAGE FILES
# ==========================================

def get_image_files(folder):

    return [
        file for file in folder.rglob("*")
        if file.is_file()
        and file.suffix.lower() in SUPPORTED_FORMATS
    ]


# ==========================================
# 4. GENERATE IMAGE HASH
# ==========================================

def get_image_hash(image):

    image_bytes = image.tobytes()

    return hashlib.md5(image_bytes).hexdigest()


# ==========================================
# 5. PREPROCESS SINGLE IMAGE
# ==========================================

def preprocess_image(image_path):

    try:

        with Image.open(image_path) as img:

            # Verify image
            img.verify()

        with Image.open(image_path) as img:

            # Convert image to RGB
            img = img.convert("RGB")

            # Resize image
            img = img.resize(
                IMAGE_SIZE,
                Image.Resampling.LANCZOS
            )

            return img.copy()

    except Exception as error:

        print(
            f"Skipping corrupted image: {image_path}"
        )

        return None


# ==========================================
# 6. PROCESS DATASET
# ==========================================

def process_split(split):

    print(f"\nProcessing {split} dataset...")

    total_images = 0
    processed_images = 0
    corrupted_images = 0
    duplicate_images = 0

    seen_hashes = set()

    for class_name in CLASSES:

        input_folder = INPUT_DIR / split / class_name

        output_folder = OUTPUT_DIR / split / class_name

        if not input_folder.exists():

            print(
                f"Missing folder: {input_folder}"
            )

            continue

        image_files = get_image_files(input_folder)

        print(
            f"\n{class_name.upper()} images: {len(image_files)}"
        )

        for image_path in tqdm(image_files):

            total_images += 1

            image = preprocess_image(image_path)

            if image is None:

                corrupted_images += 1

                continue

            # Check duplicates within this split
            image_hash = get_image_hash(image)

            if image_hash in seen_hashes:

                duplicate_images += 1

                continue

            seen_hashes.add(image_hash)

            # Save preprocessed image
            output_path = (
                output_folder /
                f"{image_path.stem}.jpg"
            )

            # Avoid overwriting files with the same name
            counter = 1

            while output_path.exists():

                output_path = (
                    output_folder /
                    f"{image_path.stem}_{counter}.jpg"
                )

                counter += 1

            image.save(
                output_path,
                format="JPEG",
                quality=95
            )

            processed_images += 1

    print(f"\n{split.upper()} SUMMARY")

    print("Total images:", total_images)

    print("Processed images:", processed_images)

    print("Corrupted images:", corrupted_images)

    print("Duplicate images:", duplicate_images)


# ==========================================
# 7. PROCESS TRAIN AND TEST
# ==========================================

def preprocess_dataset():

    print("\nStarting Deepfake Dataset Preprocessing")

    create_output_directories()

    process_split("train")

    process_split("test")

    print("\nPreprocessing completed successfully!")

    print(
        f"\nProcessed dataset saved at: {OUTPUT_DIR}"
    )


# ==========================================
# 8. MAIN FUNCTION
# ==========================================

if __name__ == "__main__":

    preprocess_dataset()
