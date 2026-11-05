import os
from pathlib import Path
from PIL import Image, ImageFile, ImageOps
from tqdm import tqdm
from collections import Counter

# ==========================================
# 1. CONFIGURATION
# ==========================================

SOURCE_DIR = Path(
    r"D:\AI_Projects\archive (2)\deepfake_database"
)

OUTPUT_DIR = Path(
    r"D:\AI_Projects\multimodal-deepfake-detection\video module\output_dir"
)

IMAGE_SIZE = (224, 224)

SUPPORTED_FORMATS = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".webp"
}

ImageFile.LOAD_TRUNCATED_IMAGES = True

SPLITS = ["train", "validation", "test"]

CLASS_MAPPING = {
    "fake": "fake",
    "real": "real",
    "0": "fake",
    "1": "real"
}


# ==========================================
# 2. FIND SOURCE FOLDERS
# ==========================================

def find_folder(parent, names):

    if not parent.exists():
        raise FileNotFoundError(
            f"Folder not found: {parent}"
        )

    lookup = {
        p.name.lower(): p
        for p in parent.iterdir()
        if p.is_dir()
    }

    for name in names:

        if name.lower() in lookup:
            return lookup[name.lower()]

    raise FileNotFoundError(
        f"Could not find {names} inside {parent}. "
        f"Available: {list(lookup.keys())}"
    )


# ==========================================
# 3. COLLECT SOURCE IMAGES
# ==========================================

def collect_images(split_name):

    split_folder = find_folder(
        SOURCE_DIR,
        [split_name, "val" if split_name == "validation" else split_name]
    )

    records = []

    for class_folder in split_folder.iterdir():

        if not class_folder.is_dir():
            continue

        class_name = CLASS_MAPPING.get(
            class_folder.name.lower()
        )

        if class_name is None:
            continue

        for image_path in class_folder.rglob("*"):

            if (
                image_path.is_file()
                and image_path.suffix.lower() in SUPPORTED_FORMATS
            ):

                records.append({
                    "path": image_path,
                    "label": class_name
                })

    if not records:
        raise RuntimeError(
            f"No images found in {split_folder}"
        )

    return records


# ==========================================
# 4. IMAGE PREPROCESSING
# ==========================================

def preprocess_image(image_path, output_path):

    try:

        with Image.open(image_path) as image:

            # Correct image orientation
            image = ImageOps.exif_transpose(image)

            # Convert to RGB
            image = image.convert("RGB")

            # Resize image
            image = image.resize(
                IMAGE_SIZE,
                Image.Resampling.BILINEAR
            )

            # Save processed image
            output_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            image.save(
                output_path,
                format="JPEG",
                quality=95,
                optimize=True
            )

        return True

    except Exception as error:

        print(
            f"\nSkipped: {image_path}\nReason: {error}"
        )

        return False


# ==========================================
# 5. PROCESS EACH DATASET SPLIT
# ==========================================

def process_split(split_name):

    print(
        f"\n========== PROCESSING {split_name.upper()} =========="
    )

    records = collect_images(split_name)

    successful = Counter()
    failed = Counter()

    for record in tqdm(records):

        image_path = record["path"]

        label = record["label"]

        # Preserve relative video folder structure
        relative_path = image_path.relative_to(
            SOURCE_DIR
        )

        output_path = (
            OUTPUT_DIR / relative_path
        ).with_suffix(".jpg")

        if preprocess_image(image_path, output_path):

            successful[label] += 1

        else:

            failed[label] += 1

    print("\nProcessing summary:")

    for label in ["fake", "real"]:

        print(
            f"{label.upper()}: "
            f"Successful={successful[label]}, "
            f"Failed={failed[label]}"
        )

    return successful, failed


# ==========================================
# 6. RUN COMPLETE PIPELINE
# ==========================================

def main():

    print("\n================================")
    print("DEEPFAKE VIDEO PREPROCESSING")
    print("================================")

    print("Source:", SOURCE_DIR)
    print("Output:", OUTPUT_DIR)
    print("Image size:", IMAGE_SIZE)

    if not SOURCE_DIR.exists():

        raise FileNotFoundError(
            f"Source dataset not found: {SOURCE_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    total_success = Counter()
    total_failed = Counter()

    for split in SPLITS:

        success, failed = process_split(split)

        total_success.update(success)
        total_failed.update(failed)

    print("\n================================")
    print("FINAL PREPROCESSING REPORT")
    print("================================")

    for label in ["fake", "real"]:

        print(
            f"{label.upper()}: "
            f"Processed={total_success[label]}, "
            f"Failed={total_failed[label]}"
        )

    print(
        "\nTotal processed:",
        sum(total_success.values())
    )

    print(
        "Total failed:",
        sum(total_failed.values())
    )

    print("\nPreprocessing completed!")


# ==========================================
# 7. EXECUTION
# ==========================================

if __name__ == "__main__":

    main()


