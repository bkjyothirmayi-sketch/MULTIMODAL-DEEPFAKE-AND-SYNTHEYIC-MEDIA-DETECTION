
import json
import shutil
import numpy as np
import librosa

from pathlib import Path
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ==========================================================
# 1. CONFIGURATION
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

OUTPUT_PATH = Path(
    r"D:\AI_Projects\intermediate fusion\output_path"
)

IMAGE_SIZE = (224, 224)

AUDIO_SAMPLE_RATE = 16000
AUDIO_DURATION = 3
AUDIO_LENGTH = AUDIO_SAMPLE_RATE * AUDIO_DURATION

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".ogg", ".m4a"
}

LABELS = ["real", "fake"]


# ==========================================================
# 2. CREATE OUTPUT FOLDERS
# ==========================================================

def create_output_folders():

    for modality in ["images", "audio", "video_frames"]:

        for label in LABELS:

            folder = OUTPUT_PATH / modality / label

            folder.mkdir(
                parents=True,
                exist_ok=True
            )

    print("Output folders created successfully.")


# ==========================================================
# 3. GET DATASET FILES
# ==========================================================

def get_files(root, extensions):

    root = Path(root)

    if not root.exists():
        raise FileNotFoundError(
            f"Dataset path does not exist: {root}"
        )

    return [
        path for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in extensions
    ]


# ==========================================================
# 4. DETECT REAL / FAKE LABEL
# ==========================================================

def get_label(file_path):

    file_path = Path(file_path)

    for folder in reversed(file_path.parts[:-1]):

        folder = folder.lower().strip()

        if folder in LABELS:
            return folder

    raise ValueError(
        f"Cannot determine real/fake label: {file_path}"
    )


# ==========================================================
# 5. IMAGE PREPROCESSING
# ==========================================================

def preprocess_images():

    print("\n========== IMAGE PREPROCESSING ==========")

    files = get_files(
        IMAGE_DATASET_PATH,
        IMAGE_EXTENSIONS
    )

    processed = 0
    failed = 0

    for index, file_path in enumerate(files):

        try:

            label = get_label(file_path)

            image = Image.open(file_path).convert("RGB")

            image = image.resize(
                IMAGE_SIZE,
                Image.Resampling.BILINEAR
            )

            output_file = (
                OUTPUT_PATH
                / "images"
                / label
                / f"{index:08d}.jpg"
            )

            image.save(
                output_file,
                format="JPEG",
                quality=95
            )

            processed += 1

        except Exception as error:

            failed += 1

            print(
                f"Image failed: {file_path} | {error}"
            )

        if (index + 1) % 500 == 0:

            print(
                f"Images processed: {index + 1}/{len(files)}"
            )

    print("Successfully processed:", processed)
    print("Failed:", failed)

    return {
        "total": len(files),
        "processed": processed,
        "failed": failed
    }


# ==========================================================
# 6. AUDIO PREPROCESSING
# ==========================================================

def preprocess_audio():

    print("\n========== AUDIO PREPROCESSING ==========")

    files = get_files(
        AUDIO_DATASET_PATH,
        AUDIO_EXTENSIONS
    )

    processed = 0
    failed = 0

    for index, file_path in enumerate(files):

        try:

            label = get_label(file_path)

            audio, sr = librosa.load(
                str(file_path),
                sr=AUDIO_SAMPLE_RATE,
                mono=True
            )

            # Remove invalid values
            audio = np.nan_to_num(audio)

            # Fixed duration
            if len(audio) < AUDIO_LENGTH:

                audio = np.pad(
                    audio,
                    (0, AUDIO_LENGTH - len(audio))
                )

            else:

                audio = audio[:AUDIO_LENGTH]

            # Peak normalization
            peak = np.max(np.abs(audio))

            if peak > 0:

                audio = audio / peak

            output_file = (
                OUTPUT_PATH
                / "audio"
                / label
                / f"{index:08d}.npy"
            )

            np.save(
                output_file,
                audio.astype(np.float32)
            )

            processed += 1

        except Exception as error:

            failed += 1

            print(
                f"Audio failed: {file_path} | {error}"
            )

        if (index + 1) % 100 == 0:

            print(
                f"Audio processed: {index + 1}/{len(files)}"
            )

    print("Successfully processed:", processed)
    print("Failed:", failed)

    return {
        "total": len(files),
        "processed": processed,
        "failed": failed
    }


# ==========================================================
# 7. VIDEO FRAME PREPROCESSING
# Every extracted frame is treated as an image.
# No original video or frame grouping is required.
# ==========================================================

def preprocess_video_frames():

    print("\n========== VIDEO FRAME PREPROCESSING ==========")

    files = get_files(
        VIDEO_FRAME_DATASET_PATH,
        IMAGE_EXTENSIONS
    )

    processed = 0
    failed = 0

    for index, file_path in enumerate(files):

        try:

            label = get_label(file_path)

            frame = Image.open(file_path).convert("RGB")

            frame = frame.resize(
                IMAGE_SIZE,
                Image.Resampling.BILINEAR
            )

            output_file = (
                OUTPUT_PATH
                / "video_frames"
                / label
                / f"{index:08d}.jpg"
            )

            frame.save(
                output_file,
                format="JPEG",
                quality=95
            )

            processed += 1

        except Exception as error:

            failed += 1

            print(
                f"Frame failed: {file_path} | {error}"
            )

        if (index + 1) % 500 == 0:

            print(
                f"Frames processed: {index + 1}/{len(files)}"
            )

    print("Successfully processed:", processed)
    print("Failed:", failed)

    return {
        "total": len(files),
        "processed": processed,
        "failed": failed
    }


# ==========================================================
# 8. SAVE PREPROCESSING REPORT
# ==========================================================

def save_report(report):

    report_path = OUTPUT_PATH / "preprocessing_report.json"

    with open(report_path, "w") as file:

        json.dump(
            report,
            file,
            indent=4
        )

    print("\nReport saved:", report_path)


# ==========================================================
# 9. MAIN EXECUTION
# ==========================================================

if __name__ == "__main__":

    print("MULTIMODAL DEEPFAKE PREPROCESSING STARTED")

    create_output_folders()

    report = {}

    report["images"] = preprocess_images()

    report["audio"] = preprocess_audio()

    report["video_frames"] = preprocess_video_frames()

    save_report(report)

    print("\n========================================")
    print("PREPROCESSING COMPLETED")
    print("========================================")

    print(json.dumps(report, indent=4))

