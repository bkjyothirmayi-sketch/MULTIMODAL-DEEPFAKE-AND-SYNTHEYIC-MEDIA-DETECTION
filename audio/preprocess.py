import os
import csv
import numpy as np
import librosa

from PIL import Image
from sklearn.model_selection import train_test_split


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR, r"D:\AI_Projects\audio dataset\deepfake_audio"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR, r"D:\AI_Projects\audio dataset\output_path"
)

SAMPLE_RATE = 16000
DURATION = 3
N_SAMPLES = SAMPLE_RATE * DURATION

N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512

IMAGE_SIZE = 224

TEST_SIZE = 0.20
RANDOM_SEED = 42

LABELS = {
    "fake": 0,
    "real": 1
}

AUDIO_EXTENSIONS = (
    ".mp3", ".wav", ".flac", ".ogg", ".m4a"
)


# ==========================================
# 2. CREATE OUTPUT DIRECTORIES
# ==========================================

def create_directories():

    for split in ["train", "test"]:

        for class_name in LABELS:

            folder = os.path.join(
                OUTPUT_PATH,
                split,
                class_name
            )

            os.makedirs(
                folder,
                exist_ok=True
            )


# ==========================================
# 3. FIND AUDIO FILES
# ==========================================

def get_audio_files():

    audio_files = []

    for class_name, label in LABELS.items():

        class_path = os.path.join(
            DATASET_PATH,
            class_name
        )

        if not os.path.isdir(class_path):

            print("Missing folder:", class_path)
            continue

        for root, _, files in os.walk(class_path):

            for filename in files:

                if filename.lower().endswith(
                    AUDIO_EXTENSIONS
                ):

                    file_path = os.path.join(
                        root,
                        filename
                    )

                    audio_files.append(
                        (file_path, label, class_name)
                    )

    return audio_files


# ==========================================
# 4. AUDIO TO MEL SPECTROGRAM
# ==========================================

def audio_to_spectrogram(file_path):

    try:

        # Load audio
        audio, sr = librosa.load(
            file_path,
            sr=SAMPLE_RATE,
            mono=True
        )

        if len(audio) == 0:
            raise ValueError("Empty audio file")

        # Crop audio to 3 seconds
        if len(audio) > N_SAMPLES:

            start = (len(audio) - N_SAMPLES) // 2

            audio = audio[
                start:start + N_SAMPLES
            ]

        # Pad short audio
        else:

            audio = np.pad(
                audio,
                (0, N_SAMPLES - len(audio))
            )

        # Generate Mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )

        # Convert to decibels
        mel_db = librosa.power_to_db(
            mel,
            ref=np.max
        )

        # Normalize to 0-255
        mel_min = mel_db.min()
        mel_max = mel_db.max()

        mel_image = (
            (mel_db - mel_min) /
            (mel_max - mel_min + 1e-8)
        ) * 255

        mel_image = mel_image.astype(
            np.uint8
        )

        # Create RGB image
        image = Image.fromarray(
            mel_image
        ).convert("RGB")

        # Resize image
        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR
        )

        return image

    except Exception as error:

        raise RuntimeError(str(error))


# ==========================================
# 5. SAVE PREPROCESSED IMAGE
# ==========================================

def save_spectrogram(
    file_path,
    label,
    class_name,
    split,
    index
):

    image = audio_to_spectrogram(file_path)

    output_folder = os.path.join(
        OUTPUT_PATH,
        split,
        class_name
    )

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    # Unique output filename
    filename = f"{index:06d}.png"

    output_file = os.path.join(
        output_folder,
        filename
    )

    image.save(output_file)

    return output_file


# ==========================================
# 6. SAVE CSV REPORTS
# ==========================================

def save_csv(file_path, headers, rows):

    with open(
        file_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(headers)

        writer.writerows(rows)


# ==========================================
# 7. MAIN PREPROCESSING PIPELINE
# ==========================================

def preprocess_dataset():

    print("\n================================")
    print(" AUDIO DEEPFAKE PREPROCESSING")
    print("================================")

    create_directories()

    all_data = get_audio_files()

    if len(all_data) == 0:

        raise ValueError(
            "No audio files found. Check DATASET_PATH."
        )

    print("\nTotal audio files:", len(all_data))

    file_paths = [
        item[0] for item in all_data
    ]

    labels = [
        item[1] for item in all_data
    ]

    # Stratified train-test split
    train_data, test_data = train_test_split(
        all_data,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels
    )

    print("\nTraining files:", len(train_data))
    print("Testing files:", len(test_data))

    manifest_rows = []
    error_rows = []

    successful = 0
    failed = 0

    # Process training and testing files
    for split, split_data in [
        ("train", train_data),
        ("test", test_data)
    ]:

        print(f"\nProcessing {split} data...")

        for index, item in enumerate(split_data):

            file_path, label, class_name = item

            try:

                output_file = save_spectrogram(
                    file_path,
                    label,
                    class_name,
                    split,
                    index
                )

                manifest_rows.append([
                    split,
                    class_name,
                    label,
                    file_path,
                    output_file
                ])

                successful += 1

            except Exception as error:

                failed += 1

                error_rows.append([
                    file_path,
                    str(error)
                ])

                print("\nFailed:", file_path)
                print("Reason:", error)

            if (index + 1) % 100 == 0:

                print(
                    f"{split}: {index + 1}/{len(split_data)} "
                    f"| Successful: {successful} "
                    f"| Failed: {failed}"
                )

    # Save manifest
    save_csv(
        os.path.join(
            OUTPUT_PATH,
            "dataset_manifest.csv"
        ),
        [
            "split",
            "class_name",
            "label",
            "original_audio",
            "processed_image"
        ],
        manifest_rows
    )

    # Save errors
    save_csv(
        os.path.join(
            OUTPUT_PATH,
            "preprocessing_errors.csv"
        ),
        [
            "audio_file",
            "error"
        ],
        error_rows
    )

    # Final report
    print("\n================================")
    print(" PREPROCESSING COMPLETED")
    print("================================")

    print("Total files:", len(all_data))
    print("Successful:", successful)
    print("Failed:", failed)

    print("\nTraining distribution:")

    for class_name in LABELS:

        count = sum(
            1 for row in manifest_rows
            if row[0] == "train"
            and row[1] == class_name
        )

        print(class_name, ":", count)

    print("\nTesting distribution:")

    for class_name in LABELS:

        count = sum(
            1 for row in manifest_rows
            if row[0] == "test"
            and row[1] == class_name
        )

        print(class_name, ":", count)

    print("\nOutput folder:", OUTPUT_PATH)

    print("\nReady for MobileNetV2 training!")


# ==========================================
# 8. RUN
# ==========================================

if __name__ == "__main__":

    preprocess_dataset()

