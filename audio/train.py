import os
import csv
import time
import copy
import torch
import numpy as np

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from model import (
    build_model,
    get_loss_function,
    get_optimizer,
    get_scheduler,
    DEVICE,
    IMAGE_SIZE
)

# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_DIR = r"D:\AI_Projects\audio dataset"

DATASET_PATH = os.path.join(
    BASE_DIR,
   r"D:\AI_Projects\audio dataset\deepfake_audio"
)

TRAIN_PATH = os.path.join(
    DATASET_PATH,
    r"D:\AI_Projects\audio dataset\output_path\train"
)

TEST_PATH = os.path.join(
    DATASET_PATH,
    r"D:\AI_Projects\audio dataset\output_path\test"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    r"D:\AI_Projects\audio dataset\output_path"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(
    OUTPUT_DIR,
    r"D:\AI_Projects\audio dataset\model_save_path"
)


MODEL_SAVE_DIR = r"D:\AI_Projects\audio dataset\model_save_path"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(
    MODEL_SAVE_DIR,
    "best_audio_model.pth"
)

print("BASE DIRECTORY:", BASE_DIR)
print("DATASET PATH:", DATASET_PATH)
print("TRAIN PATH:", TRAIN_PATH)
print("TEST PATH:", TEST_PATH)
print("MODEL SAVE PATH:", MODEL_SAVE_PATH)


# Create folder automatically
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Complete model file path
MODEL_SAVE_PATH = os.path.join(
    MODEL_SAVE_DIR,
    "best_audio_model.pth"
)

print("Model save directory:", MODEL_SAVE_DIR)
print("Model file path:", MODEL_SAVE_PATH)




HISTORY_PATH = os.path.join(
    OUTPUT_DIR,
    r"D:\AI_Projects\audio dataset\history_path\training history_path.csv.xlsx"
)

BATCH_SIZE = 8
EPOCHS = 15
LEARNING_RATE = 0.0001

VALIDATION_SIZE = 0.20
RANDOM_SEED = 42

NUM_WORKERS = 0

CLASS_NAMES = ["fake", "real"]


# ==========================================
# 2. IMAGE TRANSFORMS
# ==========================================

def get_transforms():

    image_transforms = transforms.Compose([

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    ])

    return image_transforms


# ==========================================
# 3. LOAD DATASET
# ==========================================

def load_datasets():

    print("\nLoading audio spectrogram datasets...")

    if not os.path.isdir(TRAIN_PATH):
        raise FileNotFoundError(
            f"Training folder not found: {TRAIN_PATH}"
        )

    if not os.path.isdir(TEST_PATH):
        raise FileNotFoundError(
            f"Testing folder not found: {TEST_PATH}"
        )

    transform = get_transforms()

    train_dataset = datasets.ImageFolder(
        root=TRAIN_PATH,
        transform=transform
    )

    test_dataset = datasets.ImageFolder(
        root=TEST_PATH,
        transform=transform
    )

    print("\n========== DATASET INFORMATION ==========")

    print("Training images:", len(train_dataset))
    print("Testing images:", len(test_dataset))

    print("Class mapping:", train_dataset.class_to_idx)

    expected_mapping = {
        "fake": 0,
        "real": 1
    }

    if train_dataset.class_to_idx != expected_mapping:
        raise ValueError(
            "Incorrect training class mapping. "
            f"Expected {expected_mapping}, "
            f"got {train_dataset.class_to_idx}"
        )

    if test_dataset.class_to_idx != expected_mapping:
        raise ValueError(
            "Training and testing class mappings do not match."
        )

    return train_dataset, test_dataset


# ==========================================
# 4. CREATE DATA LOADERS
# ==========================================

def create_data_loaders():

    train_dataset, test_dataset = load_datasets()

    # Split training data into training and validation
    validation_count = max(
        1,
        int(len(train_dataset) * VALIDATION_SIZE)
    )

    training_count = len(train_dataset) - validation_count

    if training_count < 1:
        raise ValueError(
            "Not enough training images for train/validation split."
        )

    generator = torch.Generator().manual_seed(
        RANDOM_SEED
    )

    training_subset, validation_subset = random_split(
        train_dataset,
        [training_count, validation_count],
        generator=generator
    )

    train_loader = DataLoader(
        training_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda")
    )

    validation_loader = DataLoader(
        validation_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda")
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda")
    )

    print("\n========== DATA LOADER INFORMATION ==========")

    print("Training samples:", len(training_subset))
    print("Validation samples:", len(validation_subset))
    print("Testing samples:", len(test_dataset))

    print("Batch size:", BATCH_SIZE)

    return train_loader, validation_loader, test_loader


# ==========================================
# 5. TRAINING FUNCTION
# ==========================================

def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer
):

    model.train()

    # Keep frozen feature extractor in evaluation mode
    # so its BatchNorm running statistics do not change.
    model.features.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ==========================================
# 6. VALIDATION FUNCTION
# ==========================================

def validate_model(
    model,
    validation_loader,
    criterion
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item() * images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    validation_loss = running_loss / total
    validation_accuracy = correct / total

    return validation_loss, validation_accuracy


# ==========================================
# 7. SAVE TRAINING HISTORY
# ==========================================

def save_training_history(history):

    with open(
        HISTORY_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "epoch",
            "train_loss",
            "train_accuracy",
            "validation_loss",
            "validation_accuracy"
        ])

        for row in history:

            writer.writerow(row)

    print("\nTraining history saved:", HISTORY_PATH)


# ==========================================
# 8. MODEL TRAINING PIPELINE
# ==========================================

def train_model():

    print("\n==========================================")
    print(" AUDIO DEEPFAKE DETECTION - MOBILENETV2")
    print("==========================================")

    print("Device:", DEVICE)
    print("Epochs:", EPOCHS)
    print("Batch size:", BATCH_SIZE)
    print("Learning rate:", LEARNING_RATE)

    # Load data
    train_loader, validation_loader, test_loader = (
        create_data_loaders()
    )

    # Initialize model
    model = build_model(pretrained=True)

    # Loss function
    criterion = get_loss_function()

    # Optimizer
    optimizer = get_optimizer(
        model,
        learning_rate=LEARNING_RATE
    )

    # Scheduler
    scheduler = get_scheduler(optimizer)

    best_validation_loss = float("inf")
    best_model_weights = copy.deepcopy(
        model.state_dict()
    )

    history = []

    # ======================================
    # TRAINING LOOP
    # ======================================

    for epoch in range(EPOCHS):

        start_time = time.time()

        print("\n------------------------------------------")
        print(f"EPOCH {epoch + 1}/{EPOCHS}")
        print("------------------------------------------")

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        validation_loss, validation_accuracy = validate_model(
            model,
            validation_loader,
            criterion
        )

        scheduler.step(validation_loss)

        epoch_time = time.time() - start_time

        history.append([
            epoch + 1,
            train_loss,
            train_accuracy,
            validation_loss,
            validation_accuracy
        ])

        print(f"Training Loss:      {train_loss:.4f}")
        print(f"Training Accuracy:  {train_accuracy * 100:.2f}%")

        print(f"Validation Loss:    {validation_loss:.4f}")
        print(f"Validation Accuracy:{validation_accuracy * 100:.2f}%")

        print(f"Epoch Time:         {epoch_time:.2f} seconds")

        print(
            "Learning Rate:",
            optimizer.param_groups[0]["lr"]
        )

        # Save best model using validation loss
        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss

            best_model_weights = copy.deepcopy(
                model.state_dict()
            )

            torch.save(
                {
                    "model_state_dict": best_model_weights,
                    "class_mapping": {
                        "fake": 0,
                        "real": 1
                    },
                    "image_size": IMAGE_SIZE,
                    "model_name": "MobileNetV2",
                    "best_validation_loss": best_validation_loss,
                    "epoch": epoch + 1
                },
                MODEL_SAVE_PATH
            )

            print("\nBest model saved!")

    # ======================================
    # RESTORE BEST MODEL
    # ======================================

    model.load_state_dict(
        best_model_weights
    )

    save_training_history(history)

    print("\n==========================================")
    print(" TRAINING COMPLETED")
    print("==========================================")

    print("Best validation loss:", best_validation_loss)
    print("Best model:", MODEL_SAVE_PATH)

    # Evaluate test dataset only after training
    evaluate_model(
        model,
        test_loader
    )

    return model


# ==========================================
# 9. MODEL EVALUATION
# ==========================================

def evaluate_model(model, test_loader):

    print("\n==========================================")
    print(" FINAL TEST DATASET EVALUATION")
    print("==========================================")

    model.eval()

    actual_labels = []
    predicted_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            actual_labels.extend(
                labels.numpy().tolist()
            )

            predicted_labels.extend(
                predictions.cpu().numpy().tolist()
            )

    accuracy = accuracy_score(
        actual_labels,
        predicted_labels
    )

    precision = precision_score(
        actual_labels,
        predicted_labels,
        zero_division=0
    )

    recall = recall_score(
        actual_labels,
        predicted_labels,
        zero_division=0
    )

    f1 = f1_score(
        actual_labels,
        predicted_labels,
        zero_division=0
    )

    cm = confusion_matrix(
        actual_labels,
        predicted_labels,
        labels=[0, 1]
    )

    print("\n========== PERFORMANCE METRICS ==========")

    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1 Score:  {f1 * 100:.2f}%")

    print("\n========== CLASSIFICATION REPORT ==========")

    print(
        classification_report(
            actual_labels,
            predicted_labels,
            labels=[0, 1],
            target_names=["Fake", "Real"],
            zero_division=0
        )
    )

    print("\n========== CONFUSION MATRIX ==========")

    print("Rows = Actual | Columns = Predicted")
    print("                 Fake   Real")
    print("Actual Fake:    ", cm[0])
    print("Actual Real:    ", cm[1])

    print("\nEvaluation completed!")


# ==========================================
# 10. RUN
# ==========================================

if __name__ == "__main__":

    trained_model = train_model()

