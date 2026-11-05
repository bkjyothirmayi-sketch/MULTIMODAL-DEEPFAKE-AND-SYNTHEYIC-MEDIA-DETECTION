# ============================================================
# VIDEO DEEPFAKE DETECTION - SHUFFLENETV2 TRAINING
# Compatible with the supplied model.py and dataset.py
# Training + validation only. No test-set evaluation.
# ============================================================

import os
import sys
import copy
import csv
import time
import tempfile
from pathlib import Path

import torch
import torch.nn as nn


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(r"D:\AI_Projects\archive (2)\deepfake_database").resolve().parent

# Allow imports from the current video module folder
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

MODEL_DIR = BASE_DIR / "saved_models"
RESULTS_DIR = BASE_DIR / "training_results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Use a new filename to avoid conflicts with previous files
MODEL_PATH = MODEL_DIR / "shufflenet_video_best_v3.pth"

HISTORY_PATH = RESULTS_DIR / "training_history.csv"


# ============================================================
# 2. IMPORT YOUR MODEL AND DATASET
# ============================================================

from model import (
    build_model,
    get_loss_function,
    get_optimizer,
    get_scheduler,
    get_device,
    IMAGE_SIZE
)

from dataset import get_dataloaders


# ============================================================
# 3. TRAINING CONFIGURATION
# ============================================================

EPOCHS = 20

LEARNING_RATE = 0.0001

WEIGHT_DECAY = 0.0001

EARLY_STOPPING_PATIENCE = 5

CLASS_MAPPING = {
    "fake": 0,
    "real": 1
}

# CPU configuration
if not torch.cuda.is_available():
    torch.set_num_threads(min(8, os.cpu_count() or 1))

DEVICE = get_device()

print("\n========== TRAINING CONFIGURATION ==========")
print("Architecture: ShuffleNetV2")
print("Image size:", IMAGE_SIZE)
print("Epochs:", EPOCHS)
print("Learning rate:", LEARNING_RATE)
print("Weight decay:", WEIGHT_DECAY)
print("Device:", DEVICE)
print("Model save path:", MODEL_PATH)


# ============================================================
# 4. SAVE CHECKPOINT SAFELY
# ============================================================

def save_checkpoint(checkpoint, save_path):

    save_path = Path(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = None

    try:
        # Create a temporary file in the same directory
        fd, temp_name = tempfile.mkstemp(
            prefix="shufflenet_temp_",
            suffix=".pth",
            dir=str(save_path.parent)
        )

        os.close(fd)

        temp_path = Path(temp_name)

        # Save using an already-open file handle
        with open(temp_path, "wb") as file:
            torch.save(checkpoint, file)
            file.flush()
            os.fsync(file.fileno())

        # Replace destination only after successful saving
        os.replace(str(temp_path), str(save_path))

        print("\nCheckpoint saved successfully!")
        print("Location:", save_path)
        print(
            "File size:",
            round(save_path.stat().st_size / (1024 * 1024), 2),
            "MB"
        )

    except Exception as error:

        print("\nCHECKPOINT SAVE ERROR:")
        print(repr(error))

        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass

        raise


# ============================================================
# 5. TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0

    correct_predictions = 0

    total_samples = 0

    start_time = time.time()

    for batch_index, (images, labels) in enumerate(train_loader):

        images = images.to(device)

        labels = labels.to(device).long()

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward propagation
        outputs = model(images)

        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Calculate batch accuracy
        predictions = torch.argmax(outputs, dim=1)

        correct_predictions += (
            predictions == labels
        ).sum().item()

        batch_size = labels.size(0)

        total_samples += batch_size

        running_loss += loss.item() * batch_size

        # Display progress
        if (batch_index + 1) % 50 == 0:

            print(
                f"Batch [{batch_index + 1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    epoch_loss = running_loss / max(total_samples, 1)

    epoch_accuracy = (
        correct_predictions / max(total_samples, 1)
    ) * 100

    epoch_time = time.time() - start_time

    return epoch_loss, epoch_accuracy, epoch_time


# ============================================================
# 6. VALIDATION PASS
# Used for best-checkpoint selection and learning-rate schedule.
# Does not use the test dataset.
# ============================================================

def validate_one_epoch(
    model,
    val_loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    correct_predictions = 0

    total_samples = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)

            labels = labels.to(device).long()

            outputs = model(images)

            loss = criterion(outputs, labels)

            predictions = torch.argmax(outputs, dim=1)

            batch_size = labels.size(0)

            running_loss += loss.item() * batch_size

            correct_predictions += (
                predictions == labels
            ).sum().item()

            total_samples += batch_size

    val_loss = running_loss / max(total_samples, 1)

    val_accuracy = (
        correct_predictions / max(total_samples, 1)
    ) * 100

    return val_loss, val_accuracy


# ============================================================
# 7. SAVE TRAINING HISTORY
# ============================================================

def save_training_history(history):

    if not history:
        return

    with open(
        HISTORY_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
                "learning_rate",
                "epoch_time"
            ]
        )

        writer.writeheader()

        writer.writerows(history)

    print("\nTraining history saved:", HISTORY_PATH)


# ============================================================
# 8. MAIN TRAINING FUNCTION
# ============================================================

def main():

    print("\n========== LOADING DATASETS ==========")

    # The dataset function returns train, validation and test
    # loaders. The test loader is deliberately not used here.
    train_loader, val_loader, _ = get_dataloaders()

    print("Training batches:", len(train_loader))

    print("Validation batches:", len(val_loader))

    if len(train_loader) == 0:
        raise RuntimeError("Training dataset is empty.")

    if len(val_loader) == 0:
        raise RuntimeError("Validation dataset is empty.")

    print("\n========== BUILDING SHUFFLENETV2 ==========")

    model = build_model(pretrained=True)

    model = model.to(DEVICE)

    criterion = get_loss_function()

    optimizer = get_optimizer(
        model,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = get_scheduler(optimizer)

    print("\nModel initialized successfully!")

    # Best validation loss
    best_val_loss = float("inf")

    best_epoch = 0

    patience_counter = 0

    best_model_weights = copy.deepcopy(
        model.state_dict()
    )

    history = []

    training_start_time = time.time()

    # ========================================================
    # 9. EPOCH TRAINING LOOP
    # ========================================================

    for epoch in range(EPOCHS):

        print("\n")
        print("=" * 60)

        print(
            f"EPOCH {epoch + 1}/{EPOCHS}"
        )

        print("=" * 60)

        current_lr = optimizer.param_groups[0]["lr"]

        print("Learning rate:", current_lr)

        # ------------------------------
        # TRAINING
        # ------------------------------

        train_loss, train_accuracy, epoch_time = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            DEVICE
        )

        # ------------------------------
        # VALIDATION
        # ------------------------------

        val_loss, val_accuracy = validate_one_epoch(
            model,
            val_loader,
            criterion,
            DEVICE
        )

        # ------------------------------
        # DISPLAY RESULTS
        # ------------------------------

        print("\n========== EPOCH RESULTS ==========")

        print(
            f"Training Loss: {train_loss:.4f}"
        )

        print(
            f"Training Accuracy: {train_accuracy:.2f}%"
        )

        print(
            f"Validation Loss: {val_loss:.4f}"
        )

        print(
            f"Validation Accuracy: {val_accuracy:.2f}%"
        )

        print(
            f"Epoch Time: {epoch_time:.2f} seconds"
        )

        # ------------------------------
        # LEARNING RATE SCHEDULER
        # ------------------------------

        scheduler.step(val_loss)

        # ------------------------------
        # SAVE TRAINING HISTORY
        # ------------------------------

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "train_accuracy": round(train_accuracy, 4),
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_accuracy, 4),
            "learning_rate": current_lr,
            "epoch_time": round(epoch_time, 2)
        })

        save_training_history(history)

        # ------------------------------
        # BEST MODEL CHECKPOINT
        # ------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_epoch = epoch + 1

            patience_counter = 0

            best_model_weights = copy.deepcopy(
                model.state_dict()
            )

            checkpoint = {
                "epoch": epoch + 1,

                "model_state_dict": model.state_dict(),

                "optimizer_state_dict": optimizer.state_dict(),

                "val_loss": val_loss,

                "val_accuracy": val_accuracy,

                "class_mapping": CLASS_MAPPING,

                "image_size": IMAGE_SIZE,

                "model_name": "ShuffleNetV2"
            }

            save_checkpoint(
                checkpoint,
                MODEL_PATH
            )

            print("New best model!")

        else:

            patience_counter += 1

            print(
                "\nNo validation-loss improvement."
            )

            print(
                "Early stopping counter:",
                patience_counter,
                "/",
                EARLY_STOPPING_PATIENCE
            )

        # ------------------------------
        # EARLY STOPPING
        # ------------------------------

        if patience_counter >= EARLY_STOPPING_PATIENCE:

            print("\nEarly stopping activated.")

            break

    # ========================================================
    # 10. FINISH TRAINING
    # ========================================================

    total_training_time = (
        time.time() - training_start_time
    )

    print("\n")
    print("=" * 60)

    print("TRAINING COMPLETED")

    print("=" * 60)

    print("Best epoch:", best_epoch)

    print("Best validation loss:", round(best_val_loss, 4))

    print("Best checkpoint:", MODEL_PATH)

    print(
        "Total training time:",
        round(total_training_time / 60, 2),
        "minutes"
    )

    # Restore best weights in memory
    model.load_state_dict(best_model_weights)

    print("\nBest model weights restored in memory.")

    print("\nTraining history:", HISTORY_PATH)

    print("\nShuffleNetV2 training finished successfully!")


# ============================================================
# 11. EXECUTION
# ============================================================

if __name__ == "__main__":

    main()
