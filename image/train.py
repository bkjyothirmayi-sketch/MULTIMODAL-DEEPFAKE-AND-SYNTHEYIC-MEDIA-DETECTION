
"""
train_mobilenet.py
MobileNetV2 - Deepfake Image Detection

Classes:
    Fake = 0
    Real = 1

Features:
    - CPU/GPU automatic detection
    - 30 epochs
    - Training and validation
    - Accuracy, Precision, Recall, F1
    - Confusion Matrix
    - ROC-AUC
    - Loss and accuracy graphs
    - Best model checkpoint
"""

import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from mobilenet_model import build_mobilenet_model, get_device


# ==========================================
# 1. CONFIGURATION
# ==========================================

DATASET_DIR ="D:\AI_Projects\archive\ddata"

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

MODEL_SAVE_PATH = "D:\AI_Projects\best _model path"

EPOCHS = 30
BATCH_SIZE = 8
LEARNING_RATE = 0.0001

IMAGE_SIZE = 224
NUM_CLASSES = 2

NUM_WORKERS = 0  # Recommended for Windows

DEVICE = get_device()

torch.set_num_threads(4)


# ==========================================
# 2. IMAGE TRANSFORMATIONS
# ==========================================

train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# 3. LOAD DATASET
# ==========================================

def load_datasets():

    train_dataset = datasets.ImageFolder(
        TRAIN_DIR,
        transform=train_transforms
    )

    val_dataset = datasets.ImageFolder(
        VAL_DIR,
        transform=val_test_transforms
    )

    test_dataset = datasets.ImageFolder(
        TEST_DIR,
        transform=val_test_transforms
    )

    expected_classes = ["Fake", "Real"]

    for dataset in [train_dataset, val_dataset, test_dataset]:

        if dataset.classes != expected_classes:
            raise ValueError(
                f"Incorrect class mapping: {dataset.classes}. "
                f"Expected: {expected_classes}"
            )

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
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print("\n========== DATASET INFORMATION ==========")

    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Testing images:", len(test_dataset))

    print("Classes:", train_dataset.classes)

    return train_loader, val_loader, test_loader


# ==========================================
# 4. TRAINING FUNCTION
# ==========================================

def train_one_epoch(model, loader, criterion, optimizer):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == labels).sum().item()

        total += labels.size(0)

    epoch_loss = total_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ==========================================
# 5. VALIDATION FUNCTION
# ==========================================

def validate_model(model, loader, criterion):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()

            total += labels.size(0)

    val_loss = total_loss / total
    val_accuracy = correct / total

    return val_loss, val_accuracy


# ==========================================
# 6. MODEL EVALUATION
# ==========================================

def evaluate_model(model, loader):

    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)

            outputs = model(images)

            probabilities = torch.softmax(outputs, dim=1)

            predictions = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.numpy())

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities[:, 1].cpu().numpy()
            )

    accuracy = accuracy_score(
        all_labels, all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        all_labels,
        all_probabilities
    ) if len(set(all_labels)) == 2 else float("nan")

    print("\n========== TEST RESULTS ==========")

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {auc:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=["Fake", "Real"],
            zero_division=0
        )
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=[0, 1]
    )

    print("Confusion Matrix:")
    print(cm)

    return cm


# ==========================================
# 7. TRAINING GRAPHS
# ==========================================

def plot_training_history(history):

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)

    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")

    plt.title("MobileNetV2 Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)

    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")

    plt.title("MobileNetV2 Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()

    plt.savefig("mobilenet_training_graph.png")

    plt.show()


# ==========================================
# 8. MAIN TRAINING PIPELINE
# ==========================================

def main():

    print("\n====================================")
    print(" MOBILE NET V2 DEEPFAKE DETECTION")
    print("====================================")

    print("Device:", DEVICE)

    # Load dataset

    train_loader, val_loader, test_loader = load_datasets()

    # Initialize model

    model = build_mobilenet_model(
        num_classes=NUM_CLASSES,
        pretrained=True,
        freeze_backbone=True
    )

    model = model.to(DEVICE)

    # Loss function

    criterion = nn.CrossEntropyLoss()

    # Optimizer: Train classifier parameters only

    optimizer = optim.Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=LEARNING_RATE
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    best_val_accuracy = 0.0

    best_weights = copy.deepcopy(model.state_dict())

    start_time = time.time()

    # Training loop

    for epoch in range(EPOCHS):

        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_acc = validate_model(
            model,
            val_loader,
            criterion
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_accuracy:

            best_val_accuracy = val_acc

            best_weights = copy.deepcopy(
                model.state_dict()
            )

            torch.save(
                best_weights,
                MODEL_SAVE_PATH
            )

        epoch_time = time.time() - epoch_start

        print(
            f"\nEpoch [{epoch + 1}/{EPOCHS}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Train Acc: {train_acc:.4f} "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val Acc: {val_acc:.4f} "
            f"| Time: {epoch_time:.2f}s"
        )

    # Restore best model

    model.load_state_dict(best_weights)

    total_time = (time.time() - start_time) / 60

    print("\n========== TRAINING COMPLETED ==========")

    print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")

    print(f"Total Training Time: {total_time:.2f} minutes")

    # Save final checkpoint

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_to_idx": {"Fake": 0, "Real": 1},
            "image_size": IMAGE_SIZE,
            "model_name": "MobileNetV2",
            "best_val_accuracy": best_val_accuracy
        },
        "mobilenet_final_checkpoint.pth"
    )

    # Test evaluation

    evaluate_model(model, test_loader)

    # Plot training graphs

    plot_training_history(history)


if __name__ == "__main__":

    main()








