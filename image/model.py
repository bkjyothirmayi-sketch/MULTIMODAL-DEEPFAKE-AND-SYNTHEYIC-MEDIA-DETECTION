
import os
import time
import copy
import json
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
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

# ==========================================================
# 1. CONFIGURATION
# ==========================================================

DATASET_DIR = r"D:\AI_Projects\archive\ddata"

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")

OUTPUT_DIR = r"D:\AI_Projects\mobilenet_results"

EPOCHS = 30
BATCH_SIZE = 8
LEARNING_RATE = 0.0001
IMAGE_SIZE = 224
NUM_WORKERS = 0

RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# 2. REPRODUCIBILITY
# ==========================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

torch.set_num_threads(4)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\n========== DEVICE INFORMATION ==========")
print("Device:", DEVICE)

if DEVICE.type == "cpu":
    print("Training will run on CPU.")
else:
    print("GPU:", torch.cuda.get_device_name(0))

# ==========================================================
# 3. CHECK DATASET
# ==========================================================

def check_dataset():

    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_DIR}"
        )

    if not os.path.exists(TRAIN_DIR):
        raise FileNotFoundError(
            f"Training directory not found: {TRAIN_DIR}"
        )

    if not os.path.exists(TEST_DIR):
        raise FileNotFoundError(
            f"Testing directory not found: {TEST_DIR}"
        )

    print("\nDataset directories verified.")

# ==========================================================
# 4. LOAD DATASET
# ==========================================================

def load_data():

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
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

    train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=train_transform
    )

    test_dataset = datasets.ImageFolder(
        root=TEST_DIR,
        transform=test_transform
    )

    print("\nTraining class mapping:",
          train_dataset.class_to_idx)

    print("Testing class mapping:",
          test_dataset.class_to_idx)

    # Case-insensitive class validation
    train_classes = {
        name.lower(): index
        for name, index in train_dataset.class_to_idx.items()
    }

    test_classes = {
        name.lower(): index
        for name, index in test_dataset.class_to_idx.items()
    }

    expected_classes = {"fake", "real"}

    if set(train_classes.keys()) != expected_classes:
        raise ValueError(
            f"Expected fake and real training folders. "
            f"Found: {train_dataset.class_to_idx}"
        )

    if set(test_classes.keys()) != expected_classes:
        raise ValueError(
            f"Expected fake and real testing folders. "
            f"Found: {test_dataset.class_to_idx}"
        )

    if train_classes != test_classes:
        raise ValueError(
            "Training and testing class mappings do not match."
        )

    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty.")

    if len(test_dataset) == 0:
        raise ValueError("Testing dataset is empty.")

    # Enforce consistent semantic labels:
    # fake = 0, real = 1
    if train_classes["fake"] != 0 or train_classes["real"] != 1:
        raise ValueError(
            "Unexpected class order. Please check folder names."
        )

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

    print("\n========== DATASET INFORMATION ==========")
    print("Training images:", len(train_dataset))
    print("Testing images:", len(test_dataset))
    print("Classes:", train_dataset.classes)
    print("Mapping:", train_dataset.class_to_idx)

    return train_loader, test_loader

# ==========================================================
# 5. BUILD MOBILENETV2 MODEL
# ==========================================================

def build_model():

    print("\nLoading MobileNetV2...")

    try:
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)

    except Exception as error:
        raise RuntimeError(
            "Could not load pretrained MobileNetV2 weights. "
            "Check your internet connection or PyTorch cache. "
            "Original error: " + str(error)
        )

    # Freeze pretrained feature extractor
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    # Replace classifier for binary classification
    input_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(input_features, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 2)
    )

    model = model.to(DEVICE)

    print("Model: MobileNetV2")
    print("Input size:", IMAGE_SIZE)
    print("Output classes: 2")
    print("Trainable parameters: classifier only")

    return model

# ==========================================================
# 6. TRAINING FUNCTION
# ==========================================================

def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer
):

    model.train()

    # Keep frozen feature extractor and its BatchNorm layers
    # in evaluation mode.
    model.features.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy

# ==========================================================
# 7. EVALUATION FUNCTION
# ==========================================================

def evaluate_model(model, test_loader):

    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            probabilities = torch.softmax(outputs, dim=1)

            predictions = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.numpy())
            all_predictions.extend(predictions.cpu().numpy())

            # Probability of REAL class (index 1)
            all_probabilities.extend(
                probabilities[:, 1].cpu().numpy()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
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

    try:
        auc = roc_auc_score(
            all_labels,
            all_probabilities
        )
    except ValueError:
        auc = float("nan")

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
            labels=[0, 1],
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

    # Save confusion matrix
    plt.figure(figsize=(6, 5))

    plt.imshow(cm)

    plt.title("MobileNetV2 Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    plt.xticks([0, 1], ["Fake", "Real"])
    plt.yticks([0, 1], ["Fake", "Real"])

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "confusion_matrix.png"
        )
    )

    plt.close()

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(auc)
    }

# ==========================================================
# 8. MAIN TRAINING
# ==========================================================

def main():

    print("\n======================================")
    print(" MOBILENETV2 DEEPFAKE IMAGE DETECTION")
    print("======================================")

    print("Epochs:", EPOCHS)
    print("Batch size:", BATCH_SIZE)
    print("Learning rate:", LEARNING_RATE)

    check_dataset()

    train_loader, test_loader = load_data()

    model = build_model()

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.classifier.parameters(),
        lr=LEARNING_RATE
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.5
    )

    history = {
        "train_loss": [],
        "train_accuracy": []
    }

    best_train_loss = float("inf")
    best_model_state = None

    start_time = time.time()

    # ======================================================
    # TRAINING LOOP
    # ======================================================

    for epoch in range(EPOCHS):

        epoch_start = time.time()

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)

        # Keep the lowest training-loss checkpoint.
        if train_loss < best_train_loss:

            best_train_loss = train_loss

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": best_model_state,
                    "class_to_idx": train_loader.dataset.class_to_idx,
                    "model_name": "MobileNetV2"
                },
                os.path.join(
                    OUTPUT_DIR,
                    "best_mobilenet.pth"
                )
            )

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Loss: {train_loss:.4f} "
            f"Accuracy: {train_accuracy:.4f} "
            f"Time: {epoch_time:.2f}s"
        )

    total_time = time.time() - start_time

    print("\nTraining completed.")

    print(
        f"Total training time: "
        f"{total_time / 60:.2f} minutes"
    )

    # ======================================================
    # SAVE BEST MODEL
    # ======================================================

    if best_model_state is not None:

        model.load_state_dict(best_model_state)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_to_idx": train_loader.dataset.class_to_idx,
            "model_name": "MobileNetV2",
            "image_size": IMAGE_SIZE
        },
        os.path.join(
            OUTPUT_DIR,
            "final_mobilenet.pth"
        )
    )

    # ======================================================
    # SAVE TRAINING GRAPH
    # ======================================================

    plt.figure(figsize=(10, 5))

    plt.plot(
        history["train_loss"],
        label="Training Loss"
    )

    plt.plot(
        history["train_accuracy"],
        label="Training Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("MobileNetV2 Training History")

    plt.legend()
    plt.grid(True)

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "training_history.png"
        )
    )

    plt.close()

    # Save history
    with open(
        os.path.join(OUTPUT_DIR, "training_history.json"),
        "w"
    ) as file:

        json.dump(history, file, indent=4)

    # ======================================================
    # FINAL TEST EVALUATION
    # ======================================================

    results = evaluate_model(
        model,
        test_loader
    )

    with open(
        os.path.join(OUTPUT_DIR, "test_results.json"),
        "w"
    ) as file:

        json.dump(results, file, indent=4)

    print("\n========== COMPLETED ==========")

    print("Saved files in:", OUTPUT_DIR)

    print("1. best_mobilenet.pth")
    print("2. final_mobilenet.pth")
    print("3. training_history.png")
    print("4. training_history.json")
    print("5. confusion_matrix.png")
    print("6. test_results.json")


# ==========================================================
# 9. RUN
# ==========================================================

if __name__ == "__main__":
    main()
