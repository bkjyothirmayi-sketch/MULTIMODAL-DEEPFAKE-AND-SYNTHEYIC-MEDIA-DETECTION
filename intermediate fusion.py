
import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

FEATURE_DIR = r"D:\AI_Projects\extracted_features"

OUTPUT_DIR = (
    r"D:\AI_Projects\fusion train _output"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_FILE = os.path.join(
    r"D:\AI_Projects\extracted_features\image_features.npz"
)

AUDIO_FILE = os.path.join(
    r"D:\AI_Projects\extracted_features\audio_features.npz"
)

VIDEO_FILES = [
    os.path.join(r"D:\AI_Projects\extracted_features\video_train_features.npz"),
    os.path.join(r"D:\AI_Projects\extracted_features\video_test_features.npz"),
    os.path.join(r"D:\AI_Projects\extracted_features\video_validation_features.npz")
]

RANDOM_STATE = 42

# Canonical mapping:
# Real = 0
# Fake = 1

# ============================================================
# 2. LOAD FEATURE FILE
# ============================================================

def load_features(filename):

    if not os.path.exists(filename):
        raise FileNotFoundError(filename)

    data = np.load(filename, allow_pickle=True)

    features = data["features"].astype(np.float32)
    saved_labels = data["labels"].astype(np.int64)
    paths = data["paths"].astype(str)

    data.close()

    if not (
        len(features) == len(saved_labels) == len(paths)
    ):
        raise ValueError(
            "Feature, label and path counts do not match: "
            + filename
        )

    return features, saved_labels, paths


# ============================================================
# 3. GET CORRECT LABELS FROM FOLDER NAMES
# ============================================================

def get_labels_from_paths(paths):

    labels = []

    for path in paths:

        parts = os.path.normpath(path).lower().split(os.sep)

        if "fake" in parts:
            labels.append(1)

        elif "real" in parts:
            labels.append(0)

        else:
            raise ValueError(
                "Cannot identify fake/real folder in path:\n"
                + path
            )

    return np.array(labels, dtype=np.int64)


# ============================================================
# 4. LOAD AND VERIFY MODALITY DATA
# ============================================================

def prepare_data(filename, modality):

    X, saved_y, paths = load_features(filename)

    y = get_labels_from_paths(paths)

    mismatches = np.sum(saved_y != y)

    print("\n", "=" * 55)
    print(modality, "DATASET")
    print("=" * 55)

    print("Feature shape:", X.shape)
    print("Total samples:", len(y))
    print("Real samples:", np.sum(y == 0))
    print("Fake samples:", np.sum(y == 1))
    print("Stored-label mismatches:", mismatches)

    if not np.isfinite(X).all():
        raise ValueError(
            modality + " features contain NaN or infinity."
        )

    return X, y, paths


# ============================================================
# 5. BUILD CLASSIFIER
# ============================================================

def build_classifier():

    return Pipeline([
        ("scaler", StandardScaler()),

        ("classifier", SGDClassifier(
            loss="log_loss",
            alpha=0.0001,
            max_iter=1000,
            tol=0.001,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        ))
    ])


# ============================================================
# 6. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    X,
    y,
    paths,
    modality,
    split_name
):

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, predictions)

    precision = precision_score(
        y, predictions, zero_division=0
    )

    recall = recall_score(
        y, predictions, zero_division=0
    )

    f1 = f1_score(
        y, predictions, zero_division=0
    )

    auc = roc_auc_score(y, probabilities)

    cm = confusion_matrix(
        y, predictions, labels=[0, 1]
    )

    print("\n", "-" * 55)
    print(modality, "-", split_name)
    print("-" * 55)

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC AUC  :", round(auc, 4))

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(
        y,
        predictions,
        labels=[0, 1],
        target_names=["Real", "Fake"],
        zero_division=0
    ))

    # Save sample-level predictions
    prediction_df = pd.DataFrame({
        "path": paths,
        "actual_label": y,
        "predicted_label": predictions,
        "fake_probability": probabilities
    })

    prediction_file = os.path.join(
        OUTPUT_DIR,
        modality.lower() + "_" + split_name.lower()
        + "_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_file, index=False
    )

    return {
        "modality": modality,
        "split": split_name,
        "samples": int(len(y)),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": cm.tolist()
    }


# ============================================================
# 7. TRAIN AND EVALUATE ONE MODALITY
# ============================================================

def train_modality(
    modality,
    X,
    y,
    paths,
    train_mask,
    test_mask,
    validation_mask=None
):

    print("\n\n", "=" * 65)
    print("TRAINING:", modality)
    print("=" * 65)

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]
    paths_test = paths[test_mask]

    print("Training samples:", len(y_train))
    print("Test samples:", len(y_test))

    if len(np.unique(y_train)) != 2:
        raise ValueError(
            modality + " training set must contain both classes."
        )

    model = build_classifier()

    print("\nTraining classifier...")

    model.fit(X_train, y_train)

    print("Training completed.")

    # Save trained feature classifier
    model_path = os.path.join(
        OUTPUT_DIR,
        modality.lower() + "_feature_classifier.joblib"
    )

    joblib.dump(model, model_path)

    results = []

    # Validation evaluation
    if validation_mask is not None:

        X_val = X[validation_mask]
        y_val = y[validation_mask]
        paths_val = paths[validation_mask]

        results.append(
            evaluate_model(
                model,
                X_val,
                y_val,
                paths_val,
                modality,
                "Validation"
            )
        )

    # Test evaluation
    results.append(
        evaluate_model(
            model,
            X_test,
            y_test,
            paths_test,
            modality,
            "Test"
        )
    )

    return results


# ============================================================
# 8. IMAGE MODEL
# ============================================================

image_X, image_y, image_paths = prepare_data(
    IMAGE_FILE, "IMAGE"
)

image_parts = [
    os.path.normpath(p).lower().split(os.sep)
    for p in image_paths
]

image_train = np.array(
    ["train" in parts for parts in image_parts]
)

image_test = np.array(
    ["test" in parts for parts in image_parts]
)

if not np.all(image_train | image_test):
    raise ValueError("Unexpected image split folder.")

if np.any(image_train & image_test):
    raise ValueError("Image split overlap detected.")

# ============================================================
# 9. AUDIO MODEL
# ============================================================

audio_X, audio_y, audio_paths = prepare_data(
    AUDIO_FILE, "AUDIO"
)

# Audio dataset has no separate train/test folders.
# Use a stratified random 70/15/15 split.

audio_indices = np.arange(len(audio_y))

audio_train_idx, audio_temp_idx = train_test_split(
    audio_indices,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=audio_y
)

audio_val_idx, audio_test_idx = train_test_split(
    audio_temp_idx,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=audio_y[audio_temp_idx]
)

audio_train = np.zeros(len(audio_y), dtype=bool)
audio_val = np.zeros(len(audio_y), dtype=bool)
audio_test = np.zeros(len(audio_y), dtype=bool)

audio_train[audio_train_idx] = True
audio_val[audio_val_idx] = True
audio_test[audio_test_idx] = True

# ============================================================
# 10. VIDEO MODEL
# ============================================================

video_arrays = [
    load_features(p) for p in VIDEO_FILES
]

video_X = np.concatenate([v[0] for v in video_arrays])
video_saved_y = np.concatenate([v[1] for v in video_arrays])
video_paths = np.concatenate([v[2] for v in video_arrays])

video_y = get_labels_from_paths(video_paths)

print("\n", "=" * 55)
print("VIDEO DATASET")
print("=" * 55)

print("Feature shape:", video_X.shape)
print("Total samples:", len(video_y))
print("Real:", np.sum(video_y == 0))
print("Fake:", np.sum(video_y == 1))
print(
    "Stored-label mismatches:",
    np.sum(video_saved_y != video_y)
)

if not np.isfinite(video_X).all():
    raise ValueError("Video features contain NaN or infinity.")

video_parts = [
    os.path.normpath(p).lower().split(os.sep)
    for p in video_paths
]

video_train = np.array(
    ["train" in parts for parts in video_parts]
)

video_test = np.array(
    ["test" in parts for parts in video_parts]
)

video_val = np.array(
    ["validation" in parts for parts in video_parts]
)

if not np.all(video_train | video_test | video_val):
    raise ValueError("Unexpected video split folder.")

# ============================================================
# 11. RUN ALL THREE INDEPENDENT CLASSIFIERS
# ============================================================

all_results = []

# Image
all_results.extend(
    train_modality(
        "IMAGE",
        image_X,
        image_y,
        image_paths,
        image_train,
        image_test
    )
)

# Audio
all_results.extend(
    train_modality(
        "AUDIO",
        audio_X,
        audio_y,
        audio_paths,
        audio_train,
        audio_test,
        audio_val
    )
)

# Video
all_results.extend(
    train_modality(
        "VIDEO",
        video_X,
        video_y,
        video_paths,
        video_train,
        video_test,
        video_val
    )
)

# ============================================================
# 12. SAVE FINAL PERFORMANCE COMPARISON
# ============================================================

results_df = pd.DataFrame(all_results)

csv_path = os.path.join(
    OUTPUT_DIR,
    "independent_model_comparison.csv"
)

results_df.to_csv(csv_path, index=False)

json_path = os.path.join(
    OUTPUT_DIR,
    "independent_model_metrics.json"
)

with open(json_path, "w") as f:
    json.dump(all_results, f, indent=4)

print("\n\n", "=" * 65)
print("FINAL INDEPENDENT MODEL COMPARISON")
print("=" * 65)

print(
    results_df[
        [
            "modality",
            "split",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc"
        ]
    ].to_string(index=False)
)

print("\nAll results saved to:")
print(OUTPUT_DIR)

print("\nINDEPENDENT PREDICTION ANALYSIS COMPLETED")
