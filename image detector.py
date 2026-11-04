
import os
import importlib.util
import torch
from PIL import Image
from torchvision import transforms

# =====================================================
# 1. DEVICE
# =====================================================

DEVICE = torch.device("cpu")

# =====================================================
# 2. FILE PATHS
# =====================================================

ARCHITECTURE_PATH = (
    r"D:\AI_Projects\intermediate fusion\model architecture.py"
)

IMAGE_MODEL_PATH = (
    r"D:\AI_Projects\mobilenet_results\best_mobilenet.pth"
)

# =====================================================
# 3. IMPORT MODEL ARCHITECTURE
# =====================================================

if not os.path.isfile(ARCHITECTURE_PATH):
    raise FileNotFoundError(ARCHITECTURE_PATH)

spec = importlib.util.spec_from_file_location(
    "model_architecture",
    ARCHITECTURE_PATH
)

architecture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(architecture)

create_image_model = architecture.create_image_model
load_checkpoint = architecture.load_checkpoint

print("Model architecture imported successfully!")

# =====================================================
# 4. IMAGE PREPROCESSING
# =====================================================

IMAGE_SIZE = 224

image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================================================
# 5. LOAD TRAINED IMAGE MODEL
# =====================================================

def load_image_detector():

    if not os.path.isfile(IMAGE_MODEL_PATH):
        raise FileNotFoundError(IMAGE_MODEL_PATH)

    model = create_image_model()

    model, checkpoint = load_checkpoint(
        model,
        IMAGE_MODEL_PATH
    )

    model = model.to(DEVICE)
    model.eval()

    class_mapping = checkpoint.get(
        "class_to_idx",
        {"fake": 0, "real": 1}
    )

    idx_to_class = {
        int(index): str(label).lower()
        for label, index in class_mapping.items()
    }

    print("\nIMAGE MODEL LOADED SUCCESSFULLY")
    print("Architecture: MobileNetV2")
    print("Class mapping:", idx_to_class)

    return model, idx_to_class

# =====================================================
# 6. IMAGE PREDICTION
# =====================================================

def predict_image(image_path, model, idx_to_class):

    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            "Image file not found: " + image_path
        )

    image = Image.open(image_path).convert("RGB")

    image_tensor = image_transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        predicted_index = int(
            torch.argmax(probabilities, dim=1).item()
        )

        original_label = idx_to_class[predicted_index]

        # Get probabilities using actual class mapping
        fake_index = next(
            index for index, label in idx_to_class.items()
            if label == "fake"
        )

        real_index = next(
            index for index, label in idx_to_class.items()
            if label == "real"
        )

        fake_probability = (
            probabilities[0, fake_index].item() * 100
        )

        real_probability = (
            probabilities[0, real_index].item() * 100
        )

        # =================================================
        # TEMPORARY LABEL REVERSAL
        # =================================================

        if original_label == "fake":
            predicted_label = "real"
        else:
            predicted_label = "fake"

        # Swap displayed probabilities
        fake_probability, real_probability = (
            real_probability,
            fake_probability
        )

        confidence = (
            probabilities[0, predicted_index].item() * 100
        )

    print("\n========== IMAGE DETECTION DEBUG ==========")
    print("Original model prediction:", original_label)
    print("Displayed prediction:", predicted_label)
    print("Fake probability:", round(fake_probability, 2))
    print("Real probability:", round(real_probability, 2))
    print("===========================================")

    return {
        "Modality": "Image",
        "Prediction": predicted_label.upper(),
        "Confidence": round(confidence, 2),
        "Fake Probability": round(fake_probability, 2),
        "Real Probability": round(real_probability, 2)
    }

# =====================================================
# 7. TEST IMAGE DETECTOR
# =====================================================

if __name__ == "__main__":

    model, idx_to_class = load_image_detector()

    image_path = input(
        "\nEnter image path: "
    ).strip().strip('"')

    result = predict_image(
        image_path,
        model,
        idx_to_class
    )

    print("\n========== IMAGE DETECTION RESULT ==========")

    for key, value in result.items():
        print(f"{key}: {value}")

    print("============================================")
