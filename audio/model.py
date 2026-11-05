
import torch
import torch.nn as nn

from torchvision import models


# ==========================================
# 1. CONFIGURATION
# ==========================================

IMAGE_SIZE = 224
NUM_CLASSES = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_SAVE_PATH =r"D:\AI_Projects\audio dataset\model_save_path"


# ==========================================
# 2. BUILD MOBILENETV2 MODEL
# ==========================================

def build_model(pretrained=True):

    print("\nInitializing MobileNetV2...")
    print("Device:", DEVICE)

    # Load MobileNetV2
    weights = (
        models.MobileNet_V2_Weights.DEFAULT
        if pretrained else None
    )

    model = models.mobilenet_v2(
        weights=weights
    )

    # Freeze feature extraction layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classification layer
    input_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(input_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, NUM_CLASSES)
    )

    model = model.to(DEVICE)

    print("\n========== MODEL INFORMATION ==========")
    print("Model: MobileNetV2")
    print("Input size:", IMAGE_SIZE)
    print("Output classes:", NUM_CLASSES)
    print("Class mapping: Fake = 0, Real = 1")
    print("Device:", DEVICE)

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Total parameters:", f"{total_params:,}")
    print("Trainable parameters:", f"{trainable_params:,}")

    return model


# ==========================================
# 3. LOSS FUNCTION
# ==========================================

def get_loss_function():

    return nn.CrossEntropyLoss()


# ==========================================
# 4. OPTIMIZER
# ==========================================

def get_optimizer(model, learning_rate=0.0001):

    optimizer = torch.optim.Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=learning_rate
    )

    return optimizer


# ==========================================
# 5. LEARNING RATE SCHEDULER
# ==========================================

def get_scheduler(optimizer):

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    return scheduler


# ==========================================
# 6. SAVE MODEL
# ==========================================

def save_model(model, path=MODEL_SAVE_PATH):

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_mapping": {
                "fake": 0,
                "real": 1
            },
            "image_size": IMAGE_SIZE,
            "model_name": "MobileNetV2"
        },
        path
    )

    print("\nModel saved successfully!")
    print("Location:", path)


# ==========================================
# 7. LOAD MODEL
# ==========================================

def load_model(path=MODEL_SAVE_PATH):

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=True
    )

    model = build_model(pretrained=False)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("\nModel loaded successfully!")

    return model


# ==========================================
# 8. TEST MODEL INITIALIZATION
# ==========================================

if __name__ == "__main__":

    model = build_model(pretrained=True)

    print("\nTesting model input...")

    dummy_input = torch.randn(
        1, 3, IMAGE_SIZE, IMAGE_SIZE
    ).to(DEVICE)

    with torch.no_grad():

        output = model(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)

    print("\nModel initialization successful!")


