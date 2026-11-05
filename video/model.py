import torch
import torch.nn as nn

from torchvision import models


# ==========================================
# 1. CONFIGURATION
# ==========================================

IMAGE_SIZE = 224

NUM_CLASSES = 2

CLASS_NAMES = {
    0: "Fake",
    1: "Real"
}

MODEL_NAME = "ShuffleNetV2"

PRETRAINED = True

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# 2. BUILD SHUFFLENETV2 MODEL
# ==========================================

def build_model(
    num_classes=NUM_CLASSES,
    pretrained=PRETRAINED
):

    print("\nInitializing ShuffleNetV2...")

    # Load pretrained ImageNet weights
    if pretrained:

        weights = models.ShuffleNet_V2_X1_0_Weights.DEFAULT

    else:

        weights = None

    model = models.shufflenet_v2_x1_0(
        weights=weights
    )

    # Get classifier input features
    in_features = model.fc.in_features

    # Replace original ImageNet classifier
    model.fc = nn.Sequential(

        nn.Dropout(p=0.3),

        nn.Linear(
            in_features,
            num_classes
        )

    )

    print("\n========== MODEL INFORMATION ==========")

    print("Architecture:", MODEL_NAME)

    print("Input size:", IMAGE_SIZE)

    print("Input channels: 3")

    print("Output classes:", num_classes)

    print("Class mapping:", CLASS_NAMES)

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Total parameters:", f"{total_params:,}")

    print(
        "Trainable parameters:",
        f"{trainable_params:,}"
    )

    return model


# ==========================================
# 3. LOSS FUNCTION
# ==========================================

def get_loss_function():

    criterion = nn.CrossEntropyLoss()

    return criterion


# ==========================================
# 4. OPTIMIZER
# ==========================================

def get_optimizer(
    model,
    learning_rate=0.0001,
    weight_decay=0.0001
):

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=learning_rate,

        weight_decay=weight_decay

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
# 6. MODEL DEVICE
# ==========================================

def get_device():

    print("\n========== DEVICE INFORMATION ==========")

    print("Device:", DEVICE)

    return DEVICE


# ==========================================
# 7. MODEL SUMMARY
# ==========================================

def model_summary(model):

    print("\n========== MODEL SUMMARY ==========")

    print(model)

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    print("\nTotal parameters:", f"{total_params:,}")


# ==========================================
# 8. TEST MODEL
# ==========================================

if __name__ == "__main__":

    model = build_model()

    device = get_device()

    model = model.to(device)

    model.eval()

    # Create dummy image batch
    dummy_input = torch.randn(
        2, 3, IMAGE_SIZE, IMAGE_SIZE
    ).to(device)

    with torch.no_grad():

        output = model(dummy_input)

    print("\n========== MODEL TEST ==========")

    print("Input shape:", dummy_input.shape)

    print("Output shape:", output.shape)

    print("Predictions:", output.argmax(dim=1).tolist())

    print("\nShuffleNetV2 working successfully!")


