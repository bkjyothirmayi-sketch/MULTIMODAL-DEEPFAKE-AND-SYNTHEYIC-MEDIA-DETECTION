
import torch
import torch.nn as nn
from torchvision import models


def create_image_model():
    model = models.mobilenet_v2(weights=None)

    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1280, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, 2)
    )

    return model


def create_audio_model():
    model = models.mobilenet_v2(weights=None)

    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1280, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, 2)
    )

    return model


def create_video_model():
    model = models.shufflenet_v2_x1_0(weights=None)

    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1024, 2)
    )

    return model


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    state_dict = checkpoint["model_state_dict"]

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model, checkpoint
