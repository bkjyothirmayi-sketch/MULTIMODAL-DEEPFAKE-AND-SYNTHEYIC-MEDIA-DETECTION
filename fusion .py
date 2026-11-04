
import torch
import torch.nn as nn


# =====================================================
# MULTIMODAL INTERMEDIATE FUSION MODEL
# =====================================================

class MultimodalFusionModel(nn.Module):

    def __init__(
        self,
        image_dim=1280,
        audio_dim=1280,
        video_dim=1024,
        projection_dim=256,
        num_classes=2,
        dropout=0.3
    ):

        super(MultimodalFusionModel, self).__init__()

        # =============================================
        # IMAGE FEATURE PROJECTION
        # =============================================

        self.image_projection = nn.Sequential(

            nn.Linear(image_dim, projection_dim),

            nn.LayerNorm(projection_dim),

            nn.ReLU(),

            nn.Dropout(dropout)
        )

        # =============================================
        # AUDIO FEATURE PROJECTION
        # =============================================

        self.audio_projection = nn.Sequential(

            nn.Linear(audio_dim, projection_dim),

            nn.LayerNorm(projection_dim),

            nn.ReLU(),

            nn.Dropout(dropout)
        )

        # =============================================
        # VIDEO FEATURE PROJECTION
        # =============================================

        self.video_projection = nn.Sequential(

            nn.Linear(video_dim, projection_dim),

            nn.LayerNorm(projection_dim),

            nn.ReLU(),

            nn.Dropout(dropout)
        )

        # =============================================
        # FEATURE FUSION
        # =============================================

        fusion_dim = projection_dim * 3

        self.fusion_classifier = nn.Sequential(

            nn.Linear(fusion_dim, 512),

            nn.LayerNorm(512),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(512, 256),

            nn.LayerNorm(256),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(256, num_classes)
        )

    # =============================================
    # FORWARD PASS
    # =============================================

    def forward(
        self,
        image_features,
        audio_features,
        video_features
    ):

        image_out = self.image_projection(
            image_features
        )

        audio_out = self.audio_projection(
            audio_features
        )

        video_out = self.video_projection(
            video_features
        )

        # Concatenate all modality features
        fused_features = torch.cat(
            [
                image_out,
                audio_out,
                video_out
            ],
            dim=1
        )

        # Final classification
        output = self.fusion_classifier(
            fused_features
        )

        return output


# =====================================================
# MODEL TEST
# =====================================================

if __name__ == "__main__":

    model = MultimodalFusionModel()

    # Dummy input for architecture testing
    image = torch.randn(4, 1280)

    audio = torch.randn(4, 1280)

    video = torch.randn(4, 1024)

    output = model(
        image,
        audio,
        video
    )

    print("MODEL: Multimodal Intermediate Fusion")

    print("Image input:", image.shape)
    print("Audio input:", audio.shape)
    print("Video input:", video.shape)

    print("Fusion output:", output.shape)

    print("\nModel architecture:")
    print(model)

    print("\nFusion model test successful!")
