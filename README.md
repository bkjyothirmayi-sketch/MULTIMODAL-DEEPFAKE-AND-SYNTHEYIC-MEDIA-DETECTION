# Multimodal Deepfake and Synthetic Media Detection System

An AI-powered deep learning project designed to detect potentially manipulated or synthetic content across **images, audio recordings, and videos** using modality-specific neural networks, feature extraction, and an intermediate feature-fusion architecture.

The project integrates MobileNetV2-based image and audio detectors and a ShuffleNetV2-based video detector into a unified Streamlit application. It also includes a feature extraction and concatenation-based intermediate fusion workflow for multimodal experimentation.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Business Problem](#2-business-problem)
3. [Project Objectives](#3-project-objectives)
4. [Business, ML and Application Success Criteria](#4-business-ml-and-application-success-criteria)
5. [Dataset Overview](#5-dataset-overview)
6. [Dataset Preprocessing](#6-dataset-preprocessing)
7. [Model Architectures and Training](#7-model-architectures-and-training)
8. [Feature Extraction](#8-feature-extraction)
9. [Intermediate Feature Fusion](#9-intermediate-feature-fusion)
10. [Project Architecture](#10-project-architecture)
11. [Key Application Features](#11-key-application-features)
12. [Technologies and Libraries](#12-technologies-and-libraries)
13. [Project Directory Structure](#13-project-directory-structure)
14. [Installation and Setup](#14-installation-and-setup)
15. [Running the Application](#15-running-the-application)
16. [Prediction and Evaluation](#16-prediction-and-evaluation)
17. [Project Constraints](#17-project-constraints)
18. [Limitations](#18-limitations)
19. [Future Enhancements](#19-future-enhancements)
20. [Economic and Practical Impact](#20-economic-and-practical-impact)
21. [Applications](#21-applications)
22. [Disclaimer](#22-disclaimer)

---

## 1. Project Overview

The rapid advancement of generative artificial intelligence has made it increasingly easy to create realistic synthetic images, cloned voices, and manipulated videos.

These developments create challenges related to digital identity protection, misinformation, online fraud, and media authenticity.

This project develops a multimodal deepfake and synthetic media detection framework that analyzes three types of unstructured digital media:

- **Image Detection:** MobileNetV2
- **Audio Detection:** Mel-spectrogram representations with MobileNetV2
- **Video Detection:** Extracted video frames with ShuffleNetV2

The system follows an individual modality detection approach, where each input type is processed by its corresponding trained model.

In addition, extracted feature representations are used in an intermediate feature-fusion workflow based on feature projection and concatenation.

The detection modules are integrated into a unified Streamlit web application, allowing users to upload media and obtain Real/Fake predictions.

### Main Objective

To develop an integrated AI-based detection system that can analyze different forms of digital media and assist in identifying potentially authentic or manipulated content.

---

## 2. Business Problem

The increasing accessibility of AI-powered media generation technologies has created new opportunities for misuse.

Potential problems include:

- Identity impersonation and digital identity misuse.
- Voice cloning and audio-based financial scams.
- Manipulated videos and synthetic visual content.
- Misinformation and misleading digital media.
- Reputation damage and social engineering attacks.
- Difficulties in verifying the authenticity of online media.

Many traditional detection approaches focus on a single modality, limiting their ability to analyze different forms of synthetic content.

### Proposed Solution

This project provides a multimodal detection framework that supports image, audio, and video analysis through individual deep learning models, with an additional feature-level fusion architecture for combining learned representations.

---

## 3. Project Objectives

1. Collect publicly available image, audio, and video datasets.
2. Develop individual deep learning detection pipelines for each modality.
3. Preprocess unstructured media into model-compatible representations.
4. Train MobileNetV2 for image and audio classification.
5. Train ShuffleNetV2 for video frame classification.
6. Extract learned feature representations from trained model architectures.
7. Implement intermediate feature fusion using concatenation.
8. Perform binary classification into Real and Fake categories.
9. Evaluate model performance using classification metrics.
10. Integrate individual detection modules into a unified Streamlit application.

---

## 4. Business, ML and Application Success Criteria

### 4.1 Business Success Criteria

- Provide a unified platform for different media types.
- Support preliminary identification of potentially manipulated content.
- Reduce dependence on separate detection interfaces.
- Provide accessible and understandable prediction results.
- Support research and experimentation in digital media verification.

### 4.2 Machine Learning Success Criteria

- Achieve reliable binary classification for all three modalities.
- Extract meaningful features from image, audio, and video data.
- Evaluate individual model performance.
- Compare modality-specific detection performance.
- Implement feature-level concatenation for intermediate fusion.
- Assess precision, recall, F1-score, accuracy, and ROC-AUC.
- Minimize false-positive and false-negative predictions.

### 4.3 Application Success Criteria

- Load trained model checkpoints successfully.
- Accept supported image, audio, and video uploads.
- Perform the required preprocessing automatically.
- Generate individual modality predictions.
- Display classification and available confidence information.
- Provide an accessible Streamlit interface.

---

## 5. Dataset Overview

Three publicly available Kaggle datasets were used for this project.

| Modality | Dataset | Reported Dataset Size |
|---|---|---:|
| Images | DeepDetect 2025 | 41,583 |
| Audio | Deepfake Audio Dataset – Fake vs Real Speech | 16,793 |
| Videos | Deepfake Database | 19,457 |
| **Total** | **Three datasets** | **77,833** |

### 5.1 Image Dataset

**Dataset:** DeepDetect 2025

Source: https://www.kaggle.com/datasets/ayushmandatta1/deepdetect-2025/data

- Data type: Unstructured image data.
- Reported dataset size: 41,583.
- Classification: Real / Fake.
- Model: MobileNetV2.
- Input dimensions: 224 × 224 pixels.

### 5.2 Audio Dataset

**Dataset:** Deepfake Audio Dataset – Fake vs Real Speech

Source: https://www.kaggle.com/datasets/jayjoshi37/deepfake-audio-dataset-fake-vs-real-speech

- Data type: Unstructured audio recordings.
- Reported dataset size: 16,793.
- Classification: Real / Fake.
- Audio representation: Mel-spectrogram images.
- Model: MobileNetV2.

### 5.3 Video Dataset

**Dataset:** Deepfake Database

Source: https://www.kaggle.com/datasets/iamshahzaibkhan/deepfake-database/data

- Data type: Unstructured video data.
- Reported dataset size: 19,457.
- Classification: Real / Fake.
- Preprocessing: Video frame extraction.
- Model: ShuffleNetV2.

**Note:** The dataset sizes represent the counts reported for the collected datasets. They are not necessarily the final number of successfully processed training or test samples.

---

## 6. Dataset Preprocessing

Each modality requires a different preprocessing pipeline because image, audio, and video data have different structures.

### 6.1 Image Preprocessing

The image dataset is processed into RGB image tensors.

Preprocessing and transformations:

1. Load image files.
2. Resize images to 224 × 224 pixels.
3. Apply random horizontal flipping.
4. Apply random rotation.
5. Apply brightness adjustment.
6. Apply contrast adjustment.
7. Convert images to tensors.
8. Normalize image values.

These transformations help introduce variation during training and reduce sensitivity to certain visual changes.

### 6.2 Audio Preprocessing

Raw audio signals cannot be directly processed as ordinary RGB images by MobileNetV2.

Therefore, audio recordings are converted into Mel-spectrogram representations.

Processing workflow:

1. Load the audio recordings.
2. Read the audio signals.
3. Convert audio signals into Mel-spectrograms.
4. Represent the spectrograms as images.
5. Resize the spectrogram images to the required model input dimensions.
6. Apply image transformations.
7. Convert to tensors.
8. Normalize the resulting tensors.

The resulting Mel-spectrogram images are used as inputs to MobileNetV2.

### 6.3 Video Preprocessing

The video dataset contains unstructured video files.

Instead of processing the entire video as a single image, selected frames are extracted for classification.

Processing workflow:

1. Load video files.
2. Read video frames.
3. Extract selected frames.
4. Resize frames to the required input dimensions.
5. Apply random horizontal flipping.
6. Apply random rotation.
7. Apply brightness and contrast adjustments.
8. Convert frames into tensors.
9. Normalize the tensors.
10. Pass the processed frames to ShuffleNetV2.

Frame-level predictions are aggregated by the video detection pipeline to produce a video-level prediction.

---

## 7. Model Architectures and Training

The project uses lightweight convolutional neural network architectures to support image, audio, and video classification.

### 7.1 Image Detection – MobileNetV2

**Architecture:** MobileNetV2

MobileNetV2 is a lightweight convolutional neural network designed for efficient feature extraction and image classification.

It uses inverted residual blocks and depthwise separable convolutions to reduce computational requirements.

**Model configuration:**

| Component | Configuration |
|---|---|
| Architecture | MobileNetV2 |
| Input type | RGB images |
| Input size | 224 × 224 |
| Optimizer | Adam |
| Loss function | Cross-Entropy Loss |
| Classification | Binary |
| Output classes | Real / Fake |

**Training workflow:**

1. Load preprocessed image data.
2. Apply the required image transformations.
3. Pass the images through MobileNetV2.
4. Extract visual features.
5. Calculate classification loss.
6. Update model parameters using Adam.
7. Evaluate model performance.

### 7.2 Audio Detection – MobileNetV2

**Architecture:** MobileNetV2

The audio detection branch uses Mel-spectrogram images as input to MobileNetV2.

Mel-spectrograms provide a visual representation of the time-frequency characteristics of audio recordings.

**Model configuration:**

| Component | Configuration |
|---|---|
| Architecture | MobileNetV2 |
| Input type | Mel-spectrogram images |
| Optimizer | Adam |
| Loss function | Cross-Entropy Loss |
| Classification | Binary |
| Output classes | Real / Fake |

**Training workflow:**

1. Load audio recordings.
2. Convert audio signals into Mel-spectrogram images.
3. Apply image preprocessing and transformations.
4. Pass the spectrogram images through MobileNetV2.
5. Extract audio-related features.
6. Calculate classification loss.
7. Update model parameters using Adam.
8. Evaluate classification performance.

### 7.3 Video Detection – ShuffleNetV2

**Architecture:** ShuffleNetV2

ShuffleNetV2 is a lightweight convolutional neural network designed for computational efficiency.

The video detection branch uses extracted video frames as model inputs.

**Model configuration:**

| Component | Configuration |
|---|---|
| Architecture | ShuffleNetV2 |
| Input type | Extracted video frames |
| Input size | 224 × 224 |
| Optimizer | Adam |
| Loss function | Cross-Entropy Loss |
| Classification | Binary |
| Output classes | Real / Fake |

**Training workflow:**

1. Load video files.
2. Extract selected video frames.
3. Apply image preprocessing.
4. Generate frame tensors.
5. Pass frames through ShuffleNetV2.
6. Extract visual representations.
7. Calculate classification loss.
8. Update model parameters using Adam.
9. Evaluate frame-level and video-level predictions.

---

## 8. Feature Extraction

Feature extraction is an important component of the project because it converts raw model inputs into learned numerical representations.

The trained models are used as feature extractors.

### Individual feature extraction

| Modality | Feature extractor | Representation |
|---|---|---|
| Image | MobileNetV2 | Image feature vector |
| Audio | MobileNetV2 | Mel-spectrogram feature vector |
| Video | ShuffleNetV2 | Video-frame feature vector |

The extracted features are stored and prepared for downstream classification and fusion experiments.

The feature extraction modules include:

- `feature extraction.py`
- `image feature extraction.py`
- `audio_feature extraction.py`
- `video_feature extraction.py`

Feature extraction allows the project to reuse learned representations without repeatedly processing the complete raw datasets during downstream experiments.

---

## 9. Intermediate Feature Fusion

### 9.1 Fusion Strategy

The project includes an intermediate feature-fusion architecture based on feature projection and concatenation.

Instead of combining only the final prediction probabilities, the architecture combines learned feature representations from the individual modalities.

### 9.2 Fusion Architecture

The three modality-specific feature vectors are processed through separate projection layers.

The architecture uses the following input feature dimensions:

| Modality | Input feature dimension | Projection dimension |
|---|---:|---:|
| Image | 1280 | 256 |
| Audio | 1280 | 256 |
| Video | 1024 | 256 |

Each modality passes through:

- Linear projection.
- Layer normalization.
- ReLU activation.
- Dropout.

### 9.3 Feature Concatenation

After projection, the three feature representations are concatenated along the feature dimension.

\[
F_{\text{fused}} =
[F_{\text{image}},F_{\text{audio}},F_{\text{video}}]
\]

The resulting feature dimension is:

\[
256 + 256 + 256 = 768
\]

### 9.4 Fusion Classification Network

The concatenated feature vector is passed through the fusion classification network.

The implemented architecture contains:

1. Linear layer: 768 → 512.
2. Layer normalization.
3. ReLU activation.
4. Dropout.
5. Linear layer: 512 → 256.
6. Layer normalization.
7. ReLU activation.
8. Dropout.
9. Final classification layer: 256 → 2.

The final output represents the two classification categories:

- Real
- Fake

### 9.5 Fusion Workflow

```text
             IMAGE INPUT
                  |
              MobileNetV2
                  |
            Image Features
               1280
                  |
           Projection Layer
                256
                  |
                  |
                  +--------------------+
                                       |
             AUDIO INPUT               |
                  |                    |
              MobileNetV2              |
                  |                    |
            Audio Features             |
               1280                    |
                  |                    |
           Projection Layer            |
                256                    |
                  |                    |
                  +--------------------+ 
                                       |
             VIDEO INPUT               |
                  |                    |
              ShuffleNetV2             |
                  |                    |
            Video Features             |
               1024                    |
                  |                    |
           Projection Layer            |
                256                    |
                  |                    |
                  +--------------------+
                                       |
                              FEATURE CONCATENATION
                                       |
                                 768 FEATURES
                                       |
                             FUSION CLASSIFIER
                                       |
                              FINAL CLASSIFICATION
                                       |
                                  REAL / FAKE
```

**Implementation note:** The concatenation-based PyTorch fusion architecture is defined in `fusion .py`. The feature preparation and modality-wise evaluation workflow is implemented separately in `intermediate fusion.py`. The fusion experiment and the individual Streamlit inference interface are documented as distinct components.

---

## 10. Project Architecture

The overall project follows a modular machine learning architecture.

### Main Pipeline

```text
             DATA COLLECTION
                    |
          +---------+---------+
          |         |         |
        IMAGE     AUDIO     VIDEO
          |         |         |
      Preprocess  Mel-Spec  Frame Extraction
          |         |         |
      MobileNetV2 MobileNetV2 ShuffleNetV2
          |         |         |
      Feature     Feature    Feature
      Extraction  Extraction Extraction
          |         |         |
          +---------+---------+
                    |
             FEATURE EXTRACTION
                    |
          INTERMEDIATE FEATURE FUSION
                    |
             FEATURE CONCATENATION
                    |
              FUSION CLASSIFIER
                    |
             BINARY CLASSIFICATION
```

The individual inference workflow is integrated through Streamlit:

```text
              STREAMLIT APPLICATION
                       |
           +-----------+-----------+
           |           |           |
         IMAGE       AUDIO       VIDEO
           |           |           |
       Individual  Individual  Individual
       Detector    Detector    Detector
           |           |           |
       MobileNetV2 MobileNetV2 ShuffleNetV2
           |           |           |
           +-----------+-----------+
                       |
                 PREDICTION
                       |
                  REAL / FAKE
```

---

## 11. Key Application Features

### Image Detection

- Upload image files.
- Apply image preprocessing.
- Use MobileNetV2 for classification.
- Generate Real/Fake predictions.
- Display available probability and confidence information.

### Audio Detection

- Upload audio recordings.
- Convert audio signals into Mel-spectrogram representations.
- Apply the required preprocessing.
- Use MobileNetV2 for classification.
- Generate Real/Fake predictions.

### Video Detection

- Upload video files.
- Extract selected frames.
- Apply frame preprocessing.
- Use ShuffleNetV2 for classification.
- Aggregate frame-level predictions into a video-level result.

### Unified Streamlit Application

The Streamlit interface provides separate detection tabs for:

- Image Detection
- Audio Detection
- Video Detection

The application integrates the three individual detector modules into one interface.

---

## 12. Technologies and Libraries

| Technology / Library | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Deep learning model training and inference |
| Torchvision | Model architectures and image transformations |
| MobileNetV2 | Image and audio classification |
| ShuffleNetV2 | Video frame classification |
| NumPy | Numerical computations |
| Pandas | Dataset and evaluation result handling |
| Pillow (PIL) | Image processing |
| OpenCV | Video frame extraction |
| Librosa | Audio processing and Mel-spectrogram generation |
| Scikit-learn | Evaluation metrics and feature classifiers |
| Joblib | Saving and loading feature classifiers |
| Streamlit | Web application |
| Matplotlib | Data visualization |
| Git | Version control |
| GitHub | Source-code hosting |

---

## 13. Project Directory Structure

The following reflects the existing repository files.

```text
MULTIMODAL-DEEPFAKE-AND-SYNTHEYIC-MEDIA-DETECTION/
│
├── app.py
│
├── image detector.py
├── audio detector.py
├── video detector.py
│
├── model architecture.py
│
├── image feature extraction.py
├── audio_feature extraction.py
├── video_feature extraction.py
├── feature extraction.py
│
├── audio.py
├── dataset.py
├── preprocess.py
│
├── fusion .py
├── intermediate fusion.py
│
├── stremlit .app.py
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

### Main File Descriptions

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit application |
| `image detector.py` | Image prediction module |
| `audio detector.py` | Audio prediction module |
| `video detector.py` | Video prediction module |
| `model architecture.py` | Model architecture definitions |
| `image feature extraction.py` | Image feature extraction |
| `audio_feature extraction.py` | Audio feature extraction |
| `video_feature extraction.py` | Video feature extraction |
| `feature extraction.py` | Feature extraction workflow |
| `audio.py` | Audio-related processing |
| `dataset.py` | Dataset handling |
| `preprocess.py` | Preprocessing utilities |
| `fusion .py` | Multimodal concatenation fusion architecture |
| `intermediate fusion.py` | Feature-based classifier training and evaluation |
| `stremlit .app.py` | Additional Streamlit application script |
| `requirements.txt` | Python dependencies |

**Note:** The trained checkpoints, original datasets, extracted feature files, and generated evaluation outputs are maintained separately from the source-code repository.

---

## 14. Installation and Setup

### Prerequisites

- Python environment compatible with the installed PyTorch version.
- Git.
- Anaconda or Miniconda (recommended).
- Internet connection for dependency installation.
- Trained model checkpoints for inference.

### Step 1: Clone the Repository

```bash
git clone https://github.com/bkjyothirmayi-sketch/MULTIMODAL-DEEPFAKE-AND-SYNTHEYIC-MEDIA-DETECTION.git
```

### Step 2: Navigate to the Repository

```bash
cd MULTIMODAL-DEEPFAKE-AND-SYNTHEYIC-MEDIA-DETECTION
```

### Step 3: Create a Virtual Environment

```bash
conda create -n deepfake python=3.10
```

Activate the environment:

```bash
conda activate deepfake
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

The repository's dependency list includes:

```text
torch
torchvision
numpy
pandas
Pillow
opencv-python
librosa
scikit-learn
streamlit
matplotlib
```

### Step 5: Configure Trained Model Checkpoints

The trained model checkpoint files are maintained separately from the source-code repository.

Before running inference:

1. Obtain the corresponding trained model checkpoints.
2. Place them in the locations expected by the detector scripts.
3. Verify the model paths in the detector files.
4. Ensure the checkpoint architecture and class mapping match the trained models.

The application cannot perform trained-model inference without the required checkpoints.

---

## 15. Running the Application

From the repository root, activate the environment and execute:

```bash
python -m streamlit run app.py
```

Streamlit starts a local web server and displays a browser URL.

The default local address is generally:

```text
http://localhost:8501
```

### Application Usage

1. Open the Streamlit application.
2. Select Image Detection, Audio Detection, or Video Detection.
3. Upload the appropriate media file.
4. Start the detection process.
5. Allow the application to preprocess the input.
6. View the Real/Fake classification.
7. Review the available confidence and probability values.

---

## 16. Prediction and Evaluation

### 16.1 Individual Prediction

Each modality-specific detector produces a binary classification.

| Modality | Model | Output |
|---|---|---|
| Image | MobileNetV2 | Real / Fake |
| Audio | MobileNetV2 | Real / Fake |
| Video | ShuffleNetV2 | Real / Fake |

For video analysis, selected frame predictions are aggregated into a video-level result.

### 16.2 Evaluation Metrics

The project uses the following classification metrics:

| Metric | Purpose |
|---|---|
| Accuracy | Measures overall correct classification |
| Precision | Measures the proportion of predicted Fake samples that are actually Fake |
| Recall | Measures the proportion of actual Fake samples correctly identified |
| F1-score | Harmonic mean of precision and recall |
| ROC-AUC | Measures the model's ability to distinguish the two classes |
| Confusion Matrix | Shows correct and incorrect classifications |

### 16.3 Evaluation Workflow

The feature-based evaluation workflow:

1. Loads the extracted image, audio, and video features.
2. Loads their corresponding labels and source paths.
3. Prepares training, validation, and testing splits.
4. Trains feature-based classifiers for individual modalities.
5. Generates predictions on the evaluation data.
6. Calculates accuracy, precision, recall, F1-score, and ROC-AUC.
7. Generates confusion matrices and classification reports.
8. Saves prediction outputs and performance summaries.

The evaluation script generates:

- `independent_model_comparison.csv`
- `independent_model_metrics.json`

These are generated evaluation outputs, not files currently listed in the source-code repository.

### 16.4 Evaluation Results

 Accuracy 
 Precision 
 Recall 
 F1-score 
 ROC-AUC 

The concatenation-based fusion architecture is a separate fusion component. Its evaluation results should be reported separately when measured on a defined multimodal test set.

**Evaluation note:** Do not interpret individual modality scores as multimodal fusion performance. The fusion classifier requires compatible features from all three modalities.

---

## 17. Project Constraints

### Technical Constraints

- Different input formats require different preprocessing pipelines.
- Deep learning training requires computational resources.
- Video frame extraction increases processing time.
- Model checkpoints require sufficient storage.
- Feature dimensions must be compatible before concatenation.
- Inference performance depends on available CPU and memory resources.

### Dataset Constraints

- Dataset quality and class distribution can influence model performance.
- Compression and recording conditions may affect input quality.
- Dataset-specific patterns may limit generalization.
- Training and test data must be separated appropriately.

### Application Constraints

- Large media files may require longer processing times.
- Deployment platforms may impose memory and upload-size restrictions.
- Model checkpoints must be available at inference time.
- Application performance depends on the hosting environment.

---

## 18. Limitations

1. The models are trained on specific datasets and may not generalize to all unseen deepfake generation techniques.
2. Image detection can be affected by image compression, resizing, and low resolution.
3. Audio detection can be affected by background noise, recording quality, and different speech conditions.
4. Video detection relies on selected frames and may miss temporal inconsistencies between sampled frames.
5. The quality of feature fusion depends on the extracted feature representations.
6. The current Streamlit interface exposes individual modality detection rather than a unified three-input fusion inference workflow.
7. Model confidence scores are not necessarily calibrated probabilities of actual authenticity.
8. The system is a research prototype and has not been validated as a forensic evidence tool.

---

## 19. Future Enhancements

- Integrate the concatenation-based fusion architecture into the deployed Streamlit inference workflow.
- Improve multimodal feature alignment and fusion.
- Introduce attention-based fusion mechanisms.
- Explore transformer-based architectures.
- Incorporate temporal learning for video detection.
- Explore Wav2Vec2 and advanced audio representations.
- Improve robustness against unseen synthetic media generation techniques.
- Implement missing-modality handling.
- Integrate explainable AI techniques such as Grad-CAM.
- Add experiment tracking and model versioning using MLflow.
- Introduce monitoring and performance tracking.
- Optimize deployment for scalable cloud infrastructure.
- Expand datasets with additional real-world and synthetic media samples.

---

## 20. Economic and Practical Impact

The project uses lightweight architectures such as MobileNetV2 and ShuffleNetV2 to support computationally efficient media classification.

Potential practical benefits include:

- Preliminary digital media screening.
- Lower-cost experimentation with deepfake detection.
- Reusable modality-specific detection models.
- A single interface for different media types.
- Support for research in media authenticity and digital security.

The project may be useful as an initial screening tool, but its real-world economic benefits require further testing and validation.

---

## 21. Applications

Potential application areas include:

- Digital media authenticity research.
- Deepfake and synthetic media analysis.
- Online misinformation screening.
- Digital identity protection research.
- Audio-based fraud detection research.
- Cybersecurity education.
- Multimedia forensic research.

---

## 22. Disclaimer

This project is developed for educational, research, and prototype purposes.

The model predictions are automated estimates and may contain false positives and false negatives.

A Real prediction does not guarantee that media is authentic, and a Fake prediction does not establish that media has been manipulated.

The system should not be used as the sole basis for legal, financial, security, or other high-impact decisions.

Dataset ownership, licensing, and usage conditions remain subject to the respective Kaggle dataset providers.

---

## Author and Repository

**Project:** Multimodal Deepfake and Synthetic Media Detection System

**GitHub Repository:**

https://github.com/bkjyothirmayi-sketch/MULTIMODAL-DEEPFAKE-AND-SYNTHEYIC-MEDIA-DETECTION

**Technologies:** Python, PyTorch, MobileNetV2, ShuffleNetV2, OpenCV, Librosa, Scikit-learn, and Streamlit.

**Project Focus:** Image, Audio, Video, Feature Extraction, Intermediate Fusion, Binary Classification, and Web Application Development.

---

*Developed as an AI/ML project exploring multimodal deepfake and synthetic media detection.*
