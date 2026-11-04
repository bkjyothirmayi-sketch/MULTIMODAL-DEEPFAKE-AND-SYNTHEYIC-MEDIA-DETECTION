# Multimodal Deepfake and Synthetic Media Detection System

An AI-powered multimedia detection application that uses deep learning to identify potentially manipulated or AI-generated images, audio recordings, and videos.

The project integrates three independently trained deep learning models into a unified Streamlit application, allowing users to upload media files and obtain individual Real/Fake predictions with confidence scores.

---

## 1. Project Overview

The rapid development of generative artificial intelligence has made it increasingly easy to create realistic synthetic images, cloned voices, and manipulated videos.

Although these technologies have many positive applications, they also introduce challenges involving misinformation, digital identity misuse, media authenticity, and online security.

This project aims to develop a deep learning-based detection system that analyzes three types of digital media:

- Images
- Audio recordings
- Videos

Each modality is processed using its corresponding trained deep learning model. The predictions are presented through a single user-friendly web interface developed using Streamlit.

### Main Objective

To build an integrated AI-based system that can analyze different forms of digital media and assist in identifying potentially real or manipulated content.

---

## 2. Key Features

### Image Deepfake Detection
- Uses MobileNetV2 architecture.
- Accepts image files as input.
- Applies image preprocessing and normalization.
- Predicts whether an image is Real or Fake.
- Displays fake probability, real probability, and confidence score.

### Audio Deepfake Detection
- Uses a deep learning model based on MobileNetV2.
- Converts audio signals into Mel spectrogram representations.
- Uses Librosa for audio processing.
- Analyzes spectrogram images for classification.
- Displays Real/Fake predictions and probability scores.

### Video Deepfake Detection
- Uses ShuffleNetV2 architecture.
- Accepts video files as input.
- Extracts selected frames from the video.
- Applies image preprocessing to sampled frames.
- Classifies the extracted frames.
- Aggregates frame-level predictions to generate a video-level result.

### Streamlit Web Application
- Provides a unified interface for image and audio detection.
- Supports media file uploads.
- Displays classification results.
- Shows confidence and probability values.
- Integrates independently trained detection modules.

---

## 3. System Architecture

The application follows a modular architecture in which each media type has its own preprocessing pipeline, trained model, and prediction function.

```text
                 USER INPUT
                     |
          -----------------------
          |          |          |
        IMAGE      AUDIO      VIDEO
          |          |          |
     Preprocessing  Mel       Frame
          |       Spectrogram  Extraction
          |          |          |
          v          v          v
      MobileNetV2 MobileNetV2 ShuffleNetV2
          |          |          |
          v          v          v
      Image       Audio       Video
      Prediction  Prediction  Prediction
          |          |          |
          -----------------------
                     |
             Prediction Results
                     |
             Streamlit Interface
```

**Important:** The current implementation integrates three independent modality-specific detectors. It does not yet implement a jointly trained multimodal feature-fusion network.

---

## 4. Deep Learning Models

### 4.1 Image Detection – MobileNetV2

MobileNetV2 is a lightweight convolutional neural network designed for efficient image classification.

It uses depthwise separable convolutions and inverted residual blocks to reduce computational requirements.

**Workflow:**
1. Upload an image.
2. Load and resize the image.
3. Convert the image into a tensor.
4. Normalize using the required preprocessing parameters.
5. Pass the tensor through the trained MobileNetV2 model.
6. Apply Softmax to obtain class probabilities.
7. Generate the final Real/Fake prediction.

**Output:**
- Predicted class
- Fake probability
- Real probability
- Confidence score

### 4.2 Audio Detection – Mel Spectrogram and MobileNetV2

Audio recordings contain temporal and frequency-based information that can be represented using spectrograms.

The audio detection pipeline converts audio signals into Mel spectrogram images before classification.

**Workflow:**
1. Upload an audio recording.
2. Load the audio using Librosa.
3. Convert the audio to mono.
4. Resample to 22,050 Hz.
5. Generate a Mel spectrogram.
6. Convert the spectrogram into a decibel representation.
7. Resize and normalize the spectrogram image.
8. Pass it through the trained MobileNetV2 model.
9. Apply Softmax for classification.

**Output:**
- Predicted class
- Fake probability
- Real probability
- Confidence score

### 4.3 Video Detection – ShuffleNetV2

ShuffleNetV2 is a computationally efficient convolutional neural network suitable for resource-constrained environments.

The video detector analyzes selected frames rather than processing the entire video as a single spatiotemporal input.

**Workflow:**
1. Upload a video.
2. Read the video using OpenCV.
3. Extract selected frames.
4. Resize and preprocess the frames.
5. Pass each frame through the trained ShuffleNetV2 model.
6. Obtain frame-level predictions.
7. Aggregate the predictions to produce a video-level classification.

**Output:**
- Video-level prediction
- Fake probability
- Real probability
- Confidence score

---

## 5. Technologies and Libraries

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Image Model | MobileNetV2 |
| Audio Processing | Librosa |
| Video Processing | OpenCV |
| Video Model | ShuffleNetV2 |
| Image Processing | PIL, NumPy |
| Data Processing | Pandas |
| Machine Learning Utilities | Scikit-learn |
| Visualization | Matplotlib |
| Web Application | Streamlit |
| Version Control | Git and GitHub |

---

## 6. Project Directory Structure

```text
MULTIMODAL-DEEPFAKE-AND-SYNTHETIC-MEDIA-DETECTION/
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

*Note: Model checkpoints, datasets, extracted features, and generated outputs are maintained separately and are not included in the source-code repository.*

---

## 7. Installation and Setup

### Prerequisites

- Python 3.10 or a compatible version supported by the installed PyTorch and other dependencies.
- Git
- Anaconda or Miniconda (recommended)
- Internet connection for installing dependencies

### Step 1: Clone the Repository

```bash
git clone https://github.com/bkjyothirmayi-sketch/MULTIMODAL-DEEPFAKE-AND-SYNTHEYIC-MEDIA-DETECTION.git
```

### Step 2: Navigate to the Project

```bash
cd MULTIMODAL-DEEPFAKE-AND-SYNTHEYIC-MEDIA-DETECTION
```

### Step 3: Create a Virtual Environment

```bash
conda create -n deepfake python=3.10
```

Activate it:

```bash
conda activate deepfake
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

**PyTorch note:** Install a PyTorch version compatible with your operating system and hardware. For CPU-only execution, use the official CPU installation instructions from the PyTorch website.

### Step 5: Configure Model Checkpoints

The trained model checkpoint files are not included in this repository.

Place your trained checkpoints in the appropriate locations and update the model paths in the detector scripts.

The application requires the corresponding trained checkpoints and the matching model architecture definitions.

---

## 8. Running the Application

After installing dependencies and configuring the model paths, navigate to the project directory.

Run:

```bash
python -m streamlit run app.py
```

Streamlit will start the application and display a local URL in the terminal.

Open the displayed URL in your web browser.

### Using the Application

1. Open the Streamlit application.
2. Select the available detection section.
3. Upload the supported media file.
4. Start the detection process.
5. View the predicted classification.
6. Review the confidence and probability scores.

---

## 9. Prediction Results

The detection models return the following information:

| Output | Description |
|---|---|
| Modality | Image, Audio, or Video |
| Prediction | Real or Fake |
| Confidence | Highest predicted class probability |
| Fake Probability | Model probability assigned to Fake |
| Real Probability | Model probability assigned to Real |

**Important:** Confidence scores are based on the model's Softmax output and are not necessarily calibrated probabilities of actual media authenticity.

---

## 10. Model Evaluation

The models should be evaluated independently using appropriate test datasets.

Recommended evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC

Evaluation should be performed separately for image, audio, and video detection.

Actual evaluation results should be added to this section after confirming the final test-set metrics and class mappings for each trained checkpoint.

---

## 11. Current Limitations

- Each modality uses a separately trained model.
- The current application does not perform learned feature-level multimodal fusion.
- Video predictions are based on sampled frames and may miss manipulations between sampled frames.
- Detection performance depends on the quality and diversity of the training datasets.
- Models may not generalize equally well to unseen manipulation techniques.
- Softmax confidence should not be interpreted as a guarantee of authenticity.
- The trained model checkpoint files must be configured separately before running the application.

---

## 12. Future Enhancements

Potential improvements include:

- Implementing early and late feature fusion.
- Developing a jointly trained multimodal detection architecture.
- Improving robustness against unseen deepfake generation methods.
- Adding temporal analysis for video detection.
- Exploring transformer-based image, audio, and video models.
- Integrating explainable AI techniques.
- Adding model performance monitoring.
- Implementing model versioning and experiment tracking using MLflow.
- Deploying the application to a cloud platform.
- Developing an API for integration with external applications.

---

## 13. Applications

Potential applications include:

- Digital media authenticity analysis
- Multimedia forensics research
- AI-generated content detection
- Online misinformation analysis
- Digital identity protection research
- Cybersecurity research and education

---

## 14. Disclaimer

This project is developed for educational and research purposes.

The predictions generated by the models are automated estimates and may contain false positives or false negatives. The system should not be used as the sole basis for legal, financial, security, or other high-impact decisions.

A Real prediction does not guarantee that media is authentic, and a Fake prediction does not establish that media has been manipulated.

---

## 15. Author and Repository

**Project:** Multimodal Deepfake and Synthetic Media Detection System

**GitHub:** [bkjyothirmayi-sketch](https://github.com/bkjyothirmayi-sketch)

**Repository:** [MULTIMODAL-DEEPFAKE-AND-SYNTHEYIC-MEDIA-DETECTION](https://github.com/bkjyothirmayi-sketch/MULTIMODAL-DEEPFAKE-AND-SYNTHEYIC-MEDIA-DETECTION)

---

## 16. Acknowledgements

This project uses open-source deep learning frameworks, image and audio processing libraries, and computer vision tools.

Special acknowledgement to the developers and research communities behind PyTorch, Torchvision, Librosa, OpenCV, NumPy, and Streamlit.
