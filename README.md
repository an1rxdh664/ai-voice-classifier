# AI Voice Detection System

A machine learning-powered REST API that classifies audio samples as either AI-generated or human voice using acoustic feature extraction and Random Forest classification.

## Overview

This system analyzes audio files by extracting spectral and acoustic features, then applies a trained ML model to detect AI-generated voices. Built with FastAPI, it provides a simple API for real-time voice classification.

**Use Cases:**
- Deepfake detection
- Voice authentication validation
- AI content moderation
- Security and fraud prevention

## How It Works

### Training Pipeline
1. Audio samples organized in `data/ai/` and `data/human/` folders
2. Features extracted using Librosa (MFCC, pitch, spectral, energy)
3. Random Forest model trained on extracted features
4. Model and scaler saved as `.pkl` files

### Prediction Pipeline
1. Audio file uploaded via API
2. Features extracted and standardized
3. Model predicts classification with confidence score
4. JSON response returned

## Features Extracted

Each audio file is converted into a 20-feature vector including:
- MFCC (Mel-Frequency Cepstral Coefficients)
- Zero Crossing Rate
- Spectral Centroid
- RMS Energy
- Pitch (Fundamental Frequency)

Each feature includes mean, standard deviation, variance, and mean difference.

## Project Structure

```
ai-voice-detection/
├── data/
│   ├── ai/              # AI-generated samples
│   └── human/           # Human voice samples
├── model/
│   ├── model.pkl        # Trained classifier
│   └── scaler.pkl       # Feature scaler
├── app.py               # FastAPI application
├── extract.py           # Feature extraction
├── model.py             # Model training
├── predict_utils.py     # Prediction logic
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/ai-voice-detection.git
cd ai-voice-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Training the Model

```bash
# Place audio samples in data/ai/ and data/human/
python model.py
```

This extracts features, trains the model, and saves artifacts to the `model/` directory.

## Running the API

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Access the API at `http://localhost:8000` and interactive docs at `http://localhost:8000/docs`

## API Endpoints

### Voice Detection (JSON)
**POST** `/api/voice-detection`

```json
{
  "language": "en",
  "audioFormat": "mp3",
  "audioBase64": "<base64_encoded_audio>"
}
```

Or with URL:
```json
{
  "audio_url": "https://example.com/audio.mp3"
}
```

**Response:**
```json
{
  "status": "success",
  "language": "en",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.92,
  "explanation": "Model confidence: 92.00%"
}
```

### Voice Detection (File Upload)
**POST** `/predict-upload`

Upload audio file directly via form data.

### Authentication
All requests require an API key via the `x-api-key` header.

## Tech Stack

- **Backend:** FastAPI, Python
- **ML/Audio:** Scikit-learn, Librosa, NumPy, Pandas
- **Model:** Random Forest Classifier (50 estimators, max depth 5)
- **Scaling:** StandardScaler

## Model Details

| Component | Details |
|-----------|---------|
| Algorithm | Random Forest Classifier |
| Features | 20 acoustic/spectral features |
| Preprocessing | StandardScaler normalization |
| Output | Binary classification + confidence score |


---

⭐ If you find this project useful, please give it a star on GitHub!