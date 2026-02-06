🎙️ AI Voice Classifier
📌 Project Overview

AI Voice Classifier is a backend system designed to detect whether an audio sample is human voice or AI-generated voice using machine learning and audio signal processing.

The system exposes a FastAPI REST endpoint where users upload an audio file. The backend processes the file, extracts audio features, runs them through a trained ML model, and returns:

Prediction (AI or Human)

Confidence Score

This project is built for real-world scenarios like:

Deepfake detection

Voice authentication validation

AI content moderation

Security & fraud prevention

⚙️ How The System Works
🧠 Stage 1 — Training Pipeline

Audio datasets are stored in:

data/ai/

data/human/

Audio is processed using Librosa to extract:

MFCC (Mel Frequency Cepstral Coefficients)

Pitch

Spectral Features

Energy Features

Extracted features are used to train a ML model:

Example: Random Forest Classifier

Two important artifacts are saved:

model.pkl → Trained ML model

scaler.pkl → Feature normalizer

🚀 Stage 2 — Prediction Pipeline

User uploads audio to FastAPI endpoint

Server:

Loads trained model + scaler

Extracts same features from uploaded audio

Scales features

Runs prediction

API returns:

{
  "prediction": "AI",
  "confidence": 0.92
}

📂 Project Structure
ai-voice-classifier/
│
├── data/
│   ├── ai/
│   │   └── insert_ai_data.txt
│   ├── human/
│   │   └── insert_human_data.txt
│
├── model/
│   ├── model.pkl        # Trained classifier
│   └── scaler.pkl       # Feature scaler
├── app.py               # FastAPI application
├── extract.py           # Feature extraction
├── model.py             # Model training
├── predict_utils.py     # Prediction logic
├── requirements.txt
├── README.md
└── .gitignore

📄 File Responsibilities
🔹 app.py

FastAPI server entry point
Handles:

API routes

File upload handling

Calling prediction pipeline

🔹 extract.py

Handles audio feature extraction:

Loads audio using Librosa

Extracts MFCC, pitch, spectral, energy features

Converts audio → numerical feature vector

🔹 model.py

Training pipeline:

Loads dataset

Trains ML model

Saves:

model.pkl

scaler.pkl

🔹 predict_utils.py

Prediction helper logic:

Loads saved model + scaler

Prepares input features

Returns prediction + confidence

🧪 Tech Stack
Backend

FastAPI

Python

Machine Learning

Scikit-Learn

Librosa

NumPy

Pandas

Joblib / Pickle

📦 Installation & Setup
1️⃣ Clone Repository
git clone https://github.com/YOUR_USERNAME/ai-voice-classifier.git
cd ai-voice-classifier

2️⃣ Create Virtual Environment
python -m venv venv


Activate:

Windows:

venv\Scripts\activate


Linux / Mac:

source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

▶️ Running The Server
uvicorn app:app --reload


Server runs at:

http://127.0.0.1:8000


Swagger Docs:

http://127.0.0.1:8000/docs

📡 API Usage
Upload Audio For Prediction

Endpoint

POST /predict


Request

Form Data

Key: file

Value: Audio File (.wav recommended)

Example Response
{
  "prediction": "Human",
  "confidence": 0.87
}

🧬 Model Details
Component	Purpose
Feature Extraction	Converts audio → numerical signals
Scaler	Normalizes feature values
ML Model	Classifies voice type
📊 Future Improvements

Add Deep Learning Models (CNN / LSTM)

Support More Languages

Real-time Streaming Detection

Docker Deployment

Cloud Hosting (AWS / GCP)

🔒 Security Considerations

Validate file type

Limit upload size

Add API authentication

Rate limiting

🤝 Contributing

Contributions are welcome.

Steps:

Fork repository

Create feature branch

Commit changes

Open Pull Request


⭐ If You Like This Project

Give it a star on GitHub.