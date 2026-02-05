# AI Voice Detection System

## Project Overview

This project is an AI Voice Detection System that classifies an input audio sample as either **AI-generated** or **Human-spoken**.  
The system works by extracting acoustic and spectral features from audio files and applying a trained machine learning model to make predictions.

The project includes:
- Audio feature extraction
- Model training
- Model inference
- A REST API built using FastAPI

---

## How the System Works

The complete workflow of the project is as follows:

1. An audio input is provided to the system (Base64, URL, or file upload)
2. The audio is temporarily saved on the server
3. Audio features are extracted using Librosa
4. Extracted features are standardized using a trained scaler
5. A Random Forest classifier predicts whether the voice is AI-generated or Human
6. The result is returned as a structured JSON response

---

## Audio Feature Extraction

For every audio file, the system extracts a fixed-length numerical feature vector.  
These features capture both time-domain and frequency-domain characteristics of the voice signal.

### Extracted Features

Each feature type includes:
- Mean
- Standard Deviation
- Variance
- Mean Difference

Feature groups:
- MFCC (Mel-Frequency Cepstral Coefficients)
- Zero Crossing Rate (ZCR)
- Spectral Centroid
- Root Mean Square (RMS) Energy
- Pitch (Fundamental Frequency)

**Total features per audio file: 20**

---

## Machine Learning Model

- Algorithm: Random Forest Classifier
- Number of estimators: 50
- Maximum depth: 5
- Feature scaling: StandardScaler

The model is trained using labeled audio samples:
- AI-generated voices
- Human voices

After training:
- The model is saved as `model.pkl`
- The scaler is saved as `scaler.pkl`

These are reused during prediction.

---

## API Implementation

The project exposes a REST API using FastAPI.

### API Security
- Requests are authenticated using an API key passed via the `x-api-key` header

---

## API Endpoints

### Root Endpoint


GET /

Returns basic API information and available routes.

---

### Voice Detection (Base64 or URL)


POST /api/voice-detection


**Request Body (JSON):**
```json
{
  "language": "en",
  "audioFormat": "mp3",
  "audioBase64": "<base64_encoded_audio>"
}


or

{
  "audio_url": "https://example.com/audio.mp3"
}


Response:

{
  "status": "success",
  "language": "en",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.92,
  "explanation": "Model confidence: 92.00%"
}

Voice Detection (File Upload)
POST /predict-upload


Accepts an audio file directly and returns the prediction result.

Project Structure
├── data/
│   ├── ai/                # AI-generated voice samples
│   └── human/             # Human voice samples
│
├── model/
│   ├── model.pkl          # Trained model
│   └── scaler.pkl         # Feature scaler
│
├── extract.py             # Feature extraction logic
├── model.py               # Model training script
├── predict_utils.py       # Prediction utilities
├── app.py                 # FastAPI application
├── requirements.txt       # Dependencies
└── README.md              # Project documentation

Training the Model

To train the model, place the audio samples in their respective folders:

data/ai/

data/human/

Then run:

python model.py


This will:

Extract features

Train the model

Save the trained model and scaler

Running the API

Start the FastAPI server using:

uvicorn app:app --host 0.0.0.0 --port 8000


The API documentation will be available at:

/docs

Dependencies

All required libraries are listed in requirements.txt.
Key dependencies include:

FastAPI

Librosa

NumPy

Scikit-learn

Joblib

Uvicorn

<h2>📌 Project Overview</h2>
<p>
AI Voice Classifier is a backend system designed to detect whether an audio sample is <b>human voice</b> or <b>AI-generated voice</b> using machine learning and audio signal processing.
</p>

<p>
The system exposes a <b>FastAPI REST endpoint</b> where users upload an audio file. The backend processes the file, extracts audio features, runs them through a trained ML model, and returns prediction and confidence score.
</p>

<h3>Real World Use Cases</h3>
<ul>
<li>Deepfake detection</li>
<li>Voice authentication validation</li>
<li>AI content moderation</li>
<li>Security & fraud prevention</li>
</ul>

<hr>

<h2>⚙️ How The System Works</h2>

<h3>🧠 Stage 1 — Training Pipeline</h3>
<ol>
<li>Audio datasets stored in data folders</li>
<li>Librosa extracts audio features</li>
<li>Features used to train ML model</li>
<li>Model and scaler saved as .pkl files</li>
</ol>

<h3>Extracted Features</h3>
<ul>
<li>MFCC (Mel Frequency Cepstral Coefficients)</li>
<li>Pitch</li>
<li>Spectral Features</li>
<li>Energy Features</li>
</ul>

<h3>Saved Artifacts</h3>
<ul>
<li><b>model.pkl</b> → Trained ML model</li>
<li><b>scaler.pkl</b> → Feature scaler</li>
</ul>

<hr>

<h3>🚀 Stage 2 — Prediction Pipeline</h3>
<ol>
<li>User uploads audio file</li>
<li>Server loads trained model + scaler</li>
<li>Features extracted from uploaded file</li>
<li>Features scaled</li>
<li>Prediction generated</li>
</ol>

<h3>Example API Response</h3>
<pre>
{
  "prediction": "AI",
  "confidence": 0.92
}
</pre>

<hr>

<h2>📂 Project Structure</h2>

<pre>
ai-voice-classifier/
│
├── data/
│   ├── ai/
│   ├── human/
│
├── model/
│   ├── model.pkl
│   └── scaler.pkl
│
├── app.py
├── extract.py
├── model.py
├── predict_utils.py
├── requirements.txt
├── README.md
└── .gitignore
</pre>

<hr>

<h2>📄 File Responsibilities</h2>

<h3>app.py</h3>
<ul>
<li>FastAPI server entry point</li>
<li>Handles API routes</li>
<li>Handles file upload</li>
<li>Calls prediction pipeline</li>
</ul>

<h3>extract.py</h3>
<ul>
<li>Loads audio using Librosa</li>
<li>Extracts MFCC, pitch, spectral and energy features</li>
<li>Converts audio to numerical feature vector</li>
</ul>

<h3>model.py</h3>
<ul>
<li>Loads dataset</li>
<li>Trains ML model</li>
<li>Saves model.pkl and scaler.pkl</li>
</ul>

<h3>predict_utils.py</h3>
<ul>
<li>Loads model and scaler</li>
<li>Prepares input features</li>
<li>Returns prediction and confidence</li>
</ul>

<hr>

<h2>🧪 Tech Stack</h2>

<h3>Backend</h3>
<ul>
<li>FastAPI</li>
<li>Python</li>
</ul>

<h3>Machine Learning</h3>
<ul>
<li>Scikit-Learn</li>
<li>Librosa</li>
<li>NumPy</li>
<li>Pandas</li>
<li>Joblib / Pickle</li>
</ul>

<hr>

<h2>📦 Installation & Setup</h2>

<h3>1️⃣ Clone Repository</h3>
<pre>
git clone https://github.com/YOUR_USERNAME/ai-voice-classifier.git
cd ai-voice-classifier
</pre>

<h3>2️⃣ Create Virtual Environment</h3>
<pre>
python -m venv venv
</pre>

<p>Activate:</p>

<p><b>Windows</b></p>
<pre>venv\Scripts\activate</pre>

<p><b>Linux / Mac</b></p>
<pre>source venv/bin/activate</pre>

<h3>3️⃣ Install Dependencies</h3>
<pre>
pip install -r requirements.txt
</pre>

<hr>

<h2>▶️ Running The Server</h2>

<pre>
uvicorn app:app --reload
</pre>

<p>Server URL:</p>
<pre>http://127.0.0.1:8000</pre>

<p>Swagger Docs:</p>
<pre>http://127.0.0.1:8000/docs</pre>

<hr>

<h2>📡 API Usage</h2>

<h3>Upload Audio For Prediction</h3>

<p><b>Endpoint</b></p>
<pre>POST /predict</pre>

<p><b>Request Type</b></p>
<ul>
<li>Form Data</li>
<li>Key: file</li>
<li>Value: Audio File (.wav recommended)</li>
</ul>

<h3>Example Response</h3>
<pre>
{
  "prediction": "Human",
  "confidence": 0.87
}
</pre>

<hr>

<h2>🧬 Model Details</h2>

<table>
<tr>
<th>Component</th>
<th>Purpose</th>
</tr>
<tr>
<td>Feature Extraction</td>
<td>Converts audio into numerical signals</td>
</tr>
<tr>
<td>Scaler</td>
<td>Normalizes feature values</td>
</tr>
<tr>
<td>ML Model</td>
<td>Classifies voice type</td>
</tr>
</table>

<hr>

<h2>📊 Future Improvements</h2>
<ul>
<li>Deep Learning Models (CNN / LSTM)</li>
<li>Multi-language Support</li>
<li>Real-time Streaming Detection</li>
<li>Docker Deployment</li>
<li>Cloud Hosting (AWS / GCP)</li>
</ul>

<hr>

<h2>🔒 Security Considerations</h2>
<ul>
<li>Validate file type</li>
<li>Limit upload size</li>
<li>Add API authentication</li>
<li>Rate limiting</li>
</ul>

<hr>

<h2>🤝 Contributing</h2>
<ol>
<li>Fork repository</li>
<li>Create feature branch</li>
<li>Commit changes</li>
<li>Open Pull Request</li>
</ol>


<h2>⭐ Support</h2>
<p>If you like this project, give it a star on GitHub.</p>

</body>
</html>
