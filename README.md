<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Voice Classifier</title>
    <style>
        body {
            font-family: Arial, Helvetica, sans-serif;
            margin: 40px;
            line-height: 1.6;
            background-color: #0d1117;
            color: #e6edf3;
        }
        h1, h2, h3 {
            color: #58a6ff;
        }
        code, pre {
            background-color: #161b22;
            padding: 10px;
            border-radius: 8px;
            display: block;
            overflow-x: auto;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin-top: 10px;
        }
        table, th, td {
            border: 1px solid #30363d;
        }
        th, td {
            padding: 8px;
            text-align: left;
        }
        ul {
            margin-left: 20px;
        }
    </style>
</head>

<body>

<h1>🎙️ AI Voice Classifier</h1>

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

<hr>

<h2>📜 License</h2>
<p>MIT License</p>

<hr>

<h2>👨‍💻 Author</h2>
<p>Aditya Sharma</p>

<hr>

<h2>⭐ Support</h2>
<p>If you like this project, give it a star on GitHub.</p>

</body>
</html>
