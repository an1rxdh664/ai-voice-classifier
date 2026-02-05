from fastapi import FastAPI, UploadFile, File, Body, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
import numpy as np
import os
import tempfile
import sys
import uvicorn
import requests
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv
from predict_utils import predict_from_file
from base64 import b64decode

# Env variables
load_dotenv()


# Create FastAPI app
app = FastAPI(
    title="AI Voice Detection API",
    description="Detects if a voice is AI-generated or human",
    version="1.0"
)

# ============ API Key Verification ============
async def verify_api_key(x_api_key: str = Header(None)):
    valid_key = os.getenv("API_KEY")
    if not valid_key:
        raise HTTPException(status_code=500, detail="API_KEY not configured")
    if x_api_key != valid_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True

# ============ Request Body Models ============
class AudioRequest(BaseModel):
    language: Optional[str] = None
    audioFormat: Optional[str] = None
    audioBase64: Optional[str] = None
    audio_url: Optional[str] = None  # keep old field to avoid breaking existing tests

# ============ API Endpoints ============
@app.get("/")
def read_root():
    """Welcome endpoint with available routes"""
    return {
        "message": "Welcome to AI Voice Detection API",
        "endpoints": {
            "predict": "/predict",
            "info": "/info",
            "docs": "/docs"
        }
    }


@app.get("/info")
def info():
    """API Information and feature details"""
    return {
        "name": "AI Voice Detection API",
        "version": "1.0",
        "description": "Detects if a voice is AI-generated or human",
        "features": {
            "MFCC": ["Mean", "Std Dev", "Variance", "Mean Difference"],
            "ZCR": ["Mean", "Std Dev", "Variance", "Mean Difference"],
            "Spectral Centroid": ["Mean", "Std Dev", "Variance", "Mean Difference"],
            "RMS": ["Mean", "Std Dev", "Variance", "Mean Difference"],
            "Pitch": ["Mean", "Std Dev", "Variance", "Mean Difference"]
        },
        "total_features": 20,
        "model": "Random Forest Classifier (50 trees)"
    }

@app.post("/api/voice-detection")
async def predict(
    request: AudioRequest = Body(...),
    _verify: bool = Depends(verify_api_key)
):
    temp_file = None
    try:
        # CASE 1 — Hackathon Format (audioBase64)
        if request.audioBase64:
            try:
                audio_bytes = b64decode(request.audioBase64)
            except Exception as e:
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "message": f"Invalid base64: {str(e)}"}
                )

            suffix = ".mp3"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_bytes)
                temp_file = tmp.name

        # CASE 2 — Audio URL (backward compatibility)
        elif request.audio_url:
            try:
                headers = {"User-Agent": "Mozilla/5.0"}
                response = requests.get(request.audio_url, headers=headers, timeout=15)
                response.raise_for_status()
                audio_content = response.content
            except requests.exceptions.RequestException as e:
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "message": f"Download failed: {str(e)}"}
                )

            suffix = ".mp3"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_content)
                temp_file = tmp.name

        else:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "No audioBase64 or audio_url provided"}
            )

        result = predict_from_file(temp_file)

        # Map your prediction to hackathon format
        classification = "AI_GENERATED" if result["label"] == "AI_GENERATED" else "HUMAN"
        confidence = result["confidence"]

        return {
            "status": "success",
            "language": request.language or "Unknown",
            "classification": classification,
            "confidenceScore": confidence,
            "explanation": f"Model confidence: {confidence * 100:.2f}%"
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Prediction error: {e}", file=sys.stderr)
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": f"Failed to process audio: {str(e)}"}
        )
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass

@app.post("/predict-upload")
async def predict_upload(
    file: UploadFile = File(...),
    _verify: bool = Depends(verify_api_key)
):
    temp_file = None
    try:
        # CASE 1 — Audio File Upload
        if file is not None:
            suffix = os.path.splitext(file.filename)[1] or ".mp3"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await file.read())
                temp_file = tmp.name
            filename = file.filename

        else:
            raise HTTPException(status_code=400, detail="No file provided")

        result = predict_from_file(temp_file)

        return {
            "filename": filename,
            "prediction": result["label"],
            "confidence": f"{result['confidence'] * 100:.2f}%",
            "confidence_score": result["confidence"],
            "probabilities": result["probabilities"]
        }

    except Exception as e:
        print(f"Prediction error: {e}", file=sys.stderr)
        return JSONResponse(
            status_code=400,
            content={"error": f"Failed to process audio: {str(e)}"}
        )
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)