from fastapi import FastAPI, UploadFile, File, Body, HTTPException, Depends, Header, Request
from fastapi.responses import JSONResponse
import numpy as np
import os
import tempfile
import sys
import uvicorn
import requests
import base64
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv
from predict_utils import predict_from_file, model, scaler
from base64 import b64decode

# Env variables
load_dotenv()


# Create FastAPI app
app = FastAPI(
    title="AI Voice Detection API",
    description="Detects if a voice is AI-generated or human",
    version="1.0"
)

async def verify_api_key(x_api_key: str = Header(None)):
    valid_key = os.getenv("API_KEY")
    if not valid_key:
        raise HTTPException(status_code=500, detail="API_KEY not configured")
    if x_api_key != valid_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True

# Request Body Model :
class AudioRequest(BaseModel):
    language: Optional[str] = None
    audioFormat: Optional[str] = None
    audioBase64: Optional[str] = None
    audio_url: Optional[str] = None  # keep old field to avoid breaking existing tests


# Unified Error Responses :
def error_response(message, details=None, code=400):
    payload = {
        "status": "error",
        "message": message
    }
    if details:
        payload["details"] = details
    return JSONResponse(status_code=code, content=payload)


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

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

# Unified Audio Extractor :
async def get_audio_from_request(request: Request, file: UploadFile = None):
    body = {}

    try:
        body = await request.json()
    except:
        pass

    # If base64 audio
    if "audioBase64" in body:
        audio_b64 = body["audioBase64"]
        file_ext = body.get("audioformat", "mp3").strip(".")
        try:
            audio_bytes = base64.b64decode(audio_b64)
        except Exception as e:
            return None, error_response("Invalid base64 audio", str(e), 400)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}")
        tmp.write(audio_bytes)
        tmp.close()
        return tmp.name, None

    # If audio url
    if "audio_url" in body:
        url = body["audio_url"]
        try:
            res = requests.get(url)
            res.raise_for_status()
        except Exception as e:
            return None, error_response("Failed to download audio from the url", str(e), 400)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp.write(res.content)
        tmp.close()
        return tmp.name, None
    
    # Multipart file upload
    if file:
        try:
            contents = await file.read()
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
            tmp.write(contents)
            tmp.close()
            return tmp.name, None
        except Exception as e:
            return None, error_response("Failed to process uploaded file", str(e), 400)
        
    return None, error_response("No audioBase64, audio_url or file provided", code=400)


@app.post("/api/voice-detection")
async def predict(request: Request, _verify: bool = Depends(verify_api_key)):
    request_json = await request.json()
    temp_path, err = await get_audio_from_request(request)

    if err:
        return err

    try:
        result = predict_from_file(temp_path)

        classification = "AI_GENERATED" if result["label"] == "AI_GENERATED" else "HUMAN"
        confidence = result["confidence"]

        return {
            "status": "success",
            "language": request_json["language"] or "Unknown",
            "classification": classification,
            "confidenceScore": confidence,
            "explanation": f"Model confidence: {confidence * 100:.2f}%"
        }

    except HTTPException:
        raise
    except Exception as e:
        return error_response("Prediction failed", str(e), 500)
    
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


@app.post("/api/upload")
async def predict_upload(
    request: Request,
    file: UploadFile = File(...),
    _verify: bool = Depends(verify_api_key)
):
    temp_path, err = await get_audio_from_request(request, file)

    if err:
        return err
    
    try:

        result = predict_from_file(temp_path)

        return {
            "prediction": result["label"],
            "confidence": f"{result['confidence'] * 100:.2f}%",
            "confidence_score": result["confidence"],
            "probabilities": result["probabilities"]
        }

    except Exception as e:
        return error_response("Prediction failed", str(e), 500)
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)