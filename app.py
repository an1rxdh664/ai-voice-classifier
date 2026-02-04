from fastapi import FastAPI, UploadFile, File
from fastapi import Depends, Header, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import os
import tempfile
import sys
import uvicorn

from predict_utils import predict_from_file

from dotenv import load_dotenv
load_dotenv()  # loads variables from .env

# Create FastAPI app
app = FastAPI(
    title="AI Voice Detection API",
    description="Detects if a voice is AI-generated or human",
    version="1.0"
)

# API KEY VERIFICATION
def verify_api_key(x_api_key: str = Header(None)):
    server_key = os.getenv("API_KEY")

    if server_key is None:
        raise HTTPException(status_code=500, detail="API key not configured on server")

    if x_api_key != server_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    return True


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


@app.post("/predict")
async def predict(
        file: UploadFile = File(...),
        _verify: bool = Depends(verify_api_key)
    ):
    temp_file = None
    try:
        # Save uploaded file temporarily
        suffix = os.path.splitext(file.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            temp_file = tmp.name

        result = predict_from_file(temp_file)

        return {
            "filename": file.filename,
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
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
