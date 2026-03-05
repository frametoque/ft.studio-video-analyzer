import os
import uuid
import shutil
import json
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from analyzer import analyze_video

app = FastAPI(title="Frametoque AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://studio.frametoque.online"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "temp_uploads"
TRAINING_FILE = "training_data.json"  # server side — all users, for ML
os.makedirs(UPLOAD_DIR, exist_ok=True)


def save_training_data(entry):
    """Save anonymized data for ML retraining — no personal info."""
    data = []
    if os.path.exists(TRAINING_FILE):
        with open(TRAINING_FILE, "r") as f:
            data = json.load(f)

    # Only save ML-relevant features, no filename or identity
    data.append({
        "overall_score":    entry.get("overall_score"),
        "filename":         entry.get("filename"), 
        "style":            entry.get("style", {}).get("detected"),
        "engagement_level": entry.get("engagement", {}).get("level"),
        "checks":           {c["name"]: c["score"] for c in entry.get("checks", [])},
    })

    with open(TRAINING_FILE, "w") as f:
        json.dump(data, f)


@app.get("/")
def root():
    return {"status": "Frametoque AI backend is running"}


@app.post("/analyze")
async def analyze(video: UploadFile = File(...)):
    allowed_types = ["video/mp4", "video/quicktime", "video/x-msvideo"]
    if video.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Invalid file type.")

    file_ext = os.path.splitext(video.filename)[1]
    temp_filename = f"{uuid.uuid4()}{file_ext}"
    temp_path = os.path.join(UPLOAD_DIR, temp_filename)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        results = analyze_video(temp_path)
        results["filename"] = video.filename

        # Save to server for ML training (anonymous)
        save_training_data(results)

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
