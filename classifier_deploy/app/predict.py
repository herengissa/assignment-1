import io
import time

from fastapi import APIRouter, File, UploadFile, Depends
from sqlalchemy.orm import Session
from fastai.vision.all import PILImage

from app.database import get_db
from app.models import Prediction

router = APIRouter()

# Model is set by main.py at startup
learn = None


@router.post("/predict")
def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    start = time.perf_counter()

    image_bytes = file.file.read()
    img = PILImage.create(io.BytesIO(image_bytes))
    pred, idx, probs = learn.predict(img)

    processing_time_ms = int((time.perf_counter() - start) * 1000)

    probabilities = {
        learn.dls.vocab[i]: round(float(p), 4)
        for i, p in enumerate(probs)
    }

    # Log to database
    db_prediction = Prediction(
        prediction=str(pred),
        confidence=float(probs[idx]),
        probabilities=probabilities,
        processing_time_ms=processing_time_ms,
    )
    db.add(db_prediction)
    db.commit()

    return {
        "prediction": str(pred),
        "confidence": round(float(probs[idx]), 4),
        "probabilities": probabilities,
        "processing_time_ms": processing_time_ms,
    }
