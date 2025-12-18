from fastapi import FastAPI, File, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from PIL import Image
import numpy as np
import tensorflow as tf
import os

from database import SessionLocal, engine
from models import Prediction
import models

CONFIDENCE_THRESHOLD = 60.0  # percentage

# -------------------------
# APP INIT
# -------------------------
app = FastAPI(title="Civic Issue ML API")

# -------------------------
# CORS (VERY IMPORTANT)
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:63342",
        "http://127.0.0.1:63342",
        "http://127.0.0.1",
        "*"  # safe for college project
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# DB INIT
# -------------------------
models.Base.metadata.create_all(bind=engine)

# -------------------------
# LOAD MODEL ONCE
# -------------------------
CLASS_NAMES = ['electricpoles', 'fallentrees', 'garbage', 'pothole']
IMG_SIZE = (224, 224)

model = tf.keras.models.load_model("model/civic_model.keras")

# -------------------------
# DB DEPENDENCY
# -------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -------------------------
# PREDICT ROUTE
# -------------------------
@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    os.makedirs("uploads", exist_ok=True)
    image_path = f"uploads/{file.filename}"

    with open(image_path, "wb") as f:
        f.write(await file.read())

    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    idx = int(np.argmax(preds))
    confidence = float(preds[0][idx]) * 100

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "predicted_class": "unknown",
            "confidence": round(confidence, 2),
            "message": "Uploaded image does not match known civic issue categories"
        }

    record = Prediction(
        filename=file.filename,
        predicted_class=CLASS_NAMES[idx],
        confidence=confidence
    )
    db.add(record)
    db.commit()

    return {
        "predicted_class": CLASS_NAMES[idx],
        "confidence": round(confidence, 2)
    }
