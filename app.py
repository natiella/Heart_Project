# app.py

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import pandas as pd
import io
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.predictor import HeartAttackPredictor

app = FastAPI(title="Heart Attack Risk Predictor")

# Загружаем модель один раз при старте
predictor = HeartAttackPredictor(
    model_path='models/heart_attack_model.pkl',
    preprocessor_path='models/preprocessor.pkl'
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        predictions = predictor.predict(df)
        result = [
            {"id": int(row.id), "prediction": int(pred)}
            for row, pred in zip(df.itertuples(), predictions)
            ]
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})