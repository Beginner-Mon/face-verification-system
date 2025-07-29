
from fastapi import  APIRouter, UploadFile, File
from pydantic import BaseModel
from model_evaluation import load_metrics_from_csv
from predict_single_batch import predict_single_batch
from typing import List
import numpy as np
import cv2

router = APIRouter()
classification_model_path = "../model/classification_model.keras"
cosine_model_path = "../model/cosine_metrics_model.keras"
concat_model_path = "../model/concat_siamese_model.keras"
# Pydantic model for metrics response
class MetricsResponse(BaseModel):
    model: str
    accuracy: float
    mae: float
    mse: float
    auc: float

@router.get("/metrics", response_model=List[MetricsResponse])
async def get_model_metrics():
    metrics = load_metrics_from_csv()
    return metrics

@router.post("/predict")
async def predict_images(image1: UploadFile = File(...), image2: UploadFile = File(...), model: str ="concat_siamese_model.keras"):
    img1_data = await image1.read()
    img2_data = await image2.read()
    img1_array = np.frombuffer(img1_data, np.uint8)
    img2_array = np.frombuffer(img2_data, np.uint8)
    img1 = cv2.imdecode(img1_array, cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(img2_array, cv2.IMREAD_COLOR)
    
    concat_model_path = f"../model/{model}"
    predictions = predict_single_batch(img1, img2, concat_model_path)
    
    return predictions
