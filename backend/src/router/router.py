
from fastapi import  APIRouter, UploadFile, File
from pydantic import BaseModel
from model_evaluation import evaluate_model_classification, evaluate_siamese_model, create_test_generator, create_pairs_from_classification_data, prepare_dataset_numpy
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
    fpr: List[float]
    tpr: List[float]
    roc_auc: float

@router.get("/metrics", response_model=List[MetricsResponse])
async def get_model_metrics():
    test_generator = create_test_generator()
    test_pairs = create_pairs_from_classification_data()
    test_images1, test_images2, test_labels = prepare_dataset_numpy(test_pairs)
    
    
    
    classification_metrics = evaluate_model_classification(classification_model_path, test_generator)
    cosine_metrics = evaluate_siamese_model(cosine_model_path, test_images1, test_images2, test_labels, is_cosine=True)
    concat_metrics = evaluate_siamese_model(concat_model_path, test_images1, test_images2, test_labels, is_cosine=False)
    
    return [classification_metrics, cosine_metrics, concat_metrics]

@router.post("/predict")
async def predict_images(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    img1_data = await image1.read()
    img2_data = await image2.read()
    img1_array = np.frombuffer(img1_data, np.uint8)
    img2_array = np.frombuffer(img2_data, np.uint8)
    img1 = cv2.imdecode(img1_array, cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(img2_array, cv2.IMREAD_COLOR)
    
    predictions = predict_single_batch(img1, img2, cosine_model_path, concat_model_path)
    
    return {
        "cosine_prediction": float(predictions["cosine"]),
        "concatenate_prediction": float(predictions["concatenate"])
    }
