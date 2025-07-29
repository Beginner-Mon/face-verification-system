
import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CSV_PATH = "../results/combine_metrics.csv"

def load_metrics_from_csv():
    logger.info(f"Loading metrics from {CSV_PATH}")
    if not os.path.exists(CSV_PATH):
        logger.error(f"CSV file {CSV_PATH} not found")
        raise FileNotFoundError(f"CSV file {CSV_PATH} not found. Please ensure it exists.")
    
    df = pd.read_csv(CSV_PATH)
    metrics_list = []
    for _, row in df.iterrows():
        model_name = row['Model']
        # Map CSV model names to API response labels
        label = {
            'classification_model': 'Classification Model',
            'concatenate_model': 'Concatenate Model',
            'cosine_metrics_model': 'Cosine Similarity Model'
        }.get(model_name, model_name)
        metrics_list.append({
            'model': label,
            'accuracy': float(row['Accuracy']),
            'mae': float(row['MAE']),
            'mse': float(row['MSE']),
            'auc': float(row['AUC']),
            'label': label  # For compatibility with previous response format
        })
    logger.info(f"Loaded metrics for {len(metrics_list)} models")
    return metrics_list