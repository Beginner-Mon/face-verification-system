
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
import numpy as np
import cv2

def load_and_preprocess_image(image, img_size=(112, 112)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, img_size)
    image = image.astype(np.float32)
    image = np.clip(image, 0, 255)
    return image / 255.0

def predict_single_batch(img1, img2, concat_model_path, threshold=0.6):
    img1_processed = load_and_preprocess_image(img1)
    img2_processed = load_and_preprocess_image(img2)
    
    img1_processed = np.expand_dims(img1_processed, axis=0)
    img2_processed = np.expand_dims(img2_processed, axis=0)
    
    concat_model = load_model(concat_model_path, safe_mode=False)
    
    with tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"]).scope():
        concat_pred = concat_model.predict([img1_processed, img2_processed], batch_size=1)[0][0]
    concat_pred = concat_pred.flatten()

    pred_value = float(concat_pred)
    is_same = pred_value >= threshold
    
    return {
        "model_name": concat_model_path.split('/')[-1],  # Extract filename from path
        "pred_value": pred_value,
        "label": "same" if is_same else "not same",
        "binary": 1 if is_same else 0
    }

def main():
   # Example usage
   concat_model_path = "../model/concat_siamese_model.keras"  # Update with your model path
   
   # Load two images (replace with your image paths)
   img1 = cv2.imread("../users/tri.jpg")
   img2 = cv2.imread("../users/ho.jpg")
   
   if img1 is None or img2 is None:
       print("Error: Could not load one or both images")
       return
   
   # Preprocess images
   img1_processed = load_and_preprocess_image(img1)
   img2_processed = load_and_preprocess_image(img2)
   
   img1_processed = np.expand_dims(img1_processed, axis=0)
   img2_processed = np.expand_dims(img2_processed, axis=0)
   
   # Load concat model only
   concat_model = load_model(concat_model_path, safe_mode=False)
   
   # Make prediction
   with tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"]).scope():
        concat_pred = concat_model.predict([img1_processed, img2_processed], batch_size=1)[0][0]
   predictions = predictions.flatten()
   # Apply transformation if siamese model
   
   
   print(f"Concatenate model prediction: {float(concat_pred)}")
   
   return float(concat_pred)

if __name__ == "__main__":
   main()