
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

def predict_single_batch(img1, img2, cosine_model_path, concat_model_path):
    img1_processed = load_and_preprocess_image(img1)
    img2_processed = load_and_preprocess_image(img2)
    
    img1_processed = np.expand_dims(img1_processed, axis=0)
    img2_processed = np.expand_dims(img2_processed, axis=0)
    
    custom_objects = {
        'cosine_similarity': lambda x: K.sum(K.l2_normalize(x[0], axis=1) * K.l2_normalize(x[1], axis=1), axis=1, keepdims=True),
        'l2_normalize': lambda x: K.l2_normalize(x, axis=1, epsilon=1e-10)
    }
    
    cosine_model = load_model(cosine_model_path, custom_objects=custom_objects, safe_mode=False)
    concat_model = load_model(concat_model_path, safe_mode=False)
    
    with tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"]).scope():
        cosine_pred = cosine_model.predict([img1_processed, img2_processed], batch_size=1)[0][0]
        concat_pred = concat_model.predict([img1_processed, img2_processed], batch_size=1)[0][0]
    
    if 'siamese' in concat_model_path.lower():
        concat_pred = 1 - concat_pred / max(concat_pred, 1e-10)
    
    return {
        "cosine": float(cosine_pred),
        "concatenate": float(concat_pred)
    }
