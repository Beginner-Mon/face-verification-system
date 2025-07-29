
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
import os
import random
import cv2

# Constants
BATCH_SIZE = 16
IMG_SIZE = (112, 112)
DATA_DIR = "/kaggle/input/11-785-fall-20-homework-2-part-2"
TEST_DIR = f"{DATA_DIR}/classification_data/test_data"
NUM_PAIRS_PER_PERSON = 1

strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])

def load_and_preprocess_image(image, img_size=IMG_SIZE):
    if isinstance(image, str):
        image = cv2.imread(image)
        if image is None:
            raise ValueError(f"Could not load image: {image}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, img_size)
    image = image.astype(np.float32)
    image = np.clip(image, 0, 255)
    return image / 255.0

def create_pairs_from_classification_data(classification_dir=TEST_DIR, num_pairs_per_person=NUM_PAIRS_PER_PERSON):
    pairs = []
    person_dirs = [d for d in os.listdir(classification_dir) 
                  if os.path.isdir(os.path.join(classification_dir, d))]
    
    for person_dir in person_dirs:
        person_path = os.path.join(classification_dir, person_dir)
        images = [f for f in os.listdir(person_path) 
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if len(images) >= 2:
            for _ in range(num_pairs_per_person):
                img1, img2 = random.sample(images, 2)
                img1_path = os.path.join(person_path, img1)
                img2_path = os.path.join(person_path, img2)
                pairs.append((img1_path, img2_path, 1))
    
    num_negative_pairs = len(pairs)
    for _ in range(num_negative_pairs):
        person1, person2 = random.sample(person_dirs, 2)
        person1_path = os.path.join(classification_dir, person1)
        person2_path = os.path.join(classification_dir, person2)
        images1 = [f for f in os.listdir(person1_path) 
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        images2 = [f for f in os.listdir(person2_path) 
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if images1 and images2:
            img1 = random.choice(images1)
            img2 = random.choice(images2)
            img1_path = os.path.join(person1_path, img1)
            img2_path = os.path.join(person2_path, img2)
            pairs.append((img1_path, img2_path, 0))
    
    random.shuffle(pairs)
    return pairs

def prepare_dataset_numpy(pairs, img_size=IMG_SIZE):
    images1, images2, labels = [], [], []
    for img1_path, img2_path, label in pairs:
        try:
            img1 = load_and_preprocess_image(img1_path, img_size)
            img2 = load_and_preprocess_image(img2_path, img_size)
            images1.append(img1)
            images2.append(img2)
            labels.append(label)
        except Exception as e:
            print(f"Error loading pair {img1_path}, {img2_path}: {e}")
            continue
    return np.array(images1), np.array(images2), np.array(labels)

def create_test_generator():
    test_datagen = ImageDataGenerator()
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode='rgb',
        class_mode='categorical',
        shuffle=False
    )
    return test_generator

def evaluate_model_classification(model_path, test_generator):
    model = load_model(model_path, safe_mode=False)
    print("\nClassification Model Summary:")
    model.summary()
    
    with strategy.scope():
        predictions = model.predict(test_generator, batch_size=BATCH_SIZE)
    true_labels = test_generator.classes
    
    predicted_labels = np.argmax(predictions, axis=1)
    test_accuracy = accuracy_score(true_labels, predicted_labels)
    
    y_true_bin = np.eye(len(test_generator.class_indices))[true_labels]
    y_score = predictions
    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc = auc(fpr, tpr)
    
    print("\nClassification Model Metrics:")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Micro-average AUC: {roc_auc:.4f}")
    
    return {
        'accuracy': float(test_accuracy),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'roc_auc': float(roc_auc),
        'label': 'Classification Model'
    }

def evaluate_siamese_model(model_path, test_images1, test_images2, test_labels, is_cosine=False):
    custom_objects = {
        'cosine_similarity': lambda x: K.sum(K.l2_normalize(x[0], axis=1) * K.l2_normalize(x[1], axis=1), axis=1, keepdims=True),
        'l2_normalize': lambda x: K.l2_normalize(x, axis=1, epsilon=1e-10)
    } if is_cosine else {}
    
    model = load_model(model_path, custom_objects=custom_objects, safe_mode=False)
    print(f"\n{'Cosine' if is_cosine else 'Concatenate'} Model Summary:")
    model.summary()
    
    with strategy.scope():
        predictions = model.predict([test_images1, test_images2], batch_size=BATCH_SIZE)
    predictions = predictions.flatten()
    
    if not is_cosine and 'siamese' in model_path.lower():
        predictions = 1 - predictions / np.max(predictions)
    
    binary_predictions = (predictions > 0.5).astype(np.int32)
    test_accuracy = accuracy_score(test_labels, binary_predictions)
    fpr, tpr, _ = roc_curve(test_labels, predictions)
    roc_auc = auc(fpr, tpr)
    
    print(f"\n{'Cosine' if is_cosine else 'Concatenate'} Model Metrics:")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Micro-average AUC: {roc_auc:.4f}")
    
    return {
        'accuracy': float(test_accuracy),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'roc_auc': float(roc_auc),
        'label': 'Cosine Similarity Model' if is_cosine else 'Concatenate Model'
    }

def plot_roc_curves(metrics_list, output_file='roc_curves_comparison.png'):
    plt.figure(figsize=(10, 8))
    colors = ['darkorange', 'blue', 'green']
    linestyles = ['-', '--', '-.']
    for metrics, color, linestyle in zip(metrics_list, colors, linestyles):
        plt.plot(metrics['fpr'], metrics['tpr'], color=color, linestyle=linestyle, lw=2, 
                 label=f'{metrics["label"]} (AUC = {metrics["roc_auc"]:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves Comparison', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
