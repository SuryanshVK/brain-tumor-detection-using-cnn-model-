import os
import warnings
import tensorflow as tf
import logging
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model

#. Suppress TensorFlow and system warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
tf.get_logger().setLevel(logging.ERROR)
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
absl.logging.set_stderrthreshold(absl.logging.FATAL)

# Class labels
class_labels = {0: "No Tumor", 1: "Pituitary Tumor", 2: "Meningioma Tumor", 3: "Glioma Tumor"}

# Load trained model
def load_trained_model(model_path):
    return load_model(model_path)

# Prediction function
def predict_image(model, image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f" Error: Unable to load image at {image_path}")
        return None, None
    img = cv2.resize(img, (150, 150))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    return img, class_labels.get(class_index, "Unknown")

# Define 4 test image paths manually
test_image_paths = [
    r"C:\Users\skj25\Downloads\dl_project\test1.jpg",
    r"C:\Users\skj25\Downloads\dl_project\test2.jpg",
    r"C:\Users\skj25\Downloads\dl_project\test3.jpg",
    r"C:\Users\skj25\Downloads\dl_project\test4.jpg"
]

# Load model
model = load_trained_model("brain_tumor_model.h5")

# Process and display images
if model:
    plt.figure(figsize=(12, 8))
    for i, img_path in enumerate(test_image_paths):
        img, label = predict_image(model, img_path)
        if img is not None:
            plt.subplot(2, 2, i+1)
            plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
            plt.title(f"Prediction: {label}")
            plt.axis("off")
    plt.tight_layout()
    plt.show()
