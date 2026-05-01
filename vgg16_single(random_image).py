import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the trained model
model_path = r"C:\Users\skj25\Downloads\dl_project\vgg\brain_tumor_vgg16.h5"
model = tf.keras.models.load_model(model_path)
print("✅ VGG16 Model loaded successfully!")

# Define class names (must match your training order!)
class_names = ["pituitary", "notumor", "glioma", "meningioma"]
IMG_SIZE = (224, 224)

# Path to the image you want to test (downloaded from internet)
image_path = r"C:\Users\skj25\Downloads\dl_project\vgg\8-me.jpg"

# Check if the image exists
if not os.path.exists(image_path):
    print(f"❌ Image not found: {image_path}")
else:
    # Load and preprocess image
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0  # Scale to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Shape (1, 224, 224, 3)

    # Predict using the model
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names[predicted_class_index]
    confidence = np.max(prediction)

    # Display the image and prediction
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"🧠 Predicted: {predicted_class}\n📊 Confidence: {confidence:.2f}", fontsize=14)
    plt.show()

    # Debugging: Print all predictions and confidence scores
    print("Prediction Probabilities:", prediction)
    print("Predicted Class Index:", predicted_class_index)
    print("Confidence for all classes:", prediction[0])

    # Display class-wise prediction probabilities
    plt.bar(class_names, prediction[0])
    plt.title("Class Prediction Probabilities")
    plt.show()
