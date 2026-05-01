import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Pathsmodel_path = r"C:\Users\skj25\Downloads\dl_project\vgg\brain_tumor_vgg16.h5"
model_path = r"C:\Users\skj25\Downloads\dl_project\vgg\brain_tumor_vgg16.h5"

test_dir = r"C:\Users\skj25\Downloads\dl_project\brain_tumor_dataset\Testing"

# Load model
model = tf.keras.models.load_model(model_path)
print("✅ VGG16 model loaded successfully.")

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load test data
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Evaluate
loss, accuracy = model.evaluate(test_generator)
print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%")
print(f"✅ Test Loss: {loss:.4f}")
