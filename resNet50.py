import tensorflow as tf #Import required libraries
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Dataset Path (Define dataset location)
dataset_path = r"C:\Users\skj25\Downloads\dl_project\brain_tumor_dataset"
train_dir = os.path.join(dataset_path, "Training")
test_dir = os.path.join(dataset_path, "Testing")

# Image Preprocessing
IMG_SIZE = (224, 224)  # ResNet expects 224x224 images
BATCH_SIZE = 32

# Normalize image data
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

#Load training & testing images
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')

# Load Pretrained ResNet50 Model (Without Top)
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Freeze Pretrained Layers
for layer in base_model.layers:
    layer.trainable = False

# Custom Fully Connected Layers
x = Flatten()(base_model.output) # Flatten CNN feature maps
x = Dense(512, activation='relu')(x) #Adds a fully connected layer with 512 neurons.
x = Dropout(0.5)(x)  # Prevent Overfitting
x = Dense(train_generator.num_classes, activation='softmax')(x) #Outputs class probabilities.

# Create Model
model = Model(inputs=base_model.input, outputs=x)

# Compile Model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train Model
model.fit(train_generator, epochs=10, validation_data=test_generator)

# Save Model
model.save("brain_tumor_resnet50.h5")
print("ResNet50 Model trained and saved successfully!")
