import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np

# Paths
dataset_path = r"C:\Users\skj25\Downloads\dl_project\brain_tumor_dataset"
model_path = r"C:\Users\skj25\Downloads\dl_project\brain_tumor_vgg16.h5"
history_path = r"C:\Users\skj25\Downloads\dl_project\vgg16_history.npy"

train_dir = os.path.join(dataset_path, "Training")
test_dir = os.path.join(dataset_path, "Testing")

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2,
                                   height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                   horizontal_flip=True, fill_mode="nearest")
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=IMG_SIZE,
                                                    batch_size=BATCH_SIZE, class_mode='categorical')
test_generator = test_datagen.flow_from_directory(test_dir, target_size=IMG_SIZE,
                                                  batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

print("🧾 Class Indices:", train_generator.class_indices)

# Load VGG16 base model
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Custom top layers
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=x)
model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(train_generator, epochs=EPOCHS, validation_data=test_generator)
model.save(model_path)
np.save(history_path, history.history)

print("✅ VGG16 Model trained and saved.")
