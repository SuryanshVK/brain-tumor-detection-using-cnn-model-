import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Dataset path
train_data_dir = r"C:\Users\skj25\Downloads\dl_project\brain_tumor_dataset\Training"
val_data_dir = r"C:\Users\skj25\Downloads\dl_project\brain_tumor_dataset\Testing"
img_size = (224, 224)
batch_size = 32
epochs = 25

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Load Pre-trained ResNet50 Model
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Fine-tuning: Unfreeze some layers
for layer in base_model.layers[-20:]:  # Unfreezing last 20 layers
    layer.trainable = True

# Custom layers on top
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(len(train_generator.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=x)

# Compile model
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train model
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=[lr_scheduler, early_stopping]
)

# Save model
model.save(r"C:\Users\skj25\Downloads\dl_project\resNet\resnet50_train.h5")

# Save training history
np.save(r"C:\Users\skj25\Downloads\dl_project\history.npy", history.history)

print("Model training completed and saved!")
