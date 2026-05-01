from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Test dataset ka path
test_data_dir = r"C:\Users\skj25\Downloads\dl_project\brain_tumor_dataset\Testing"
  
# Image size aur batch size
img_size = (150, 150)  # Same size as the model training 
batch_size = 32

# Test data generator
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  #for evaluation
)
import tensorflow as tf

# Saved model loading
model = tf.keras.models.load_model("brain_tumor_model.h5")  
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")
