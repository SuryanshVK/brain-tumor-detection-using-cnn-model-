import os
import warnings
import tensorflow as tf
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Suppress TensorFlow and system warnings
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
model = tf.keras.models.load_model("brain_tumor_model.h5")

# Test dataset path
test_data_dir = r"C:\Users\skj25\Downloads\dl_project\brain_tumor_dataset\Testing"

# Image size & batch size
img_size = (150, 150)
batch_size = 32

# Test data generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Get predictions
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

# Confusion Matrix 
conf_matrix = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues",
            xticklabels=class_labels.values(),
            yticklabels=class_labels.values())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Class-wise Accuracy Bar Chart 
class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

plt.figure(figsize=(8, 6))
sns.barplot(x=list(class_labels.values()), y=class_accuracy, palette="viridis")
plt.xlabel("Tumor Type")
plt.ylabel("Accuracy")
plt.title("Class-wise Accuracy")
plt.ylim(0, 1)  # Ensure values are between 0 and 1
plt.show()

# ROC Curve for Each Class 
y_true_binarized = label_binarize(y_true, classes=[0, 1, 2, 3])

plt.figure(figsize=(8, 6))
for i in range(len(class_labels)):
    fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_pred[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{class_labels[i]} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Histogram of Model Predictions 
plt.figure(figsize=(8, 6))
sns.histplot(y_pred_classes, bins=np.arange(len(class_labels) + 1) - 0.5, kde=False)
plt.xticks(range(len(class_labels)), class_labels.values())
plt.xlabel("Predicted Class")
plt.ylabel("Frequency")
plt.title("Histogram of Predictions")
plt.show()

# Classification Report 
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred_classes, target_names=class_labels.values()))

 