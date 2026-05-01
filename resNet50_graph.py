import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

# Load model
MODEL_PATH = r"C:\Users\skj25\Downloads\dl_project\resnet50_train.h5"
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# Class names
class_names = ["pituitary", "notumor", "meningioma", "glioma"]

# True labels manually defined
true_labels = [3, 1, 0, 2]  # glioma, notumor, pituitary, meningioma

# Image paths
test_image_paths = [
    r"C:\Users\skj25\Downloads\dl_project\test1.jpg",
    r"C:\Users\skj25\Downloads\dl_project\test2.jpg",
    r"C:\Users\skj25\Downloads\dl_project\test3.jpg",
    r"C:\Users\skj25\Downloads\dl_project\test4.jpg"
]

img_size = (224, 224)
pred_labels = []
all_probs = []

# Display image predictions
fig, axes = plt.subplots(1, len(test_image_paths), figsize=(20, 5))
for i, img_path in enumerate(test_image_paths):
    if not os.path.exists(img_path):
        print(f"{img_path} not found!")
        continue

    img = load_img(img_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    pred_labels.append(predicted_class)
    all_probs.append(prediction[0])

    axes[i].imshow(img)
    axes[i].set_title(f"Predicted: {class_names[predicted_class]}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# Confusion Matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1, 2, 3])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# ROC Curve
plt.figure(figsize=(6, 5))
for i in range(len(class_names)):
    fpr, tpr, _ = roc_curve([1 if y == i else 0 for y in true_labels],
                            [p[i] for p in all_probs])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Bar Graph of True Class Distribution
plt.figure(figsize=(6, 5))
unique, counts = np.unique(true_labels, return_counts=True)
plt.bar([class_names[i] for i in unique], counts, color=['red', 'blue', 'green', 'purple'])
plt.xlabel("True Classes")
plt.ylabel("Count")
plt.title("True Class Distribution")
plt.show()

# Histogram of Predictions
plt.figure(figsize=(6, 5))
plt.hist(pred_labels, bins=np.arange(len(class_names)+1)-0.5, edgecolor='black', alpha=0.7)
plt.xticks(range(len(class_names)), class_names)
plt.xlabel("Predicted Classes")
plt.ylabel("Frequency")
plt.title("Prediction Histogram")
plt.show()

# Classification Report
print("\n Classification Report:")
print(classification_report(true_labels, pred_labels, target_names=class_names))
