import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import cv2
import os
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import preprocess_input

benign_dir = 'E:/Breast_Cancer_DL/Final Dataset/train/benign'
malignant_dir = 'E:/Breast_Cancer_DL/Final Dataset/train/malignant'

#Optimized Image Loading & Preprocessing
def load_images_from_directory(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            img = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (224, 224))
            img = np.stack([img] * 3, axis=-1)  
            img = preprocess_input(img.astype(np.float32))  
            images.append(img)
    return np.array(images)

# Load images
benign_images = load_images_from_directory(benign_dir)
malignant_images = load_images_from_directory(malignant_dir)

#Feature Extraction Using VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

benign_features = base_model.predict(benign_images, batch_size=32)
malignant_features = base_model.predict(malignant_images, batch_size=32)

# Create labels (0 = benign, 1 = malignant)
benign_labels = np.zeros(len(benign_features), dtype=int)
malignant_labels = np.ones(len(malignant_features), dtype=int)

# Combine features and labels
X = np.concatenate((benign_features, malignant_features), axis=0)
y = np.concatenate((benign_labels, malignant_labels), axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Flatten feature vectors for the Fully Connected model
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

#Improved Neural Network Model
model = Sequential([
    Dense(1024, activation='relu', input_shape=(X_train_flat.shape[1],)),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_flat, y_train, epochs=10, batch_size=32, validation_data=(X_test_flat, y_test))

# Evaluation
y_pred = (model.predict(X_test_flat) > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print(report)

#  Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Save the model
model_filename = 'E:/Breast_Cancer_DL/Website/final trial/Run/BreastCancer_DL_Model.h5'
model.save(model_filename)
