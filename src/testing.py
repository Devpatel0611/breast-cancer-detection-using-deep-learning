import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

# Load trained model
model = load_model('E:/Breast_Cancer_DL/Website/final trial/Run/BreastCancer_DL_Model.h5')

# Load VGG16 for feature extraction
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Function to preprocess image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  
    img = cv2.resize(img, (224, 224))  
    img = np.stack([img] * 3, axis=-1)  
    img = img.astype(np.float32) / 255.0  
    
    # Convert to array and add batch dimension
    img = np.expand_dims(img, axis=0)
    
    # Apply VGG16 preprocessing
    img = preprocess_input(img * 255)  # Convert back to [0, 255] for VGG16
    
    return img

# Load and preprocess new image
new_image_path = 'E:/Breast_Cancer_DL/Final Dataset/train/malignant/malignant (18)-rotated32.png'
new_image = preprocess_image(new_image_path)

# Extract features using VGG16
features = base_model.predict(new_image)

# Flatten feature shape
features_flat = features.reshape(1, -1)

# Predict using trained model
prediction = model.predict(features_flat)

# Debugging print statements
print(f"Raw Model Prediction: {prediction[0][0]}")  

# Threshold for classification
predicted_class = "Malignant" if prediction[0][0] >= 0.5 else "Benign"

print(f"Predicted class: {predicted_class}")
