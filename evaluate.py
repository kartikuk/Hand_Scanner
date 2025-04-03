import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load Trained Model
MODEL_PATH = "models/hand_scan_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define image size
IMG_SIZE = 224

# Function to Predict Image
def predict_image(image_path):
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Convert to batch format

    prediction = model.predict(img_array)
    return "Clean" if prediction[0][0] < 0.5 else "Dirty"

# Directory where validation images are stored
VAL_DIR = "train_data/val"

# Test all images from validation dataset
for category in ["clean", "dirty"]:
    category_path = os.path.join(VAL_DIR, category)
    
    if os.path.exists(category_path):
        for image_name in os.listdir(category_path):
            image_path = os.path.join(category_path, image_name)
            if image_path.endswith(".jpg") or image_path.endswith(".png"):
                result = predict_image(image_path)
                print(f"🖼 {image_name} -> Predicted: {result} (Actual: {category})")
