import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
MODEL_DIR = "models"
model = load_model(os.path.join(MODEL_DIR, "hand_scan_model.h5"))

# Function to preprocess the image for prediction
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))  # Resize image to match the input shape of the model
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img.astype('float32') / 255  # Normalize the image
    return img

# Initialize webcam (0 for the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Preprocess the captured image
    preprocessed_frame = preprocess_image(frame)
    
    # Make prediction
    prediction = model.predict(preprocessed_frame)
    
    # Determine the class (clean or dirty)
    label = "Clean" if prediction[0] < 0.5 else "Dirty"
    color = (0, 255, 0) if label == "Clean" else (0, 0, 255)
    
    # Display the result on the frame
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.rectangle(frame, (10, 10), (240, 60), color, 2)
    
    # Show the frame with the prediction
    cv2.imshow('Hand Scan', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
