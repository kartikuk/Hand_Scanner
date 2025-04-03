import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define Paths
DATASET_DIR = "dataset"  # Check this folder contains images
AUGMENTED_DIR = "augmented_data"

# Ensure the augmented directory exists
os.makedirs(AUGMENTED_DIR, exist_ok=True)

# Image Augmentation Setup
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Iterate over images in the dataset folder
for filename in os.listdir(DATASET_DIR):
    img_path = os.path.join(DATASET_DIR, filename)

    # Check if it's a valid image
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        print(f"Skipping non-image file: {filename}")
        continue

    print(f"Processing image: {filename}")  # Debugging print

    image = cv2.imread(img_path)
    if image is None:
        print(f"Failed to load image: {filename}")
        continue

    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Generate augmented images
    i = 0
    for batch in datagen.flow(image, batch_size=1, save_to_dir=AUGMENTED_DIR, save_prefix="aug", save_format="jpg"):
        i += 1
        if i >= 5:  # Generate 5 variations per image
            break  

print("✅ Image augmentation completed!")
