import cv2
import os


def classify_image(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Convert to grayscale (simplified for dirty/clean classification)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use simple thresholding to detect areas of dirt
    _, thresholded = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    # Calculate the percentage of "dirty" pixels (e.g., dark spots might indicate dirt)
    dirty_percentage = cv2.countNonZero(thresholded) / (gray.shape[0] * gray.shape[1]) * 100

    # If more than 20% of the image is dirty, classify as "dirty"
    if dirty_percentage > 5:
        return "dirty"
    else:
        return "clean"

def classify_images_in_folder(folder_path):
    for image_file in os.listdir(folder_path):
        if image_file.endswith(".jpg"):
            image_path = os.path.join(folder_path, image_file)
            label = classify_image(image_path)
            print(f"{image_file}: {label}")

# Specify the directory where your images are stored
images_directory = "./augmented_data"
classify_images_in_folder(images_directory)
