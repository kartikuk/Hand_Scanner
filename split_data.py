import os
import shutil
import random
import pandas as pd

# Define paths
AUGMENTED_DIR = "augmented_data"
LABELS_CSV = "labels.csv"  # Ensure this file exists
TRAIN_DIR = "train_data/train"
VAL_DIR = "train_data/val"

CLEAN_DIR_TRAIN = os.path.join(TRAIN_DIR, "clean")
DIRTY_DIR_TRAIN = os.path.join(TRAIN_DIR, "dirty")
CLEAN_DIR_VAL = os.path.join(VAL_DIR, "clean")
DIRTY_DIR_VAL = os.path.join(VAL_DIR, "dirty")

# Create output directories if they don't exist
for folder in [CLEAN_DIR_TRAIN, DIRTY_DIR_TRAIN, CLEAN_DIR_VAL, DIRTY_DIR_VAL]:
    os.makedirs(folder, exist_ok=True)

print("🚀 Script started...")

# Load labels from CSV
try:
    labels_df = pd.read_csv(LABELS_CSV)
    
    # Normalize filenames and labels (strip spaces, lowercase)
    labels_df["filename"] = labels_df["filename"].str.strip().str.lower()
    labels_df["label"] = labels_df["label"].str.strip().str.lower()
    
    # Convert to dictionary for quick lookup
    labels_dict = dict(zip(labels_df["filename"], labels_df["label"]))
    
    print("✅ Labels file loaded successfully!")
except Exception as e:
    print(f"⚠️ Error loading labels file: {e}")
    exit()

# Get list of images in augmented_data/
all_images = [img.strip().lower() for img in os.listdir(AUGMENTED_DIR) if img.endswith(".jpg")]

# Debugging: Print sample filenames to check for mismatches
print(f"📂 Found {len(all_images)} images in '{AUGMENTED_DIR}'.")
print("🔍 Sample images from folder:", all_images[:5])
print("🔍 Sample filenames from labels.csv:", list(labels_dict.keys())[:5])

if not all_images:
    print("⚠️ No images found in 'augmented_data/'. Make sure preprocess.py ran successfully.")
    exit()

# Separate clean & dirty images based on CSV labels
clean_images = []
dirty_images = []

for img in all_images:
    label = labels_dict.get(img)
    if label == "clean":
        clean_images.append(img)
    elif label == "dirty":
        dirty_images.append(img)
    else:
        print(f"⚠️ Warning: No label found for '{img}' in labels.csv!")

# Debugging: Print counts after classification
print(f"✅ Classified images -> Clean: {len(clean_images)}, Dirty: {len(dirty_images)}")

# Shuffle data
random.shuffle(clean_images)
random.shuffle(dirty_images)

# Split (80% train, 20% validation)
split_clean = int(len(clean_images) * 0.8)
split_dirty = int(len(dirty_images) * 0.8)

train_clean, val_clean = clean_images[:split_clean], clean_images[split_clean:]
train_dirty, val_dirty = dirty_images[:split_dirty], dirty_images[split_dirty:]

print(f"🔀 Train: {len(train_clean)} clean, {len(train_dirty)} dirty")
print(f"🔀 Validation: {len(val_clean)} clean, {len(val_dirty)} dirty")

# Move images to respective folders
for img in train_clean:
    shutil.move(os.path.join(AUGMENTED_DIR, img), os.path.join(CLEAN_DIR_TRAIN, img))

for img in val_clean:
    shutil.move(os.path.join(AUGMENTED_DIR, img), os.path.join(CLEAN_DIR_VAL, img))

for img in train_dirty:
    shutil.move(os.path.join(AUGMENTED_DIR, img), os.path.join(DIRTY_DIR_TRAIN, img))

for img in val_dirty:
    shutil.move(os.path.join(AUGMENTED_DIR, img), os.path.join(DIRTY_DIR_VAL, img))

print(f"✅ Data Split Completed! 🎉")

if __name__ == "__main__":
    print("✅ split_data.py execution finished successfully!")
