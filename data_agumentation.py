import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil

# ==== CONFIG ====
# Source directories (set these before running)
SOURCE_IMG_DIR = "data/images"       # path to raw images
SOURCE_LABEL_DIR = "data/labels"     # path to raw annotations

# Output directories
OUTPUT_IMG_DIR = "augmented/images"
OUTPUT_LABEL_DIR = "augmented/labels"

# ==== NIGHT AUGMENTATION FUNCTION ====

def simulate_night(image):
    # Step 1: Slight darkening
    dark = cv2.convertScaleAbs(image, alpha=0.8, beta=0)  # Try alpha=0.7 instead of 0.4

    # Step 2: Mild gamma correction
    gamma = 1.2  # Lower gamma for less extreme correction
    look_up_table = np.array([((i / 255.0) ** gamma) * 255
                              for i in range(256)]).astype("uint8")
    gamma_corrected = cv2.LUT(dark, look_up_table)

    # Step 3: Optional blur to simulate sensor softness
    blurred = cv2.GaussianBlur(gamma_corrected, (3, 3), 0.5)

    return blurred



# ==== MAKE SURE OUTPUT DIRS EXIST ====
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

# ==== PROCESS ALL IMAGES ====
image_files = [f for f in os.listdir(SOURCE_IMG_DIR) if f.endswith(('.jpg', '.png'))]

for img_name in tqdm(image_files, desc="Augmenting images"):
    base_name = os.path.splitext(img_name)[0]
    img_path = os.path.join(SOURCE_IMG_DIR, img_name)
    label_path = os.path.join(SOURCE_LABEL_DIR, base_name + '.txt')

    # Read and augment image
    image = cv2.imread(img_path)
    if image is None:
        print(f" Warning: Skipping {img_name} (could not load)")
        continue

    night_img = simulate_night(image)

    # Save augmented image
    new_img_name = base_name + '_night.jpg'
    cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, new_img_name), night_img)

    # Copy label
    new_label_name = base_name + '_night.txt'
    if os.path.exists(label_path):
        shutil.copyfile(label_path, os.path.join(OUTPUT_LABEL_DIR, new_label_name))
    else:
        print(f" Warning: No label found for {img_name}, skipping label.")

print(" Dataset augmentation complete!")
