import os
import numpy as np
from scipy.io import loadmat
from PIL import Image

def extract_svhn_images(mat_file_path, output_dir):
    """Extract images and labels from SVHN .mat file and save as .png."""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the .mat file
    print(f"Loading {mat_file_path}...")
    data = loadmat(mat_file_path)

    # Extract images and labels
    images = data['X']  # Shape: (32, 32, 3, num_images)
    labels = data['y'].flatten()  # Shape: (num_images,)

    # Transpose images to (num_images, 32, 32, 3)
    images = np.transpose(images, (3, 0, 1, 2))

    # Save images and create label file
    label_file_path = os.path.join(output_dir, "labels.txt")
    with open(label_file_path, 'w') as label_file:
        for i in range(images.shape[0]):
            # Convert image to PIL format
            img = images[i].astype(np.uint8)  # Shape: (32, 32, 3)
            img_pil = Image.fromarray(img)

            # Save image as .png
            img_path = os.path.join(output_dir, f"{i+1}.png")
            img_pil.save(img_path)

            # Adjust label (SVHN uses 10 for 0)
            label = labels[i]
            if label == 10:
                label = 0

            # Write label to file
            label_file.write(f"{i+1}.png {label}\n")

    print(f"Extracted {images.shape[0]} images to {output_dir}")

# Paths to .mat files
train_mat_path = "Dataset/train_32X32.mat"
test_mat_path = "Dataset/test_32X32.mat"

# Output directories
train_output_dir = "Dataset/train"
test_output_dir = "Dataset/test"

# Extract images
extract_svhn_images(train_mat_path, train_output_dir)
extract_svhn_images(test_mat_path, test_output_dir)