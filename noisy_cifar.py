
import os
import random
import cv2
import numpy as np


def add_gaussian_noise(image, mean=0, std=25):
    """Add Gaussian noise to an image."""
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def process_images(root_dir, noise_ratio=0.05):
    """
    Randomly add Gaussian noise to 5% of images in all subfolders.
    Images are overwritten in their original locations.
    """
    all_images = []

    # Collect all image paths
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_images.append(os.path.join(folder_path, file_name))

    # Select 5% of images randomly
    num_to_modify = max(1, int(len(all_images) * noise_ratio))
    selected_images = random.sample(all_images, num_to_modify)

    # Apply noise and overwrite
    for img_path in selected_images:
        img = cv2.imread(img_path)
        if img is None:
            continue
        noisy_img = add_gaussian_noise(img)
        cv2.imwrite(img_path, noisy_img)  # Save back to original location

    print(f"Added Gaussian noise to {len(selected_images)} images.")


if __name__ == "__main__":
    root_directory = "/data/Cifar100_ImageFolder_25%/train/"  # Change this to your image directory
    output_directory = "/data/Cifar100_ImageFolder_25%/train/"  # Change this if you want to save in a different location
    process_images(root_directory, noise_ratio=0.25)

