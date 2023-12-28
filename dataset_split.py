import os
import shutil
import random

SPLITS=10

def split_dataset(input_folder, output_base_folder, num_splits=SPLITS):
    # Create output base folder if it doesn't exist
    if not os.path.exists(output_base_folder):
        os.makedirs(output_base_folder)

    # List subfolders (animal labels)
    subfolders = [f.path for f in os.scandir(input_folder) if f.is_dir()]

    # Create 1000 smaller datasets
    for i in range(num_splits):
        # Create a subfolder for each split
        split_folder = os.path.join(output_base_folder, f'split_{i}')
        os.makedirs(split_folder)

        # Randomly sample images from each class and copy to the split folder
        for subfolder in subfolders:
            label = os.path.basename(subfolder)
            output_class_folder = os.path.join(split_folder, label)
            os.makedirs(output_class_folder)

            images = [f.path for f in os.scandir(subfolder) if f.is_file() and f.name.endswith(('.jpg', '.jpeg', '.png'))]
            selected_images = random.sample(images, min(260, len(images)))  # Adjust the number of images to copy per class

            for image_path in selected_images:
                shutil.copy(image_path, output_class_folder)

# Example usage
input_folder = r'data/raw-img'
output_base_folder = r'data/split-img'
split_dataset(input_folder, output_base_folder)
