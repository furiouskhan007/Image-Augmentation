import imgaug.augmenters as iaa
import os
from PIL import Image
import numpy as np

# Define the augmentation sequence
augmentation = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Affine(rotate=(-10, 10)),
    iaa.GaussianBlur(sigma=(0, 1.0)),
    iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),
])

# Path to the directory containing the images
image_directory = "inputs/"

# Output directory to save augmented images
output_directory = "Results/"

# List all image files in the directory
image_files = os.listdir(image_directory)

# Augment each image and save three augmented images for each input
for filename in image_files:
    # Open the image
    image_path = os.path.join(image_directory, filename)
    image = Image.open(image_path)

    # Generate and save three augmented images
    for i in range(4):
        # Apply augmentation
        augmented_image = augmentation(image=np.array(image))

        # Convert back to PIL Image object
        augmented_image_pil = Image.fromarray(augmented_image)

        # Save augmented image
        output_path = os.path.join(output_directory, f"augmented_{filename[:-4]}_{i+1}.jpg")
        augmented_image_pil.save(output_path)

        print(f"Augmented image saved: {output_path}")
