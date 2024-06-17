from PIL import Image
import os
from tqdm import tqdm
# '/data/ly/SID/Sony_rgb/ratio300/train/noisy/': '/data/ly/SID/Sony_rgb/ratio300/train/noisy_cropped/',
# Define the input and output directories dictionary
input_output_dirs = {
    # '/data/ly/SID/Sony_rgb/ratio300/train/clean/': '/data/ly/SID/Sony_rgb/ratio300/train/clean_cropped/',
    # '/data/ly/SID/Sony_rgb/ratio300/val/noisy/': '/data/ly/SID/Sony_rgb/ratio300/val/noisy_cropped/',
    # '/data/ly/SID/Sony_rgb/ratio300/val/clean/': '/data/ly/SID/Sony_rgb/ratio300/val/clean_cropped/',
    # '/data/ly/SID/Sony_rgb/ratio300/test/clean/': '/data/ly/SID/Sony_rgb/ratio300/test/clean_cropped/',
    # '/data/ly/SID/Sony_rgb/ratio300/test/noisy/': '/data/ly/SID/Sony_rgb/ratio300/test/noisy_cropped/',

    # '/data/ly/SID/Sony_rgb/ratio250/train/noisy/': '/data/ly/SID/Sony_rgb/ratio250/train/noisy_cropped/',
    # '/data/ly/SID/Sony_rgb/ratio250/train/clean/': '/data/ly/SID/Sony_rgb/ratio250/train/clean_cropped/',
    # '/data/ly/SID/Sony_rgb/ratio250/val/noisy/': '/data/ly/SID/Sony_rgb/ratio250/val/noisy_cropped/',
    # '/data/ly/SID/Sony_rgb/ratio250/test/clean/': '/data/ly/SID/Sony_rgb/ratio250/test/clean_cropped/',
    # '/data/ly/SID/Sony_rgb/ratio250/test/noisy/': '/data/ly/SID/Sony_rgb/ratio250/test/noisy_cropped/',
    # '/data/ly/SID/Sony_rgb/ratio250/val/clean/': '/data/ly/SID/Sony_rgb/ratio250/val/clean_cropped/',
    #
    # '/data/ly/SID/Sony_rgb/ratio100/train/noisy/': '/data/ly/SID/Sony_rgb/ratio100/train/noisy_cropped/',
    # '/data/ly/SID/Sony_rgb/ratio100/train/clean/': '/data/ly/SID/Sony_rgb/ratio100/train/clean_cropped/',
    # '/data/ly/SID/Sony_rgb/ratio100/val/noisy/': '/data/ly/SID/Sony_rgb/ratio100/val/noisy_cropped/',
    # '/data/ly/SID/Sony_rgb/ratio100/test/clean/': '/data/ly/SID/Sony_rgb/ratio100/test/clean_cropped/',
    # '/data/ly/SID/Sony_rgb/ratio100/test/noisy/': '/data/ly/SID/Sony_rgb/ratio100/test/noisy_cropped/',
    # '/data/ly/SID/Sony_rgb/ratio100/val/clean/': '/data/ly/SID/Sony_rgb/ratio100/val/clean_cropped/',
}

# Traverse each input and output directory pair
for input_dir, output_dir in input_output_dirs.items():
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Traverse all image files in the input directory
    for filename in tqdm(os.listdir(input_dir), desc=f"Processing {input_dir}"):
        if filename.endswith('.png'):  # Ensure it's an image file

            # Open the image
            image = Image.open(os.path.join(input_dir, filename))

            # Get the width and height of the image
            width, height = image.size

            # Define the size and position of each part
            parts = [
                (0, 0, width // 3, height // 2),            # Top left
                (width // 3, 0, 2 * width // 3, height // 2),  # Top center
                (2 * width // 3, 0, width, height // 2),     # Top right
                (0, height // 2, width // 3, height),      # Bottom left
                (width // 3, height // 2, 2 * width // 3, height),  # Bottom center
                (2 * width // 3, height // 2, width, height)      # Bottom right
            ]

            # Traverse each part, crop, and save
            for i, part in enumerate(parts):
                cropped_filename = f'{filename}_crop{i+1}.png'
                if not os.path.exists(os.path.join(output_dir, cropped_filename)):  # Check if file exists
                    cropped_image = image.crop(part)
                    cropped_image.save(os.path.join(output_dir, cropped_filename))

print("All images have been cropped.")
