import os

# Define the directory containing the images
image_dir = 'C:/Users/xcomb/OneDrive/Desktop/ML project/appleuni/Unknown'

# List all image files in the directory
all_images = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

# Sort images to ensure consistent renaming
all_images.sort()

# Rename images
for i, filename in enumerate(all_images):
    new_name = f"unknown_{i + 1}.jpg"
    old_path = os.path.join(image_dir, filename)
    new_path = os.path.join(image_dir, new_name)
    os.rename(old_path, new_path)

print(f"Renamed {len(all_images)} images to unknown_1.jpg to unknown_{len(all_images)}.jpg")