import os
import random
import shutil

# Define the directory containing the images
image_dir = 'C:/Users/xcomb/OneDrive/Desktop/ML project/appleuni/Unknown'
removed_dir = 'C:/Users/xcomb/OneDrive/Desktop/ML project/appleuni/Unknown_Removed'

# Ensure the removed_images directory exists
if not os.path.exists(removed_dir):
    os.makedirs(removed_dir)

# List all image files
all_images = [f for f in os.listdir(image_dir) if f.startswith('unknown_') and f.endswith('.jpg')]

# Split the images into two categories
category1 = [f for f in all_images if int(f.split('_')[1].split('.')[0]) <= 350]
category2 = [f for f in all_images if int(f.split('_')[1].split('.')[0]) > 350]

# Randomly select 100 images from each category
random.shuffle(category1)
random.shuffle(category2)
removed_category1 = category1[:100]
removed_category2 = category2[:100]

# Move the selected images to the removed_images directory
for img in removed_category1 + removed_category2:
    shutil.move(os.path.join(image_dir, img), os.path.join(removed_dir, img))

print(f"Moved {len(removed_category1)} images from category 1 and {len(removed_category2)} images from category 2 to {removed_dir}")