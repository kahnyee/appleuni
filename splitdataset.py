import os
import shutil

from sklearn.model_selection import train_test_split

def split_data(root, label, test_size=0.1, validate_size=0.111, random_state=1):
    # Define source directory
    src = os.path.join(root, label)
    
    # List all files in the source directory
    data = os.listdir(src)
    
    # Split the data into training and testing sets
    train, test = train_test_split(data, test_size=test_size, random_state=random_state)
    
    # Split the training data into training and validation sets
    train, validate = train_test_split(train, test_size=validate_size, random_state=random_state)
    
    # Define destination directories
    train_dst = root +"/Train/" + label
    test_dst = root +"/Test/" + label
    validate_dst = root +"/Validate/" + label
    
    # Ensure destination directories exist
    os.makedirs(train_dst, exist_ok=True)
    os.makedirs(test_dst, exist_ok=True)
    os.makedirs(validate_dst, exist_ok=True)
    
    # Copy training images
    for image in train:
        shutil.copy(os.path.join(src, image), train_dst)
    
    # Copy test images
    for image in test:
        shutil.copy(os.path.join(src, image), test_dst)
    
    # Copy validation images
    for image in validate:
        shutil.copy(os.path.join(src, image), validate_dst)

# Insert github folder path here
root = 'C:/Users/Jayden/Desktop/ml_project_github/appleuni'

labels = ['Apple', 'Uni Sushi', 'Unknown']

# Split data for each label
for label in labels:
    split_data(root, label)
