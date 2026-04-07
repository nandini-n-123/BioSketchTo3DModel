import os
import shutil
import random

def split_dataset(base_dir="dataset", split_ratio=0.85):
    """
    Automatically splits the 'train' folder into 'train' and 'test' folders.
    split_ratio=0.85 means 85% of images stay in train, 15% go to test.
    """
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')

    # Classes are 'brain' and 'hibiscus'
    classes = os.listdir(train_dir)

    for class_name in classes:
        class_train_path = os.path.join(train_dir, class_name)
        class_test_path = os.path.join(test_dir, class_name)

        # Skip if it's not a folder
        if not os.path.isdir(class_train_path):
            continue

        # Create the test folder for this class if it doesn't exist
        os.makedirs(class_test_path, exist_ok=True)

        # Get all images in the folder
        images = os.listdir(class_train_path)
        
        # Calculate how many images should go to the test folder
        num_test_images = int(len(images) * (1 - split_ratio))

        # Randomly select images to move
        test_images = random.sample(images, num_test_images)

        print(f"Moving {num_test_images} images from {class_name} to test folder...")

        # Move the selected files
        for img in test_images:
            src = os.path.join(class_train_path, img)
            dst = os.path.join(class_test_path, img)
            shutil.move(src, dst)

    print("\n✅ Dataset successfully split! You can now run train_ai.py")

if __name__ == "__main__":
    split_dataset()