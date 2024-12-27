import numpy as np
from PIL import Image
import os
import json
import random
import shutil

WIDTH = 4
HEIGHT = 4

TRAIN_SIZE = 20  # Total size of training dataset
TEST_SIZE = int(TRAIN_SIZE*0.10)  # Size of test dataset


def generate_image():
    array = np.random.randint(0, 2, size=(WIDTH, HEIGHT)) * 255
    black_pixels = np.sum(array == 0)
    label = 1 if black_pixels >= int((WIDTH*HEIGHT)/2) else 0
    return array, label


if __name__ == "__main__":
    # Create directories
    train_dir = f'train{WIDTH}x{HEIGHT}'
    test_dir = f'test{WIDTH}x{HEIGHT}'
    true_test_dir = f'true_test{WIDTH}x{HEIGHT}'  # New true_test directory
    # Remove old directories if they exist
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)

    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    if os.path.exists(true_test_dir):
        shutil.rmtree(true_test_dir)

    os.makedirs(train_dir)
    os.makedirs(test_dir)
    os.makedirs(true_test_dir)

    # Generate all images first
    all_images = []
    for i in range(TRAIN_SIZE):
        array, label = generate_image()
        filename = f'image_{i}.png'
        all_images.append((array, filename, label))

    # Randomly select test images
    test_indices = random.sample(range(TRAIN_SIZE), TEST_SIZE)

    # Dictionaries for labels
    train_labels = {}
    test_labels = {}
    true_test_labels = {}

    # Process all images
    for i, (array, filename, label) in enumerate(all_images):
        img = Image.fromarray(array.astype('uint8'), 'L')

        if i in test_indices:
            # Save as test image
            img.save(os.path.join(test_dir, filename))
            img.save(os.path.join(train_dir, filename))
            test_labels[filename] = label
            train_labels[filename] = label
        else:
            # Save as train image
            img.save(os.path.join(train_dir, filename))
            train_labels[filename] = label

    for i in range(TRAIN_SIZE):  # Generating TRAIN_SIZE images for true_test
        array, label = generate_image()
        filename = f'true_test_image_{i}.png'
        img = Image.fromarray(array.astype('uint8'), 'L')
        img.save(os.path.join(true_test_dir, filename))
        true_test_labels[filename] = label
        
    # Save labels to JSON files
    with open(os.path.join(train_dir, 'labels.json'), 'w') as f:
        json.dump(train_labels, f)

    with open(os.path.join(test_dir, 'labels.json'), 'w') as f:
        json.dump(test_labels, f)

    with open(os.path.join(true_test_dir, 'labels.json'), 'w') as f:
        json.dump(true_test_labels, f)