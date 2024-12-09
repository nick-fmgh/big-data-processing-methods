import numpy as np
from PIL import Image
import os
import json
import random
import shutil

WIDTH = 4
HEIGHT = 4
COLOR_DEPTH = 2
SIZE = WIDTH * HEIGHT

TRAIN_SIZE = SIZE ** COLOR_DEPTH  # Общий размер тренировочного набора
TEST_SIZE = int(TRAIN_SIZE * 0.10)  # Размер тестового набора


def generate_image_by_criterion(match_criterion=True):
    """
    Генерирует изображение, соответствующее или не соответствующее критерию
    match_criterion: True - изображение должно соответствовать критерию
                    False - изображение не должно соответствовать критерию
    """
    while True:
        array = np.random.randint(0, 2, size=(WIDTH, HEIGHT)) * 255
        white_pixels = np.sum(array == 255)
        matches = white_pixels == SIZE

        if matches == match_criterion:
            return array, 1 if matches else 0


def generate_dataset(size, match_criterion=True):
    """Генерирует набор изображений с заданным критерием"""
    images = []
    for i in range(size):
        array, label = generate_image_by_criterion(match_criterion)
        filename = f'image_{i}.png'
        images.append((array, filename, label))
    return images


if __name__ == "__main__":
    # Создаем директории
    train_dir = f'train{WIDTH}x{HEIGHT}'
    test_dir = f'test{WIDTH}x{HEIGHT}'
    true_test_dir = f'true_test{WIDTH}x{HEIGHT}'

    for dir_path in [train_dir, test_dir, true_test_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)

    # Генерируем тренировочные изображения
    half_size = TRAIN_SIZE // 2
    matching_images = generate_dataset(half_size, True)
    non_matching_images = generate_dataset(half_size, False)
    all_images = matching_images + non_matching_images
    random.shuffle(all_images)

    # Выбираем тестовые изображения
    test_indices = random.sample(range(TRAIN_SIZE), TEST_SIZE)

    # Словари для меток
    train_labels = {}
    test_labels = {}
    true_test_labels = {}

    # Сохраняем изображения
    for i, (array, filename, label) in enumerate(all_images):
        img = Image.fromarray(array.astype('uint8'), 'L')

        if i in test_indices:
            img.save(os.path.join(test_dir, filename))
            img.save(os.path.join(train_dir, filename))
            test_labels[filename] = label
            train_labels[filename] = label
        else:
            img.save(os.path.join(train_dir, filename))
            train_labels[filename] = label

    # Генерируем true_test набор
    half_true_test = TRAIN_SIZE // 2
    true_test_matching = generate_dataset(half_true_test, True)
    true_test_non_matching = generate_dataset(half_true_test, False)

    for array, filename, label in true_test_matching + true_test_non_matching:
        img = Image.fromarray(array.astype('uint8'), 'L')
        img.save(os.path.join(true_test_dir, filename))
        true_test_labels[filename] = label

    # Сохраняем метки в JSON файлы
    for dir_path, labels in [
        (train_dir, train_labels),
        (test_dir, test_labels),
        (true_test_dir, true_test_labels)
    ]:
        with open(os.path.join(dir_path, 'labels.json'), 'w') as f:
            json.dump(labels, f)