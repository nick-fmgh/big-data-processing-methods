import numpy as np
from PIL import Image
import os

WIDTH=3
HEIGHT=3

TEST_SIZE=20 # Размер тестовой выборки
TRAIN_SIZE=10 # Размер обучающей выборки

if __name__ == "__main__":
    # Создаем папки для обучающей и тестовой выборок
    train_dir = f'train{WIDTH}x{HEIGHT}'
    test_dir = f'test{WIDTH}x{HEIGHT}'

    # Создаем директории, если они не существуют
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Задаем количество изображений для генерации
    num_train_images = TRAIN_SIZE
    num_test_images = TEST_SIZE

    # Генерируем обучающую выборку
    for i in range(num_train_images):
        # Создаем случайное черно-белое изображение 3x3
        array = np.random.randint(0, 2, size=(WIDTH, HEIGHT)) * 255
        img = Image.fromarray(array.astype('uint8'), 'L')
        img.save(os.path.join(train_dir, f'train_image_{i}.png'))

    # Генерируем тестовую выборку
    for i in range(num_test_images):
        array = np.random.randint(0, 2, size=(WIDTH, HEIGHT)) * 255
        img = Image.fromarray(array.astype('uint8'), 'L')
        img.save(os.path.join(test_dir, f'test_image_{i}.png'))

