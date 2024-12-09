import os
import sys
import json
import random

import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QPushButton, QLabel, QFileDialog,
                            QSpinBox, QDoubleSpinBox, QTextEdit, QTabWidget, QScrollArea)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
from PIL import Image


DEFAULT_INPUT_SIZE=16
DEFAULT_EPOCHS=100
DEFAULT_POPULATION_SIZE=100
DEFAULT_MUTATION_RATE=0.01

def bin_activate(z):
    return 1 if z >= 0 else 0

class GeneticAlgorithm:
    def __init__(self, input_size, mutation_rate=0.2, population_size=50):
        self.input_size = input_size
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        # Инициализация популяции случайными весами
        self.population = self._init_population()
        self.weights = random.choice(self.population)

    def _init_population(self):
        return [np.random.uniform(-1, 1, self.input_size) for _ in range(self.population_size)]

    def _crossover(self, parent1, parent2):
        # Одноточечное скрещивание
        crossover_point = np.random.randint(0, self.input_size)
        child = np.concatenate([
            parent1[:crossover_point],
            parent2[crossover_point:]
        ])
        return child

    def _mutate(self, weights):
        # Мутация с заданной вероятностью
        for i in range(len(weights)):
            if np.random.random() < self.mutation_rate:
                weights[i] += np.random.normal(0, 0.1)
        return weights

    def _calculate_fitness(self, inputs, target):
        prediction = self.predict(inputs)
        error = abs(target - prediction)
        return error

    def _select(self, population, fitness_scores):
        # Размер турнира как процент от размера популяции (например, 20%)
        tournament_size = max(2, int(self.population_size * 0.5))
        new_population = []

        for _ in range(tournament_size):
            # Выбираем случайных участников турнира
            tournament_indices = np.random.choice(
                len(population),
                min(tournament_size, len(population)),
                replace=False
            )
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]

            # Выбираем победителя турнира (с лучшим значением fitness)
            winner_idx = tournament_indices[np.argmin(tournament_fitness)]
            new_population.append(population[winner_idx])

        return new_population



    def best_error_best_weights(self, inputs, target):
        fitness_scores = [self._calculate_fitness(inputs, target) for self.weights in self.population]
        population_fitness = list(zip(fitness_scores, self.population))
        sorted_population = sorted(population_fitness, key=lambda x: x[0])
        return sorted_population[0]

    def predict(self, inputs):
        return np.dot(inputs, self.weights)

    @staticmethod
    def activate(x):
        return bin_activate(x)

    def train(self, inputs, target):
        fitness_scores = [self._calculate_fitness(inputs, target) for self.weights in self.population]
        population_fitness = list(zip(fitness_scores, self.population))
        sorted_population = sorted(population_fitness, key=lambda x: x[0])
        best_error, best_weights = sorted_population[0]

        new_population = self._select(self.population, fitness_scores)

        # Скрещивание и мутация для создания потомков
        while len(new_population) < self.population_size:
            idx1, idx2 = 0, 0
            while idx1 == idx2:
                idx1 = np.random.randint(0, len(self.population))
                idx2 = np.random.randint(0, len(self.population))
            parent1 = self.population[idx1]
            parent2 = self.population[idx2]
            child = self._crossover(parent1, parent2)
            child = self._mutate(child)
            new_population.append(child)

        self.population = new_population
        self.weights = best_weights

        return best_error, best_weights


class ImageRecognizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.mutation_rate_spin = None
        self.population_size_spin = None
        self.epochs_spin = None
        self.input_size_spin = None
        self.setWindowTitle("Image Recognition with Perceptron")
        self.setGeometry(100, 100, 800, 600)

        self.training_images = []
        self.training_labels = []
        self.neural_network = None

        self.init_ui()

    def init_ui(self):
        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        # Training tab
        training_tab = QWidget()
        training_layout = QVBoxLayout()

        # Settings
        settings_layout = QHBoxLayout()

        self.input_size_spin = QSpinBox()
        self.input_size_spin.setRange(9, 1000)
        self.input_size_spin.setValue(DEFAULT_INPUT_SIZE)
        settings_layout.addWidget(QLabel("Input Size:"))
        settings_layout.addWidget(self.input_size_spin)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(DEFAULT_EPOCHS)
        self.epochs_spin.setSingleStep(100)
        settings_layout.addWidget(QLabel("Epochs:"))
        settings_layout.addWidget(self.epochs_spin)

        self.population_size_spin = QSpinBox()
        self.population_size_spin.setRange(1, 1000)
        self.population_size_spin.setValue(DEFAULT_POPULATION_SIZE)
        self.population_size_spin.setSingleStep(100)
        settings_layout.addWidget(QLabel("Population Size:"))
        settings_layout.addWidget(self.population_size_spin)

        self.mutation_rate_spin = QDoubleSpinBox()
        self.mutation_rate_spin.setDecimals(4)
        self.mutation_rate_spin.setRange(0.00001, 1.0)
        self.mutation_rate_spin.setSingleStep(0.01)
        self.mutation_rate_spin.setValue(DEFAULT_MUTATION_RATE)
        settings_layout.addWidget(QLabel("Mutation Rate:"))
        settings_layout.addWidget(self.mutation_rate_spin)

        training_layout.addLayout(settings_layout)

        # Buttons
        buttons_layout = QHBoxLayout()
        load_btn = QPushButton("Load Training Images")
        load_btn.clicked.connect(self.load_training_images)
        train_btn = QPushButton("Train")
        train_btn.clicked.connect(self.train_neural_network)
        buttons_layout.addWidget(load_btn)
        buttons_layout.addWidget(train_btn)
        training_layout.addLayout(buttons_layout)

        # Training log
        self.training_log = QTextEdit()
        self.training_log.setReadOnly(True)
        training_layout.addWidget(self.training_log)

        training_tab.setLayout(training_layout)
        tabs.addTab(training_tab, "Training")

        # Recognition tab
        recognition_tab = QWidget()
        recognition_layout = QVBoxLayout()

        load_test_btn = QPushButton("Load Test Images")
        load_test_btn.clicked.connect(self.load_test_images)
        recognition_layout.addWidget(load_test_btn)

        # Создаем скроллируемую область для результатов
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        self.results_layout = QVBoxLayout(scroll_widget)
        scroll_area.setWidget(scroll_widget)
        recognition_layout.addWidget(scroll_area)

        recognition_tab.setLayout(recognition_layout)
        tabs.addTab(recognition_tab, "Recognition")

    def preprocess_image(self, image_path):
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img_array = np.array(img)
        # Преобразуем в бинарный массив (0 для черных пикселей, 1 для белых)
        binary_array = (img_array > 128).astype(int)
        return binary_array.flatten()

    def load_training_images(self):
        self.training_log.clear()
        directory = QFileDialog.getExistingDirectory(self, "Select Training Directory")
        if directory:
            self.training_images = []
            self.training_labels = []

            # Загружаем метки из JSON файла
            with open(os.path.join(directory, 'labels.json'), 'r') as f:
                labels = json.load(f)

            for filename, label in labels.items():
                img_path = os.path.join(directory, filename)
                if os.path.exists(img_path):
                    img_array = self.preprocess_image(img_path)
                    self.training_images.append(img_array)
                    self.training_labels.append(label)

            self.training_log.append(f"Loaded {len(self.training_images)} training images")

    def train_neural_network(self):
        self.training_log.clear()
        if not self.training_images:
            self.training_log.append("No training images loaded!")
            return

        input_size = self.input_size_spin.value()
        population_size = self.population_size_spin.value()
        mutation_rate = self.mutation_rate_spin.value()
        epochs = self.epochs_spin.value()

        self.training_log.append(f"Input Size: {input_size}\tMutation Rate: {mutation_rate}\tPopulation Size: {population_size}\tEpochs: {epochs}")

        self.neural_network = GeneticAlgorithm(input_size, mutation_rate, population_size)

        for epoch in range(epochs):
            total_error = 0
            for inputs, target in zip(self.training_images, self.training_labels):
                error, weights = self.neural_network.train(inputs, target)
                total_error += error

            self.training_log.append(f"Epoch {epoch + 1}, Error: {total_error}")
            if total_error == 0:
                break

        self.training_log.append("Training completed!")
        self.training_log.append(f"Final weights: {self.neural_network.weights}")

    def load_test_images(self):
        global result_widget
        if not self.neural_network:
            self.training_log.append("Train the perceptron first!")
            return

        directory = QFileDialog.getExistingDirectory(self, "Select Test Directory")
        if not directory:
            return

        # Загружаем метки из JSON файла
        try:
            with open(os.path.join(directory, 'labels.json'), 'r') as f:
                labels = json.load(f)
        except FileNotFoundError:
            self.training_log.append("Labels file not found!")
            return

        # Очищаем предыдущие результаты
        for i in reversed(range(self.results_layout.count())):
            self.results_layout.itemAt(i).widget().setParent(None)

        for filename, expected in labels.items():
            img_path = os.path.join(directory, filename)
            if os.path.exists(img_path):
                img_array = self.preprocess_image(img_path)
                raw_prediction = self.neural_network.predict(img_array)
                activated_prediction = self.neural_network.activation(raw_prediction)
                # Создаем виджет для каждого изображения
                result_widget = QWidget()
                result_layout = QVBoxLayout()

                # Превью изображения
                pixmap = QPixmap(img_path)
                image_label = QLabel()
                image_label.setPixmap(pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio))
                result_layout.addWidget(image_label)

                # Результаты распознавания
                result_text = (f"Prediction: {activated_prediction}\n"
                             f"Expected: {expected}\n"
                             f"Raw output: {raw_prediction:.4f}")
                result_label = QLabel(result_text)
                result_layout.addWidget(result_label)

                # Set text color based on prediction accuracy
                if activated_prediction == expected:
                    result_label.setStyleSheet("color: green;")
                else:
                    result_label.setStyleSheet("color: red;")

                result_layout.addWidget(result_label)

                result_widget.setLayout(result_layout)
                self.results_layout.addWidget(result_widget)

            # Добавляем виджет в вертикальный layout результатов
            self.results_layout.addWidget(result_widget)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageRecognizer()
    window.show()
    sys.exit(app.exec())