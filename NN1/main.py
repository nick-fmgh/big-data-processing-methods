import math
import os
import sys
import json
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QPushButton, QLabel, QFileDialog,
                            QSpinBox, QDoubleSpinBox, QTextEdit, QTabWidget, QScrollArea)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
from PIL import Image


def activation(x):
    return 1 if x >= 0 else 0

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.learning_rate = learning_rate

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights) + self.bias
        return summation, summation

    def train(self, inputs, target):
        prediction, summation = self.predict(inputs)
        error = target - prediction
        self.weights += self.learning_rate * error * inputs
        self.bias += self.learning_rate * error
        return error, self.weights.copy(), self.bias


class ImageRecognizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Recognition with Perceptron")
        self.setGeometry(100, 100, 800, 600)

        self.training_images = []
        self.training_labels = []
        self.perceptron = None

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
        self.input_size_spin.setValue(9)
        settings_layout.addWidget(QLabel("Input Size:"))
        settings_layout.addWidget(self.input_size_spin)

        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setDecimals(4)
        self.learning_rate_spin.setRange(0.00001, 1.0)
        self.learning_rate_spin.setSingleStep(0.01)
        self.learning_rate_spin.setValue(0.1)
        settings_layout.addWidget(QLabel("Learning Rate:"))
        settings_layout.addWidget(self.learning_rate_spin)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(100)
        self.epochs_spin.setSingleStep(100)
        settings_layout.addWidget(QLabel("Epochs:"))
        settings_layout.addWidget(self.epochs_spin)

        training_layout.addLayout(settings_layout)

        # Buttons
        buttons_layout = QHBoxLayout()
        load_btn = QPushButton("Load Training Images")
        load_btn.clicked.connect(self.load_training_images)
        train_btn = QPushButton("Train")
        train_btn.clicked.connect(self.train_perceptron)
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

    def train_perceptron(self):
        self.training_log.clear()
        if not self.training_images:
            self.training_log.append("No training images loaded!")
            return

        input_size = self.input_size_spin.value()
        learning_rate = self.learning_rate_spin.value()
        epochs = self.epochs_spin.value()

        self.training_log.append(f"Input Size: {input_size}\tLearning Rate: {learning_rate}\tEpochs: {epochs}")

        self.perceptron = Perceptron(input_size, learning_rate)

        for epoch in range(epochs):
            total_error = 0
            for inputs, target in zip(self.training_images, self.training_labels):
                error, weights, bias = self.perceptron.train(inputs, target)
                total_error += abs(error)

            self.training_log.append(f"Epoch {epoch + 1}, Error: {total_error}")
            if total_error == 0:
                break

        self.training_log.append("Training completed!")
        self.training_log.append(f"Final weights: {self.perceptron.weights}")
        self.training_log.append(f"Final bias: {self.perceptron.bias}")

    def load_test_images(self):
        if not self.perceptron:
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
                prediction, summation = self.perceptron.predict(img_array)
                prediction = self.perceptron.activation(prediction)
                # Создаем виджет для каждого изображения
                result_widget = QWidget()
                result_layout = QVBoxLayout()

                # Превью изображения
                pixmap = QPixmap(img_path)
                image_label = QLabel()
                image_label.setPixmap(pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio))
                result_layout.addWidget(image_label)

                # Результаты распознавания
                result_text = (f"Prediction: {prediction}\n"
                             f"Expected: {expected}\n"
                             f"Raw output: {summation:.4f}")
                result_label = QLabel(result_text)
                result_layout.addWidget(result_label)

                # Set text color based on prediction accuracy
                if prediction == expected:
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