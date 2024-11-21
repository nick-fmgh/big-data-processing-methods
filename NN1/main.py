import os
import sys
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QSpinBox, QDoubleSpinBox, QTextEdit, QTabWidget, QScrollArea)
from PyQt6.QtGui import QPixmap, QImage, QTextCursor
from PyQt6.QtCore import Qt
from PIL import Image


class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.learning_rate = learning_rate

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights) + self.bias
        return self.activation(summation), summation

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
        self.learning_rate_spin.setRange(0.001, 1.0)
        self.learning_rate_spin.setValue(0.1)
        self.learning_rate_spin.setSingleStep(0.01)
        settings_layout.addWidget(QLabel("Learning Rate:"))
        settings_layout.addWidget(self.learning_rate_spin)

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
        # recognition_tab = QWidget()
        # recognition_layout = QVBoxLayout()
        #
        # load_test_btn = QPushButton("Load Test Image")
        # load_test_btn.clicked.connect(self.load_test_images)
        # recognition_layout.addWidget(load_test_btn)
        #
        # self.test_image_label = QLabel()
        # recognition_layout.addWidget(self.test_image_label)
        #
        # self.recognition_result = QLabel()
        # recognition_layout.addWidget(self.recognition_result)
        #
        # recognition_tab.setLayout(recognition_layout)
        # tabs.addTab(recognition_tab, "Recognition")
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
        img = img.resize((3, 3))  # Resize to 3x3
        img_array = np.array(img).flatten() / 255.0  # Normalize
        return img_array

    def load_training_images(self):
        file_names, _ = QFileDialog.getOpenFileNames(self, "Select Training Images")
        if file_names:
            self.training_images = []
            self.training_labels = []
            for file_name in file_names:
                img_array = self.preprocess_image(file_name)
                self.training_images.append(img_array)
                # Здесь можно добавить диалог для установки метки (0 или 1)
                self.training_labels.append(1)  # Упрощенно всегда 1

            self.training_log.append(f"Loaded {len(self.training_images)} training images")

    def train_perceptron(self):
        if not self.training_images:
            self.training_log.append("No training images loaded!")
            return

        input_size = self.input_size_spin.value()
        learning_rate = self.learning_rate_spin.value()

        self.perceptron = Perceptron(input_size, learning_rate)

        epochs = 100
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
            self.recognition_result.setText("Train the perceptron first!")
            return

        file_names, _ = QFileDialog.getOpenFileNames(self, "Select Test Images")
        # Очищаем предыдущие результаты
        for i in reversed(range(self.results_layout.count())):
            self.results_layout.itemAt(i).widget().setParent(None)
        for file_name in file_names:
            img_array = self.preprocess_image(file_name)
            prediction, summation = self.perceptron.predict(img_array)

            # # Display image
            # pixmap = QPixmap(file_name)
            # self.test_image_label.setPixmap(pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio))
            #
            # # Display results
            # self.recognition_result.setText(
            #     f"Prediction: {prediction}\n"
            #     f"Raw output: {summation:.4f}"
            # )
            # Создаем виджет для каждого изображения
            result_widget = QWidget()
            result_layout = QVBoxLayout()

            # Превью изображения
            pixmap = QPixmap(file_name)
            image_label = QLabel()
            image_label.setPixmap(pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio))
            result_layout.addWidget(image_label)

            # Результаты распознавания
            result_text = f"Prediction: {prediction}\nRaw output: {summation:.4f}"
            result_label = QLabel(result_text)
            result_layout.addWidget(result_label)

            result_widget.setLayout(result_layout)

            # Добавляем виджет в вертикальный layout результатов
            self.results_layout.addWidget(result_widget)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageRecognizer()
    window.show()
    sys.exit(app.exec())