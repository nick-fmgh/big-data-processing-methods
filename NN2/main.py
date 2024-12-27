import math
import os
import sys
import json
import random
from idlelib.debugobj_r import remote_object_tree_item

import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QPushButton, QLabel, QFileDialog,
                            QSpinBox, QDoubleSpinBox, QTextEdit, QTabWidget, QScrollArea)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtCore import Qt
from PIL import Image

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot as plt


DEFAULT_INPUT_SIZE=16
DEFAULT_EPOCHS=100

DEFAULT_LEARNING_RATE=0.01


class Layer:
    def __init__(self, neuron_count, learning_rate=0.01):
        self.bias = np.random.randn()
        self.learning_rate = learning_rate
        self.neuron_count = neuron_count
        self.weights = np.random.uniform(-1, 1, self.neuron_count)
        self.last_output = None

    def activate(self, x):
        return 1 / (1 + np.exp(-x))

    def activate_derivative(self, x):
        return x * (1 - x)

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights) + self.bias
        self.last_output = self.activate(summation)
        return self.last_output

    def train_layer(self, inputs, error):
        local_gradient = self.activate_derivative(self.last_output) * error

        inputs = np.array(inputs).reshape(-1)
        weight_updates = self.learning_rate * local_gradient * inputs[:, np.newaxis]

        self.weights += np.sum(weight_updates, axis=0)
        self.bias += self.learning_rate * np.sum(local_gradient)


class NeuralNetwork:
    def __init__(self, n, k, learning_rate=0.1):
        self.hidden_layer = Layer(n, learning_rate)
        self.output_layer = Layer(k, learning_rate)
        self.is_running = True
        self.progress_callback = None

    def set_progress_callback(self, callback):
        self.progress_callback = callback

    def stop_training(self):
        self.is_running = False
    def predict(self, inputs):
        hidden_output = self.hidden_layer.predict(inputs)
        final_output = self.output_layer.predict(hidden_output)
        return max(final_output)
    
    def train(self, training_images, training_labels, epochs):
        for epoch in range(epochs):
            if not self.is_running:
                break

            total_error = 0
            for inputs, target in zip(training_images, training_labels):

                hidden_output = self.hidden_layer.predict(inputs)
                final_output = self.output_layer.predict(hidden_output)

                error = target - final_output
                total_error += np.sum(error ** 2)

                self.output_layer.train_layer(hidden_output, error)
                hidden_error = np.dot(error, self.output_layer.weights)
                self.hidden_layer.train_layer(inputs, hidden_error)

            if self.progress_callback:
                self.progress_callback(f"Epoch {epoch + 1}, Error: {total_error}")


class TrainingThread(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)

    def __init__(self, neural_network, training_images, training_labels, epochs):
        super().__init__()
        self.neural_network = neural_network
        self.training_images = training_images
        self.training_labels = training_labels
        self.epochs = epochs
        self.is_running = True
        # Устанавливаем callback для логирования
        self.neural_network.set_progress_callback(self.progress.emit)

    def stop(self):
        self.is_running = False
        self.neural_network.stop_training()  # Добавляем остановку

    def run(self):
        self.neural_network.train(self.training_images, self.training_labels, self.epochs)
        self.finished.emit(self.neural_network)


class PlotThread(QThread):
    """Поток для асинхронного обновления графика"""
    plot_updated = pyqtSignal()

    def __init__(self, ax, error_history):
        super().__init__()
        self.ax = ax
        self.error_history = error_history

    def run(self):
        self.ax.clear()
        epochs = range(1, len(self.error_history) + 1)
        self.ax.plot(epochs, self.error_history, 'b-')
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Error')
        self.ax.set_title('Training Error Distribution')
        self.ax.grid(True)
        self.ax.relim()
        self.ax.autoscale_view()
        self.plot_updated.emit()

class ImageRecognizer(QMainWindow):
    """
        Главное окно приложения для распознавания изображений с использованием перцептрона.

        Функциональность:
        - Загрузка и предобработка обучающих изображений
        - Настройка параметров обучения нейронной сети
        - Обучение с помощью генетического алгоритма
        - Тестирование на новых изображениях

        Attributes:
            training_images (list): Набор обучающих изображений
            training_labels (list): Метки для обучающих изображений
            neural_network (NeuralNetwork): Экземпляр нейронной сети
            input_size_spin (QSpinBox): Размер входного слоя
            epochs_spin (QSpinBox): Количество эпох обучения
            population_size_spin (QSpinBox): Размер популяции
            learning_rate_spin (QDoubleSpinBox): Коэффициент мутации
            training_log (QTextEdit): Лог процесса обучения
            results_layout (QVBoxLayout): Область отображения результатов
        """
    def __init__(self):
        super().__init__()
        # plot
        self.plot_btn = None
        self.plot_window = None
        self.figure = None
        self.canvas = None
        self.ax = None

        self.plot_thread = None

        self.learning_rate_spin = None
        self.epochs_spin = None
        self.input_size_spin = None
        self.setWindowTitle("Image Recognition with Perceptron")
        self.setGeometry(100, 100, 800, 600)
        self.training_thread = None
        self.training_images = []
        self.training_labels = []
        self.neural_network = None
        self.error_history = []

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
        settings_layout.addWidget(QLabel("Image Size:"))
        settings_layout.addWidget(self.input_size_spin)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(DEFAULT_EPOCHS)
        self.epochs_spin.setSingleStep(100)
        settings_layout.addWidget(QLabel("Epochs:"))
        settings_layout.addWidget(self.epochs_spin)

        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setDecimals(4)
        self.learning_rate_spin.setRange(0.00001, 1.0)
        self.learning_rate_spin.setSingleStep(0.01)
        self.learning_rate_spin.setValue(DEFAULT_LEARNING_RATE)
        settings_layout.addWidget(QLabel("Learning Rate:"))
        settings_layout.addWidget(self.learning_rate_spin)

        training_layout.addLayout(settings_layout)

        # Buttons
        buttons_layout = QHBoxLayout()
        load_btn = QPushButton("Load Training Images")
        load_btn.clicked.connect(self.load_training_images)
        self.train_btn = QPushButton("Train")
        self.train_btn.clicked.connect(self.train_neural_network)
        buttons_layout.addWidget(load_btn)
        buttons_layout.addWidget(self.train_btn)

        # кнопка для вывода графика
        self.plot_btn = QPushButton("Show Error Plot")
        self.plot_btn.clicked.connect(self.show_error_plot)
        buttons_layout.addWidget(self.plot_btn)

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
        if self.train_btn.text() == "Stop":
            self.training_thread.stop()
            self.train_btn.setText("Train")
            return

        self.error_history = []
        self._update_plot_data()

        self.training_log.clear()
        self.training_log.append(f"Loaded {len(self.training_images)} training images")

        if not self.training_images:
            self.training_log.append("No training images loaded!")
            return

        self.train_btn.setText("Stop")
        self.training_log.append("Training started...")

        input_size = self.input_size_spin.value()
        learning_rate = self.learning_rate_spin.value()
        epochs = self.epochs_spin.value()
        classes_count = len(set(self.training_labels))
        self.training_log.append(f"Input Size: {input_size}\tLearning Rate: {learning_rate}\tEpochs: {epochs}")

        self.neural_network = NeuralNetwork(input_size, classes_count, learning_rate)
        self.training_thread = TrainingThread(
            self.neural_network,
            self.training_images,
            self.training_labels,
            epochs
        )
        self.training_thread.progress.connect(self.update_training_log)
        self.training_thread.finished.connect(self.training_finished)
        self.training_thread.start()


    def update_training_log(self, message):
        self.training_log.append(message)
        if "Error:" in message:
            error = float(message.split("Error: ")[1])
            self.error_history.append(error)
            self._update_plot_data()

    def training_finished(self, trained_network):
        self.neural_network = trained_network
        self.training_log.append("Training completed!")
        self.training_log.append(f"Final hidden layer weights: {self.neural_network.hidden_layer.weights}\n"
                                 f"Final output layer weights: {self.neural_network.output_layer.weights}\n")
        self.train_btn.setText("Train")

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
                prediction = self.neural_network.predict(img_array)
                rounded_prediction = round(prediction)
                # Создаем виджет для каждого изображения
                result_widget = QWidget()
                result_layout = QVBoxLayout()

                # Превью изображения
                pixmap = QPixmap(img_path)
                image_label = QLabel()
                image_label.setPixmap(pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio))
                result_layout.addWidget(image_label)

                # Результаты распознавания
                result_text = (
                    f"Prediction: {rounded_prediction}\n"
                    f"Expected: {expected}\n"
                    f"Raw: {prediction:.4f}"
                )
                result_label = QLabel(result_text)
                result_layout.addWidget(result_label)

                # Set text color based on prediction accuracy
                if rounded_prediction == expected:
                    result_label.setStyleSheet("color: green;")
                else:
                    result_label.setStyleSheet("color: red;")

                result_layout.addWidget(result_label)

                result_widget.setLayout(result_layout)
                self.results_layout.addWidget(result_widget)

            # Добавляем виджет в вертикальный layout результатов
            self.results_layout.addWidget(result_widget)

    def show_error_plot(self):
        if not self.error_history:
            self.training_log.append("No training data available")
            return

        if not hasattr(self, 'plot_window') or self.plot_window is None:
            self._create_plot_window()
        self._update_plot_data()
        self.plot_window.show()

    def _create_plot_window(self):
        self.plot_window = QWidget()
        self.plot_window.setWindowTitle("Error Distribution")

        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)

        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.canvas)
        self.plot_window.setLayout(plot_layout)

    def _update_plot_data(self):
        if self.ax is None:
            return

        if self.plot_thread is not None and self.plot_thread.isRunning():
            return

        self.plot_thread = PlotThread(self.ax, self.error_history)
        self.plot_thread.plot_updated.connect(self._on_plot_updated)
        self.plot_thread.start()

    def _on_plot_updated(self):
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageRecognizer()
    window.show()
    sys.exit(app.exec())