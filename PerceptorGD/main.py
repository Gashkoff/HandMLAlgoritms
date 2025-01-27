import random
from data_config import *
import numpy as np
from matplotlib import pyplot as plt


class PerceptronModel:
    """Класс модели Перцептрона"""
    def __init__(self):
        self.LEARNING_DATASET = self.generate_dataset_second(100)

    def generate_dataset_second(self, matrix_volume: int):
        """Создание shuffle обучающей выборки"""
        dataset = []

        for vector_num in range(3):
            for key in DATASET.keys():
                for item in range(matrix_volume):
                    dataset.append({key: self.normalize(np.array(DATASET[key][vector_num]))})

        random.shuffle(dataset)

        return dataset

    @staticmethod
    def normalize(vector):
        """Нормализация Min-Max признаков"""
        x_min = vector.min()
        x_max = vector.max()

        return (vector - x_min) / (x_max - x_min)

    @staticmethod
    def softmax(x):
        """Функция активации Softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    @staticmethod
    def cross_entropy_loss(y_true, y_pred, epsilon=1e-12):
        """Функция потерь перекрестная энтропия"""
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / len(y_true)

    def forward(self, class_vector, weights, biases):
        """Прямой проход нейронов Перцептрона. Скалярное умножение векторов"""
        z = np.dot(weights, class_vector) + biases
        return self.softmax(z)

class PerceptronOne(PerceptronModel):
    """Первый слой. Перцептрон"""
    def __init__(self):
        super().__init__()
        self.classes_detect = [0, 2, 3, 7]
        self.weights = np.random.uniform(-0.1, 0.1, size=(5, 20))
        self.all_classes = [0, 2, 3, 7, 99]
        self.biases = np.zeros(5)
        self.learning_rate = 0.5
        self.losses_per_epoch = []

    def probabilities_calculation(self, sign):
        return super().forward(sign, self.weights, self.biases)

    @staticmethod
    def one_hot_encode(class_index, num_classes):
        one_hot = np.zeros(num_classes)

        match class_index:
            case 0:
                one_hot[0] = 1
            case 2:
                one_hot[1] = 1
            case 3:
                one_hot[2] = 1
            case 7:
                one_hot[3] = 1
            case 99:
                one_hot[4] = 1
            case _:
                raise Exception
        return one_hot

    def map_class_to_group(self, class_index):
        if class_index in self.classes_detect:
            return class_index
        else:
            return 99

    def update_weights(self, vector, y_true, y_pred):
        """Функция оптимизации градиентный спуск"""
        error_diff = y_pred - y_true

        gradients = np.outer(error_diff, vector)

        for i in range(self.weights.shape[0]):
            for j in range(self.weights.shape[1]):
                self.weights[i, j] -= self.learning_rate * gradients[i, j]

        self.biases -= self.learning_rate * error_diff

    def train(self):
        """Метод обучения"""
        for epoch in range(500):
            losses_function = []
            total_loss = 0

            for face in self.LEARNING_DATASET:

                for authentic_class, vector in face.items():
                    mapped_class = self.map_class_to_group(authentic_class)

                    y_pred = self.probabilities_calculation(vector)
                    y_true = self.one_hot_encode(mapped_class, len(self.classes_detect) + 1)
                    error = self.cross_entropy_loss(y_true, y_pred)
                    losses_function.append(error)
                    total_loss += error

                    self.update_weights(vector, y_true, y_pred)

            # Сохраняем среднюю потерю на эпоху
            avg_loss = total_loss / len(self.LEARNING_DATASET)
            self.losses_per_epoch.append(avg_loss)
            print(f"Epoch {epoch + 1}/{500}, Loss: {avg_loss:.4f}")

        self.loss_plot()

    def loss_plot(self):
        """Построение графика функции потерь по эпохам"""
        plt.plot(self.losses_per_epoch, label="Функция потерь", linestyle='-', marker='')
        plt.xlabel("Эпохи")
        plt.ylabel("Потери")
        plt.title("Потери 1 слоя")
        plt.legend()
        plt.show()

    def detect_new(self, rights_class, vector):
        """Метод предсказания"""
        y_pred = self.probabilities_calculation(vector)
        predicted_class = np.argmax(y_pred)

        print(f"Истинный класс: {SURNAMES[rights_class]} | {rights_class}, распознанный класс: {self.all_classes[predicted_class]}")
        return self.all_classes[predicted_class]


class PerceptronTwo(PerceptronModel):
    """Второй слой. Перцептрон"""
    def __init__(self):
        super().__init__()
        self.classes_detect = [1, 5, 8, 9]
        self.weights = np.random.uniform(-0.1, 0.1, size=(5, 20))
        self.all_classes = [1, 5, 8, 9, 99]
        self.biases = np.zeros(5)
        self.learning_rate = 0.5
        self.losses_per_epoch = []

    def probabilities_calculation(self, sign):
        return super().forward(sign, self.weights, self.biases)

    @staticmethod
    def one_hot_encode(class_index, num_classes):
        one_hot = np.zeros(num_classes)

        match class_index:
            case 1:
                one_hot[0] = 1
            case 5:
                one_hot[1] = 1
            case 8:
                one_hot[2] = 1
            case 9:
                one_hot[3] = 1
            case 99:
                one_hot[4] = 1
            case _:
                raise Exception
        return one_hot

    def map_class_to_group(self, class_index):
        if class_index in self.classes_detect:
            return class_index  # Возвращаем 0, 1 или 2 как есть
        else:
            return 99

    def update_weights(self, vector, y_true, y_pred):
        error_diff = y_pred - y_true

        gradients = np.outer(error_diff, vector)

        for i in range(self.weights.shape[0]):
            for j in range(self.weights.shape[1]):
                self.weights[i, j] -= self.learning_rate * gradients[i, j]

        self.biases -= self.learning_rate * error_diff

    def train(self):
        for epoch in range(500):
            losses_function = []
            total_loss = 0

            for face in self.LEARNING_DATASET:

                for authentic_class, vector in face.items():
                    mapped_class = self.map_class_to_group(authentic_class)

                    y_pred = self.probabilities_calculation(vector)
                    y_true = self.one_hot_encode(mapped_class, len(self.classes_detect) + 1)
                    error = self.cross_entropy_loss(y_true, y_pred)
                    losses_function.append(error)
                    total_loss += error

                    self.update_weights(vector, y_true, y_pred)

            # Сохраняем среднюю потерю на эпоху
            avg_loss = total_loss / len(self.LEARNING_DATASET)
            self.losses_per_epoch.append(avg_loss)
            print(f"Epoch {epoch + 1}/{500}, Loss: {avg_loss:.4f}")

        self.loss_plot()

    def loss_plot(self):
        plt.plot(self.losses_per_epoch, label="Функция потерь", linestyle='-', marker='')
        plt.xlabel("Эпохи")
        plt.ylabel("Потери")
        plt.title("Потери 2 слоя")
        plt.legend()
        plt.show()

    def detect_new(self, rights_class, vector):
        y_pred = self.probabilities_calculation(vector)  # Получить вероятности
        predicted_class = np.argmax(y_pred)  # Найти индекс максимальной вероятности

        print(f"Истинный класс: {SURNAMES[rights_class]} | {rights_class}, распознанный класс: {self.all_classes[predicted_class]}")
        return self.all_classes[predicted_class]


class PerceptronThree(PerceptronModel):
    """Третий слой. Перцептрон"""
    def __init__(self):
        super().__init__()
        self.classes_detect = [4, 10, 11, 13]
        self.weights = np.random.uniform(-0.1, 0.1, size=(5, 20))
        self.all_classes = [4, 10, 11, 13, 99]
        self.biases = np.zeros(5)
        self.learning_rate = 0.5
        self.losses_per_epoch = []

    def probabilities_calculation(self, sign):
        return super().forward(sign, self.weights, self.biases)

    @staticmethod
    def one_hot_encode(class_index, num_classes):
        one_hot = np.zeros(num_classes)

        match class_index:
            case 4:
                one_hot[0] = 1
            case 10:
                one_hot[1] = 1
            case 11:
                one_hot[2] = 1
            case 13:
                one_hot[3] = 1
            case 99:
                one_hot[4] = 1
            case _:
                raise Exception
        return one_hot

    def map_class_to_group(self, class_index):
        if class_index in self.classes_detect:
            return class_index  # Возвращаем 0, 1 или 2 как есть
        else:
            return 99

    def update_weights(self, vector, y_true, y_pred):
        error_diff = y_pred - y_true

        gradients = np.outer(error_diff, vector)

        for i in range(self.weights.shape[0]):
            for j in range(self.weights.shape[1]):
                self.weights[i, j] -= self.learning_rate * gradients[i, j]

        self.biases -= self.learning_rate * error_diff

    def train(self):
        for epoch in range(500):
            losses_function = []
            total_loss = 0

            for face in self.LEARNING_DATASET:

                for authentic_class, vector in face.items():
                    mapped_class = self.map_class_to_group(authentic_class)

                    y_pred = self.probabilities_calculation(vector)
                    y_true = self.one_hot_encode(mapped_class, len(self.classes_detect) + 1)
                    error = self.cross_entropy_loss(y_true, y_pred)
                    losses_function.append(error)
                    total_loss += error

                    self.update_weights(vector, y_true, y_pred)

            # Сохраняем среднюю потерю на эпоху
            avg_loss = total_loss / len(self.LEARNING_DATASET)
            self.losses_per_epoch.append(avg_loss)
            print(f"Epoch {epoch + 1}/{500}, Loss: {avg_loss:.4f}")

        self.loss_plot()

    def loss_plot(self):
        plt.plot(self.losses_per_epoch, label="Функция потерь", linestyle='-', marker='')
        plt.xlabel("Эпохи")
        plt.ylabel("Потери")
        plt.title("Потери 3 слоя")
        plt.legend()
        plt.show()

    def detect_new(self, rights_class, vector):
        y_pred = self.probabilities_calculation(vector)  # Получить вероятности
        predicted_class = np.argmax(y_pred)  # Найти индекс максимальной вероятности

        print(f"Истинный класс: {SURNAMES[rights_class]} | {rights_class}, распознанный класс: {self.all_classes[predicted_class]}")
        return self.all_classes[predicted_class]


class PerceptronFour(PerceptronModel):
    """Четвертый слой. Перцептрон"""
    def __init__(self):
        super().__init__()
        self.classes_detect = [6, 12, 14]
        self.weights = np.random.uniform(-0.1, 0.1, size=(4, 20))
        self.all_classes = [6, 12, 14, 99]
        self.biases = np.zeros(4)
        self.learning_rate = 0.5
        self.losses_per_epoch = []

    def probabilities_calculation(self, sign):
        return super().forward(sign, self.weights, self.biases)

    @staticmethod
    def one_hot_encode(class_index, num_classes):
        one_hot = np.zeros(num_classes)

        match class_index:
            case 6:
                one_hot[0] = 1
            case 12:
                one_hot[1] = 1
            case 14:
                one_hot[2] = 1
            case 99:
                one_hot[3] = 1
            case _:
                raise Exception
        return one_hot

    def map_class_to_group(self, class_index):
        if class_index in self.classes_detect:
            return class_index  # Возвращаем 0, 1 или 2 как есть
        else:
            return 99

    def update_weights(self, vector, y_true, y_pred):
        error_diff = y_pred - y_true

        gradients = np.outer(error_diff, vector)

        for i in range(self.weights.shape[0]):
            for j in range(self.weights.shape[1]):
                self.weights[i, j] -= self.learning_rate * gradients[i, j]

        self.biases -= self.learning_rate * error_diff

    def train(self):
        for epoch in range(500):
            losses_function = []
            total_loss = 0

            for face in self.LEARNING_DATASET:

                for authentic_class, vector in face.items():
                    mapped_class = self.map_class_to_group(authentic_class)

                    y_pred = self.probabilities_calculation(vector)
                    y_true = self.one_hot_encode(mapped_class, len(self.classes_detect) + 1)
                    error = self.cross_entropy_loss(y_true, y_pred)
                    losses_function.append(error)
                    total_loss += error

                    self.update_weights(vector, y_true, y_pred)

            # Сохраняем среднюю потерю на эпоху
            avg_loss = total_loss / len(self.LEARNING_DATASET)
            self.losses_per_epoch.append(avg_loss)
            print(f"Epoch {epoch + 1}/{500}, Loss: {avg_loss:.4f}")

        self.loss_plot()

    def loss_plot(self):
        plt.plot(self.losses_per_epoch, label="Функция потерь", linestyle='-', marker='')
        plt.xlabel("Эпохи")
        plt.ylabel("Потери")
        plt.title("Потери 4 слоя")
        plt.legend()
        plt.show()

    def detect_new(self, rights_class, vector):
        y_pred = self.probabilities_calculation(vector)  # Получить вероятности
        predicted_class = np.argmax(y_pred)  # Найти индекс максимальной вероятности

        print(f"Истинный класс: {SURNAMES[rights_class]} | {rights_class}, распознанный класс: {self.all_classes[predicted_class]}")
        return self.all_classes[predicted_class]

def modela_detect(first: PerceptronOne,
                  second: PerceptronTwo,
                  third: PerceptronThree,
                  four: PerceptronFour,
                  vector: list[int]):
    layer_one = first.detect_new(0, vector)
    if layer_one == 99:
        layer_two = second.detect_new(0, vector)
        if layer_two == 99:
            layer_three = third.detect_new(0, vector)
            if layer_three == 99:
                layer_four = four.detect_new(0, vector)
                if layer_four == 99:
                    print("Невозможно определить персонажа")
                else:
                    print(f"Это {SURNAMES[layer_four]}")
            else:
                print(f"Это {SURNAMES[layer_three]}")
        else:
            print(f"Это {SURNAMES[layer_two]}")
    else:
        print(f"Это {SURNAMES[layer_one]}")

def main():
    model_one = PerceptronOne()
    model_one.train()

    for class_num, vectors in DATASET.items():
        for vector in vectors:
            vector = model_one.normalize(np.array(vector))
            model_one.detect_new(class_num, vector)

    print()
    print(model_one.weights)
    print()
    model_two = PerceptronTwo()
    model_two.train()

    for class_num, vectors in DATASET.items():
        for vector in vectors:
            vector = model_two.normalize(np.array(vector))
            model_two.detect_new(class_num, vector)

    print()
    print(model_two.weights)
    model_three = PerceptronThree()
    model_three.train()

    for class_num, vectors in DATASET.items():
        for vector in vectors:
            vector = model_three.normalize(np.array(vector))
            model_three.detect_new(class_num, vector)

    print()
    print(model_three.weights)
    model_four = PerceptronFour()
    model_four.train()

    for class_num, vectors in DATASET.items():
        for vector in vectors:
            vector = model_four.normalize(np.array(vector))
            model_four.detect_new(class_num, vector)

    print()
    print(model_four.weights)

if __name__ == "__main__":
    main()