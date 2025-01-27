from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
from mathMethods.mathFunctions import MathFunctions


class StateValues:
    """
    Класс для решения классификации объектов из обучающей выборке

    Реализованы методы построения графика на каждом этапе, а также вывод результатов
    """
    uncertainty_functions = {
        "B-ACD": {},
        "D-AC": {},
        "A-C": {},
    }

    def __init__(self):
        """Обучающая выборка"""
        self.classes_learn_dataset: dict[str, dict[int, list[int]]] = {
            "A": {
                1: [21, 23, 2, 52, 11, 26, 22, 27, 29, 10],
                2: [29, 24, 3, 44, 13, 22, 26, 25, 22, 7],
                3: [22, 22, 4, 30, 19, 15, 38, 28, 27, 9],
                4: [27, 28, 5, 34, 16, 17, 32, 26, 25, 8],
                5: [25, 26, 6, 54, 12, 27, 24, 23, 28, 9],
                6: [21, 23, 8, 46, 11, 23, 22, 21, 26, 9],
                7: [28, 21, 9, 26, 17, 13, 34, 29, 23, 8],
                8: [26, 29, 10, 36, 18, 18, 36, 25, 21, 7],
                9: [23, 22, 10, 58, 13, 29, 26, 23, 29, 10],
                10: [21, 27, 11, 50, 11, 25, 22, 24, 23, 8],
                11: [29, 25, 13, 28, 15, 14, 30, 22, 24, 8],
                12: [28, 28, 14, 42, 16, 21, 32, 28, 22, 7],
                13: [23, 26, 15, 56, 19, 28, 38, 26, 28, 9],
                14: [24, 23, 16, 34, 17, 17, 34, 23, 26, 9],
                15: [22, 21, 17, 48, 12, 24, 24, 21, 23, 8],
            },
            "B": {
                1: [21, 29, 24, 24, 13, 12, 26, 29, 21, 7],
                2: [28, 23, 25, 12, 11, 6, 22, 22, 29, 10],
                3: [26, 24, 26, 18, 15, 9, 30, 27, 22, 7],
                4: [23, 22, 27, 8, 16, 4, 32, 25, 27, 9],
                5: [21, 28, 27, 28, 19, 14, 38, 28, 25, 8],
                6: [29, 26, 29, 24, 17, 12, 34, 27, 28, 9],
                7: [28, 23, 30, 16, 13, 8, 26, 25, 26, 9],
                8: [23, 21, 30, 30, 19, 15, 38, 28, 23, 8],
                9: [24, 29, 32, 4, 16, 2, 32, 26, 21, 7],
                10: [22, 22, 32, 22, 12, 11, 24, 23, 29, 10],
                11: [28, 27, 33, 12, 11, 6, 22, 27, 25, 8],
                12: [26, 25, 34, 26, 17, 13, 34, 25, 23, 8],
                13: [23, 28, 35, 6, 18, 3, 36, 28, 24, 8],
                14: [21, 26, 36, 20, 11, 10, 22, 26, 22, 7],
                15: [29, 23, 37, 12, 14, 6, 28, 23, 28, 9],
            },
            "C": {
                1: [22, 21, 20, 88, 17, 44, 34, 21, 26, 9],
                2: [27, 29, 21, 96, 13, 48, 26, 29, 23, 8],
                3: [25, 25, 22, 80, 19, 40, 38, 25, 21, 7],
                4: [28, 23, 23, 90, 16, 45, 32, 23, 29, 10],
                5: [26, 24, 23, 102, 12, 51, 24, 24, 22, 7],
                6: [23, 22, 25, 106, 11, 53, 22, 22, 27, 9],
                7: [21, 28, 26, 78, 17, 39, 34, 28, 25, 8],
                8: [29, 26, 26, 88, 18, 44, 36, 26, 28, 9],
                9: [25, 23, 26, 100, 11, 50, 22, 23, 29, 10],
                10: [27, 21, 27, 72, 14, 36, 28, 21, 22, 7],
                11: [28, 29, 29, 92, 15, 46, 30, 29, 27, 9],
                12: [26, 22, 30, 82, 13, 41, 26, 22, 25, 8],
                13: [23, 27, 30, 104, 19, 52, 38, 27, 28, 9],
                14: [21, 25, 32, 86, 17, 43, 34, 25, 26, 9],
                15: [29, 28, 32, 96, 12, 48, 24, 27, 23, 8],
            },
            "D": {
                1: [24, 29, 7, 70, 11, 35, 22, 25, 29, 10],
                2: [22, 22, 4, 72, 19, 36, 38, 28, 22, 7],
                3: [28, 27, 7, 74, 13, 37, 26, 26, 29, 10],
                4: [26, 25, 10, 74, 10, 37, 20, 23, 22, 7],
                5: [23, 28, 6, 78, 11, 39, 22, 21, 27, 9],
                6: [21, 26, 9, 78, 18, 39, 36, 29, 25, 8],
                7: [29, 23, 11, 80, 15, 40, 30, 25, 28, 9],
                8: [22, 21, 5, 82, 11, 41, 22, 23, 26, 9],
                9: [27, 29, 7, 84, 13, 42, 26, 24, 23, 8],
                10: [25, 25, 13, 86, 10, 43, 20, 22, 21, 7],
                11: [28, 23, 10, 88, 17, 44, 34, 28, 29, 10],
                12: [26, 24, 8, 90, 11, 45, 22, 26, 23, 8],
                13: [23, 22, 12, 92, 18, 46, 36, 23, 24, 8],
                14: [21, 28, 9, 96, 15, 48, 30, 21, 22, 7],
                15: [29, 26, 11, 100, 11, 50, 22, 29, 28, 7],
            },
        }
        self.Mathematics = MathFunctions(self.classes_learn_dataset)


    def get_corr_matrix(self):
        """Получение матрицы корреляционных взаимосвязей"""
        matrix = self.Mathematics.correlation_matrix()

        for i in matrix.values():
            print(i)

    def get_informative_signs(self):
        """Построение графиков по информативным признакам"""
        b_acd = self.Mathematics.informative_signs(
            [*self.classes_learn_dataset["B"].values()],
            [*self.classes_learn_dataset["A"].values(),
             *self.classes_learn_dataset["C"].values(),
             *self.classes_learn_dataset["D"].values()]
        )
        print(F"{b_acd}")

        d_ac = self.Mathematics.informative_signs(
            [*self.classes_learn_dataset["D"].values()],
            [*self.classes_learn_dataset["A"].values(),
             *self.classes_learn_dataset["C"].values()]
        )
        print(F"{d_ac}")

        a_c = self.Mathematics.informative_signs(
            [*self.classes_learn_dataset["A"].values()],
            [*self.classes_learn_dataset["C"].values()]
        )
        print(F"{a_c}")

    def plots(self):
        """Построение графиков объектов, объектов с ЛДФ и с областью неопределенности Jackknife"""
        # B - ACD 3 - 6

        self.create_informative_plot(2, 5,
                                              [*self.classes_learn_dataset["B"].values()],
                                              [*self.classes_learn_dataset["A"].values(),
                                               *self.classes_learn_dataset["C"].values(),
                                               *self.classes_learn_dataset["D"].values()],
                                     "График отображения объектов классов B и не B(ACD)",
                                     "Признак 3", "Признак 6",
                                     "Класс B", "Класс не B"
                                     )

        self.create_informative_plot_with_func(
            2, 5,
            [*self.classes_learn_dataset["B"].values()],
            [*self.classes_learn_dataset["A"].values(),
             *self.classes_learn_dataset["C"].values(),
             *self.classes_learn_dataset["D"].values()],
            "График отображения объектов классов B и не B(ACD) с ЛДФ",
            "Признак 3", "Признак 6",
            "Класс B", "Класс не B"
        )

        self.create_informative_plot_with_func_jackknife(2, 5,
                                                         [*self.classes_learn_dataset["B"].values()],
                                                         [*self.classes_learn_dataset["A"].values(),
                                                          *self.classes_learn_dataset["C"].values(),
                                                          *self.classes_learn_dataset["D"].values()],
                                                         "График отображения объектов классов B и не B(ACD) с ЛДФ и Jackknife",
                                                         "Признак 3", "Признак 6",
                                                         "Класс B", "Класс не B")

        # D - AC 6 - 3

        self.create_informative_plot(5, 2,
                                     [*self.classes_learn_dataset["D"].values()],
                                     [*self.classes_learn_dataset["A"].values(),
                                      *self.classes_learn_dataset["C"].values()],
                                     "График отображения объектов классов D и не D(AC)",
                                     "Признак 6", "Признак 3",
                                     "Класс D", "Класс не D"
                                     )

        self.create_informative_plot_with_func(
            5, 2,
            [*self.classes_learn_dataset["D"].values()],
            [*self.classes_learn_dataset["A"].values(),
             *self.classes_learn_dataset["C"].values()],
            "График отображения объектов классов D и не D(AC) с ЛДФ",
            "Признак 6", "Признак 3",
            "Класс D", "Класс не D"
        )

        self.create_informative_plot_with_func_jackknife(5, 2,
                                                         [*self.classes_learn_dataset["D"].values()],
                                                         [*self.classes_learn_dataset["A"].values(),
                                                         *self.classes_learn_dataset["C"].values()],
                                                         "График отображения объектов классов D и не D(AC) с ЛДФ и Jackknife",
                                                         "Признак 6", "Признак 3",
                                                         "Класс D", "Класс не D")

        # A - C 4 - 3

        self.create_informative_plot(2, 3,
                                     [*self.classes_learn_dataset["A"].values()],
                                     [*self.classes_learn_dataset["C"].values()],
                                     "График отображения объектов классов A и не A(C)",
                                     "Признак 3", "Признак 4",
                                     "Класс A", "Класс C"
                                     )

        self.create_informative_plot_with_func(
            2, 3,
            [*self.classes_learn_dataset["A"].values()],
            [*self.classes_learn_dataset["C"].values()],
            "График отображения объектов классов A и не A(C) с ЛДФ",
            "Признак 3", "Признак 4",
            "Класс A", "Класс C"
        )

        self.create_informative_plot_with_func_jackknife(2, 3,
                                                         [*self.classes_learn_dataset["A"].values()],
                                                             [*self.classes_learn_dataset["C"].values()],
                                                         "График отображения объектов классов A и не A(C) с ЛДФ и Jackknife",
                                                         "Признак 3", "Признак 4",
                                                         "Класс A", "Класс C")

    @staticmethod
    def solve_function_coefficients(vector_a: tuple,
                                    vector_b: tuple,
                                    ) -> tuple:
        """Расчет коэффициентов функции ЛДФ"""
        vector_C_y = round(vector_a[0] - vector_b[0], 4)
        vector_C_x = round(vector_a[1] - vector_b[1], 4)
        a = round((vector_a[0] + vector_b[0]), 4)
        b = round((vector_a[1] + vector_b[1]), 4)
        c2 = (vector_C_y * a + vector_C_x * b) * (-0.5)
        coeff_2 = c2 / (vector_C_y * -1)
        coeff_1 = vector_C_x / (vector_C_y * -1)
        return coeff_1, coeff_2

    def devide_classes(self,
                       x: int, y: int,
                       y_class: list, not_class: list, ) -> tuple[list, list, list, list]:
        """Функция разделения классов"""
        class_A_x = []
        class_A_y = []

        class_B_x = []
        class_B_y = []

        for elems in range(len(y_class)):
            class_A_x.append(y_class[elems][x])
            class_A_y.append(y_class[elems][y])

        for elems in range(len(not_class)):
            class_B_x.append(not_class[elems][x])
            class_B_y.append(not_class[elems][y])

        return class_A_x, class_A_y, class_B_x, class_B_y

    @staticmethod
    def average_values(class_A_x, class_A_y,  class_B_x, class_B_y):
        Xa = (round(sum(class_A_x) / len(class_A_x), 4), round(sum(class_A_y) / len(class_A_y), 4))
        Xb = (round(sum(class_B_x) / len(class_B_x), 4), round(sum(class_B_y) / len(class_B_y), 4))

        return Xa, Xb

    def create_informative_plot(self,
                                x: int,
                                y: int,
                                y_class: list,
                                not_class: list,
                                title: str,
                                xlabel: str, ylabel: str,
                                alabel: str, blabel: str,
                                ):
        """Построение графика по информативным признакам"""
        class_A_x, class_A_y, class_B_x, class_B_y = self.devide_classes(x, y, y_class, not_class)

        plt.scatter(class_A_x, class_A_y, color='blue', label=alabel)
        plt.scatter(class_B_x, class_B_y, color='red', label=blabel)

        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.legend()
        plt.show()

    def create_informative_plot_with_func(self,
                                          x: int, y: int,
                                          y_class: list, not_class: list,
                                          title: str,
                                          xlabel: str, ylabel: str,
                                          alabel: str, blabel: str,
                                          ):
        """Построение графика по информативным признакам с ЛДФ"""
        class_A_x, class_A_y, class_B_x, class_B_y = self.devide_classes(x, y, y_class, not_class)
        if len(not_class) == 45:
            vector_a, vector_b = self.average_values(class_A_x, class_A_y, class_B_x, class_B_y)
        else:
            vector_a, vector_b = self.average_values(class_A_y, class_A_x, class_B_y, class_B_x)
        print(vector_a, vector_b)

        plt.figure(figsize=(8, 6))

        plt.scatter(class_A_x, class_A_y, color='blue', label=alabel)
        plt.scatter(class_B_x, class_B_y, color='red', label=blabel)

        plt.xlim(min(class_A_x + class_B_x) - 5, max(class_A_x + class_B_x) + 5)
        plt.ylim(min(class_A_y + class_B_y) - 5, max(class_A_y + class_B_y) + 5)

        x_values = np.linspace(min(class_A_x + class_B_x) - 10, max(class_A_x + class_B_x) + 10, 100)
        coef1, coef2 = self.solve_function_coefficients(vector_a, vector_b)
        y_values = coef1 * x_values + coef2
        print(coef1, coef2)

        plt.plot(x_values, y_values, color='black', linestyle='--', label='ЛДФ')

        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.legend()
        plt.grid()
        plt.show()

    def create_informative_plot_with_func_jackknife(self,
                                                    x: int, y: int,
                                                    y_class: list, not_class: list,
                                                    title: str,
                                                    xlabel: str, ylabel: str,
                                                    alabel: str, blabel: str,
                                                    ):
        """Построение графика по информативным признакам с ЛДФ и ОН Jackknife"""
        # Построение графика с ЛДФ
        class_A_x, class_A_y, class_B_x, class_B_y = self.devide_classes(x, y, y_class, not_class)

        if len(not_class) == 45:
            vector_a, vector_b = self.average_values(class_A_x, class_A_y, class_B_x, class_B_y)
        else:
            vector_a, vector_b = self.average_values(class_A_y, class_A_x, class_B_y, class_B_x)

        maxix = []
        minix = []

        plt.figure(figsize=(8, 6))

        plt.scatter(class_A_x, class_A_y, color='blue', label=alabel)
        plt.scatter(class_B_x, class_B_y, color='red', label=blabel)

        plt.xlim(min(class_A_x + class_B_x) - 5, max(class_A_x + class_B_x) + 5)
        plt.ylim(min(class_A_y + class_B_y) - 5, max(class_A_y + class_B_y) + 5)

        x_values = np.linspace(min(class_A_x + class_B_x) - 10, max(class_A_x + class_B_x) + 10, 100)
        coef1, coef2 = self.solve_function_coefficients(vector_a, vector_b)
        y_values = coef1 * x_values + coef2
        plt.plot(x_values, y_values, color='black', linestyle='--', label='ЛДФ', zorder=5)

        label_added = False

        # Добавление области неопределенности
        for a_elem in range(len(y_class)):
            temp_class_A_x = class_A_x.copy()
            temp_class_A_y = class_A_y.copy()
            temp_class_A_x.pop(a_elem)
            temp_class_A_y.pop(a_elem)

            for b_elem in range(len(not_class)):
                temp_class_B_x = class_B_x.copy()
                temp_class_B_y = class_B_y.copy()
                temp_class_B_x.pop(b_elem)
                temp_class_B_y.pop(b_elem)

                if len(not_class) == 45:
                    vector_jack_a, vector_jack_b = self.average_values(temp_class_A_x, temp_class_A_y, temp_class_B_x, temp_class_B_y)
                else:
                    vector_jack_a, vector_jack_b = self.average_values(temp_class_A_y, temp_class_A_x, temp_class_B_y, temp_class_B_x)


                plt.xlim(min(temp_class_A_x + temp_class_B_x) - 5, max(temp_class_A_x + temp_class_B_x) + 5)
                plt.ylim(min(temp_class_A_y + temp_class_B_y) - 5, max(temp_class_A_y + temp_class_B_y) + 5)

                x_values = np.linspace(min(temp_class_A_x + temp_class_B_x) - 10, max(temp_class_A_x + temp_class_B_x) + 10, 100)
                coef1, coef2 = self.solve_function_coefficients(vector_jack_a, vector_jack_b)
                maxix.append([coef1, coef2])
                minix.append([coef1, coef2])
                y_values = coef1 * x_values + coef2

                if label_added:
                    plt.plot(x_values, y_values, color='grey', linestyle='solid', zorder=1)
                else:
                    plt.plot(x_values, y_values, color='grey', linestyle='solid', zorder=1, label="О.Н.")
                    label_added = True

        max_x_b, = [i for i in maxix if i[0] == max([couple[0] for couple in maxix])]
        min_x_b, = [i for i in maxix if i[0] == min([couple[0] for couple in maxix])]

        match len(not_class):
            case 45:
                self.uncertainty_functions["B-ACD"]["max"] = max_x_b
                self.uncertainty_functions["B-ACD"]["min"] = min_x_b
            case 30:
                self.uncertainty_functions["D-AC"]["max"] = max_x_b
                self.uncertainty_functions["D-AC"]["min"] = min_x_b
            case 15:
                self.uncertainty_functions["A-C"]["max"] = max_x_b
                self.uncertainty_functions["A-C"]["min"] = min_x_b

        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.legend()
        plt.grid()
        plt.show()


class Classification(StateValues):
    """
    Класс для проверки классификации объектов по стратегии OvR(OvA)
    """
    def __init__(self):
        super().__init__()
        self.checking_dataset = {
            "A": {
                1: [28, 23, 6, 40, 13, 20, 26, 27, 25, 8],
                2: [23, 24, 9, 32, 19, 16, 38, 25, 23, 8],
                3: [24, 22, 10, 42, 16, 21, 32, 28, 24, 8],
                4: [22, 28, 12, 54, 12, 27, 24, 26, 22, 7],
                5: [21, 26, 13, 34, 11, 17, 22, 23, 28, 9],
            },
            "B": {
                1: [28, 23, 28, 14, 14, 7, 28, 21, 26, 9],
                2: [26, 21, 30, 8, 15, 4, 30, 29, 23, 8],
                3: [23, 29, 30, 22, 13, 11, 26, 25, 21, 7],
                4: [21, 22, 33, 22, 19, 11, 38, 23, 29, 10],
                5: [29, 27, 35, 16, 17, 8, 34, 24, 22, 7],
            },
            "C": {
                1: [28, 25, 23, 98, 17, 49, 34, 22, 27, 9],
                2: [23, 28, 24, 84, 13, 42, 26, 28, 25, 8],
                3: [24, 26, 26, 94, 19, 47, 38, 26, 28, 9],
                4: [22, 23, 28, 86, 16, 43, 32, 23, 29, 10],
                5: [28, 21, 29, 98, 12, 49, 24, 21, 22, 7],
            },
            "D": {
                1: [24, 22, 11, 70, 11, 35, 22, 27, 27, 9],
                2: [22, 28, 4, 78, 16, 39, 32, 25, 25, 8],
                3: [21, 26, 13, 80, 18, 40 ,36, 28, 28, 9],
                4: [28, 23, 10, 84, 17, 42, 34, 26, 26, 9],
                5: [26, 21 ,12 ,88, 14 ,44, 28, 23, 23, 8],
            }
        }
        self.designated_signs = {
            "B-ACD": {
                "x": 2, "y": 5,
            },
            "D-AC": {
                "x": 5, "y": 2,
            },
            "A-C": {
                "x": 2, "y": 3,
            },
        }
        self.intersection_x = {
            "B-ACD": {
                "x": 25,
            },
            "D-AC": {
                "x": 37.47,
            },
            "A-C": {
                "x": 18,
            },
        }

    def what_class(self):
        print("Проход по проверяющей выборке:")
        for main_class, classes in self.checking_dataset.items():
            for key_object, class_object in classes.items():
                response = self.b_acd(class_object)

                print(f"Объект: {key_object} Класса: {main_class} принадлежит - {response}")

    def b_acd(self, _signs: list):
        """Метод проверки на принадлежность к классу B или к остальным"""
        a_upper, b_upper = self.uncertainty_functions["B-ACD"]["max"]
        a_lower, b_lower = self.uncertainty_functions["B-ACD"]["min"]

        x, y = _signs[self.designated_signs["B-ACD"]["x"]], _signs[self.designated_signs["B-ACD"]["y"]]

        y_upper = a_upper * x + b_upper
        y_lower = a_lower * x + b_lower

        if -1 <= x - self.intersection_x["B-ACD"]["x"] <= 1 and -.5 <= y - y_upper <= .5:
            return "Область неопределенности на узле B-ACD"

        elif x > self.intersection_x["B-ACD"]["x"]:
            zone = y_lower - y

            match zone:
                case z if z > 0:
                    return "Класс B"

                case z if z < 0:
                    new_zone = y_upper - y
                    if new_zone > 0:
                        return "Область неопределенности на узле B-ACD"
                    elif new_zone < 0:
                        return self.d_ac(_signs)
                    else:
                        return "Область неопределенности на узле B-ACD"

                case _:
                    return "Область неопределенности на узле B-ACD"

        elif x < self.intersection_x["B-ACD"]["x"]:
            zone = y_upper - y

            match zone:
                case z if z > 0:
                    return "Класс B"

                case z if z < 0:
                    new_zone = y_lower - y
                    if new_zone > 0:
                        return "Область неопределенности на узле B-ACD"
                    elif new_zone < 0:
                        return self.d_ac(_signs)
                    else:
                        return "Область неопределенности на узле B-ACD"

                case _:
                    return "Область неопределенности на узле B-ACD"

    def d_ac(self, _signs: list):
        """Метод проверки на принадлежность к классу D или к AC"""
        a_upper, b_upper = self.uncertainty_functions["D-AC"]["max"]
        a_lower, b_lower = self.uncertainty_functions["D-AC"]["min"]

        x, y = _signs[self.designated_signs["D-AC"]["x"]], _signs[self.designated_signs["D-AC"]["y"]]

        y_upper = a_upper * x + b_upper
        y_lower = a_lower * x + b_lower

        if -1 <= x - self.intersection_x["D-AC"]["x"] <= 1.5 and -.5 <= y - y_upper <= .5:
            return "Область неопределенности на узле D-AC"

        elif x > self.intersection_x["D-AC"]["x"]:
            zone = y_lower - y

            match zone:
                case z if z > 0:
                    return "Класс D"

                case z if z < 0:
                    new_zone = y_upper - y
                    if new_zone > 0:
                        return "Область неопределенности на узле D-AC"
                    elif new_zone < 0:
                        return self.a_c(_signs)
                    else:
                        return "Область неопределенности на узле D-AC"

                case _:
                    return "Область неопределенности на узле D-AC"

        elif x < self.intersection_x["D-AC"]["x"]:
            zone = y_upper - y

            match zone:
                case z if z > 0:
                    return "Класс D"
                case z if z < 0:
                    new_zone = y_lower - y
                    if new_zone > 0:
                        return "Область неопределенности на узле D-AC"
                    elif new_zone < 0:
                        return self.a_c(_signs)
                    else:
                        return "Область неопределенности на узле D-AC"
                case _:
                    return "Область неопределенности на узле D-AC"

    def a_c(self, _signs: list):
        """Метод проверки на принадлежность к классу A или к C"""
        a_upper, b_upper = self.uncertainty_functions["A-C"]["max"]
        a_lower, b_lower = self.uncertainty_functions["A-C"]["min"]

        x, y = _signs[self.designated_signs["A-C"]["x"]], _signs[self.designated_signs["A-C"]["y"]]

        y_upper = a_upper * x + b_upper
        y_lower = a_lower * x + b_lower

        if -2 <= x - self.intersection_x["A-C"]["x"] <= 2 and -.5 <= y - y_upper <= .5:
            return "Область неопределенности на узле A-C"

        elif x > self.intersection_x["A-C"]["x"]:
            zone = y_lower - y

            match zone:
                case z if z > 0:
                    return "Класс A"

                case z if z < 0:
                    new_zone = y_upper - y
                    if new_zone > 0:
                        return "Область неопределенности на узле A-C"
                    elif new_zone < 0 and abs(new_zone) > 1:
                        return "Класс C"
                    else:
                        return "Область неопределенности на узле A-C"

                case _:
                    return "Область неопределенности на узле A-C"

        elif x < self.intersection_x["A-C"]["x"]:
            zone = y_upper - y

            match zone:
                case z if z > 0:
                    return "Класс A"
                case z if z < 0:
                    new_zone = y_lower - y
                    if new_zone > 0:
                        return "Область неопределенности на узле A-C"
                    elif new_zone < 0 and abs(new_zone) > 1:
                        return "Класс C"
                    else:
                        return "Область неопределенности на узле A-C"
                case _:
                    return "Область неопределенности на узле A-C"


def main():
    object_one = StateValues()
    object_one.get_informative_signs()
    object_one.plots()
    pprint(object_one.uncertainty_functions)
    object_two = Classification()
    object_two.what_class()

if __name__ == '__main__':
    main()
