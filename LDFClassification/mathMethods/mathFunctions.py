import math

_QUANTITY_SIGNS: int = 10  # Признаков всего 10


class MathFunctions:
    """
    Класс реализаций математических функций

    Args:
        _learning_dataset: Обучающая выборка в типе dict
    """

    def __init__(self, _learning_dataset: dict):
        self.dataset: dict[str, dict[int, list[int]]] = _learning_dataset
        self.length_dataset: int = 0
        self.concat_weight: list = []
        self.new_correlation_matrix: dict[int, list[int]] = {
            0: [],
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
            6: [],
            7: [],
            8: [],
            9: [],
        }

    def __concatenate_weights(self):
        """Конкатенация всех признаков"""
        for signs in self.dataset.values():
            for elems in signs.values():
                self.concat_weight.append(elems)

        self.length_dataset = len(self.concat_weight)

    def average_sign(self,
                     ind: int  # Индекс, по которому считаем среднее
                     ) -> float:
        """Расчет среднего значения признака"""
        average_sign = 0
        for values in self.concat_weight:
            average_sign += values[ind]

        return round(average_sign / self.length_dataset, 2)

    def standard_deviation(self,
                           m: float,  # Среднее значение столбца посчитанное в average_sign
                           ind: int  # Индекс, по которому считаем значения
                           ) -> float:
        """Расчет среднеквадратического отклонения"""
        counter_results = 0

        for values in self.concat_weight:
            counter_results += (values[ind] - m) ** 2

        return math.sqrt(counter_results / self.length_dataset)

    def correlation_matrix(self) -> dict[int, list[int]]:
        """Расчет матрицы корреляционных взаимосвязей"""
        self.__concatenate_weights()

        for main_sign in range(_QUANTITY_SIGNS):
            m1 = self.average_sign(main_sign)
            dev1 = self.standard_deviation(m1, main_sign)

            for second_sign in range(_QUANTITY_SIGNS):
                m2 = self.average_sign(second_sign)
                dev2 = self.standard_deviation(m2, second_sign)

                sums = 0
                for elem in range(self.length_dataset):
                    sums += (self.concat_weight[elem][main_sign] - m1) * (self.concat_weight[elem][second_sign] - m2)

                sums /= self.length_dataset

                self.new_correlation_matrix[main_sign].append(abs(round(sums / (dev1 * dev2), 4)))

        return self.new_correlation_matrix

    @staticmethod
    def informative_average_sign(ind: int,
                                 array: list) -> float:
        """Расчет средних значений информативности признака"""
        average_sign = 0
        for values in range(len(array)):
            average_sign += array[values][ind]

        return round(average_sign / len(array), 4)

    @staticmethod
    def informative_standard_deviation(m: float,
                                       ind: int,
                                       array: list,
                                       ) -> float:
        """Расчет среднеквадратического отклонения"""
        counter_results = 0

        for values in range(len(array)):
            counter_results += (array[values][ind] - m) ** 2

        return round(counter_results / len(array), 4)

    def informative_signs(self,
                          y_class: list[int],
                          not_class: list[int]
                          ) -> list:
        """Функция расчета информативности признаков по методу Фишера"""
        average_sign_y: list[float] = []
        average_sign_not_class: list[float] = []
        deviation_sign_y: list[float] = []
        deviation_sign_not_class: list[float] = []
        info_signs: list[float] = []

        for sign in range(_QUANTITY_SIGNS):
            average_sign_y.append(self.informative_average_sign(sign, y_class))
            average_sign_not_class.append(self.informative_average_sign(sign, not_class))

        for deviation_sign in range(_QUANTITY_SIGNS):
            deviation_sign_y.append(
                self.informative_standard_deviation(average_sign_y[deviation_sign], deviation_sign, y_class))
            deviation_sign_not_class.append(
                self.informative_standard_deviation(average_sign_not_class[deviation_sign], deviation_sign, not_class))

        for info_sign in range(_QUANTITY_SIGNS):
            a = (average_sign_y[info_sign] - average_sign_not_class[info_sign]) ** 2
            b = deviation_sign_y[info_sign] + deviation_sign_not_class[info_sign]
            info_signs.append(round(a / b, 4))

        return info_signs
