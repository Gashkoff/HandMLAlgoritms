import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.cm as cm
from matplotlib.ticker import MultipleLocator

class StateValues:
    """
    Класс для расчета и построения методов кластеризации методами:
        K-means
        Hierarchical clustering
    """
    uncertainty_functions = {
        "B-ACD": {},
        "D-AC": {},
        "A-C": {},
    }

    def __init__(self):
        self.classes_learn_dataset = {
            0: [2, 26],
            1: [3, 22],
            2: [4, 15],
            3: [5, 17],
            4: [6, 27],
            5: [8, 23],
            6: [9, 13],
            7: [10, 18],
            8: [10, 29],
            9: [11, 25],
            10: [13, 14],
            11: [14, 21],
            12: [15, 28],
            13: [16, 17],
            14: [17, 24],
            15: [24, 12],
            16: [25, 6],
            17: [26, 9],
            18: [27, 4],
            19: [27, 14],
            20: [29, 12],
            21: [30, 8],
            22: [30, 15],
            23: [32, 2],
            24: [32, 11],
            25: [33, 6],
            26: [34, 13],
            27: [35, 3],
            28: [36, 10],
            29: [37, 6],
            30: [20, 44],
            31: [21, 48],
            32: [22, 40],
            33: [23, 45],
            34: [23, 51],
            35: [25, 53],
            36: [26, 39],
            37: [26, 44],
            38: [26, 50],
            39: [27, 36],
            40: [29, 46],
            41: [30, 41],
            42: [30, 52],
            43: [32, 43],
            44: [32, 48],
            45: [7, 35],
            46: [4, 36],
            47: [7, 37],
            48: [10, 37],
            49: [6, 39],
            50: [9, 39],
            51: [11, 40],
            52: [5, 41],
            53: [7, 42],
            54: [13, 43],
            55: [10, 44],
            56: [8, 45],
            57: [12, 46],
            58: [9, 48],
            59: [11, 50],
        }

        self.K = 4

    @staticmethod
    def euclidean_distance(point1, point2):
        """Метод расчета Евклидова расстояния"""
        return np.sqrt(sum(pow(a - b, 2) for a, b in zip(point1, point2)))

    @staticmethod
    def additional_counting_avg(array: list):
        counter = 0

        for i in array:
            counter += i

        return counter / len(array)

    def average_sign(self,
            first_points: list,
            second_points: list,
    ):
        """Расчет среднего значения признака"""

        avg_x_1 = self.additional_counting_avg([self.classes_learn_dataset[i][0] for i in first_points])
        avg_x_2 = self.additional_counting_avg([self.classes_learn_dataset[i][0] for i in second_points])

        avg_y_1 = self.additional_counting_avg([self.classes_learn_dataset[i][1] for i in first_points])
        avg_y_2 = self.additional_counting_avg([self.classes_learn_dataset[i][1] for i in second_points])

        return (avg_x_1, avg_y_1), (avg_x_2, avg_y_2)

    def hierarchical_clusterization(self):
        """Метод иерархической кластеризации"""
        main_classes: list[list[int]] = [[i] for i in range(len(self.classes_learn_dataset))]
        class_distance: list = []
        self.cluster_plot(main_classes)
        while len(main_classes) > 1:
            point_one: int = 0
            point_two: int = 0
            min_len: int = 1000
            for class_1, elem_1 in enumerate(main_classes):

                for class_2, elem_2 in enumerate(main_classes):
                    if elem_1 != elem_2:

                        elem_point_1, elem_point_2 = self.average_sign(elem_1, elem_2)
                        way = self.euclidean_distance((elem_point_1[0], elem_point_1[1]), (elem_point_2[0], elem_point_2[1]))
                        if way < min_len:
                            min_len = way
                            point_one = class_1
                            point_two = class_2

            class_distance.append(min_len)
            main_classes[point_one] += main_classes[point_two]
            del main_classes[point_two]

        self.cluster_plot(main_classes)
        self.thorndyke_plot(class_distance)

    @staticmethod
    def thorndyke_plot(values):
        """Метод построения кривой Торндайка"""
        retard_length = [i for i in range(1, len(values) + 1)]
        b = retard_length[::-1]
        class_a_x = []
        class_a_y = []
        for idx, elen in enumerate(values):
            class_a_x.append(b[idx])
            class_a_y.append(elen)
            plt.plot(class_a_x, class_a_y, color='blue')

        plt.title("Кривая Торндайка")
        plt.ylabel("Расстояние между классами")
        plt.xlabel("Количество классов")

        plt.gca().xaxis.set_major_locator(MultipleLocator(5))
        plt.gca().invert_xaxis()

        plt.grid()
        plt.legend()
        plt.show()

    def cluster_plot(self, points):
        """Построение исходных значений иерархической кластеризации"""
        colors = cm.get_cmap("tab20", 20)

        for idx, elem in enumerate(points):
            class_a_x = [self.classes_learn_dataset[i][0] for i in elem]
            class_a_y = [self.classes_learn_dataset[i][1] for i in elem]

            color = colors(idx % 20) if len(points) <= 20 else colors(random.randint(0, 19))

            plt.scatter(class_a_x, class_a_y, color=color)

        plt.title("График классификации")
        plt.ylabel("Признак 6")
        plt.xlabel("Признак 3")
        plt.legend()
        plt.show()

    def create_clear_plot(self):
        class_a_x = [i[2] for i in self.classes_learn_dataset]
        class_a_y = [i[5] for i in self.classes_learn_dataset]
        plt.scatter(class_a_x, class_a_y, color='blue', label="alabel")

        plt.title("ds")
        plt.ylabel("ylabel")
        plt.xlabel("xlabel")
        plt.legend()
        plt.show()

    def k_means(self):
        """Кластеризация методом K-means"""
        clusters = self.random_points(*self.max_coord())
        classes = [[], [], [], []]
        i = 0

        while not all(len(class_list) == 15 for class_list in classes):
            for point in self.classes_learn_dataset.values():
                min_length = 1000
                cluster_for_point = 0

                for cluster_idx, cluster in enumerate(clusters):
                    way = self.euclidean_distance(point, cluster)
                    if way < min_length:
                        min_length = way
                        cluster_for_point = cluster_idx

                classes[cluster_for_point].append(point)

            self.k_means_plot(classes, clusters, i)
            i += 1
            clusters = self.upgrade_centers(classes)
            classes = [[], [], [], []]

            if i > 5:
                break

    @staticmethod
    def average_point(points):
        return sum(points) / len(points)


    def upgrade_centers(self, classes):
        """Метод обновления центров кластеров в K-means"""
        point_1 = (self.average_point([point[0] for point in classes[0]]), self.average_point([point[1] for point in classes[0]]))
        point_2 = (self.average_point([point[0] for point in classes[1]]), self.average_point([point[1] for point in classes[1]]))
        point_3 = (self.average_point([point[0] for point in classes[2]]), self.average_point([point[1] for point in classes[2]]))
        point_4 = (self.average_point([point[0] for point in classes[3]]), self.average_point([point[1] for point in classes[3]]))

        return point_1, point_2, point_3, point_4


    @staticmethod
    def k_means_plot(classes, center_clusters, iteration):
        """Построение графика K-means"""
        class_a_x = [i[0] for i in classes[0]]
        class_a_y = [i[1] for i in classes[0]]

        class_b_x = [i[0] for i in classes[1]]
        class_b_y = [i[1] for i in classes[1]]

        class_c_x = [i[0] for i in classes[2]]
        class_c_y = [i[1] for i in classes[2]]

        class_d_x = [i[0] for i in classes[3]]
        class_d_y = [i[1] for i in classes[3]]

        plt.scatter(class_a_x, class_a_y, color='blue', label="Кластер A")
        plt.scatter(class_b_x, class_b_y, color='cyan', label="Кластер B")
        plt.scatter(class_c_x, class_c_y, color='orange', label="Кластер C")
        plt.scatter(class_d_x, class_d_y, color='red', label="Кластер D")

        plt.scatter(center_clusters[0][0], center_clusters[0][1], color='black', marker='x', s=100, label="Центры кластеров")
        plt.scatter(center_clusters[1][0], center_clusters[1][1], color='black', marker='x', s=100)
        plt.scatter(center_clusters[2][0], center_clusters[2][1], color='black', marker='x', s=100)
        plt.scatter(center_clusters[3][0], center_clusters[3][1], color='black', marker='x', s=100)

        plt.title(f"Итерация {iteration}")
        plt.ylabel("Признак 6")
        plt.xlabel("Признак 3")
        plt.legend()
        plt.show()

    def max_coord(self):
        max_x = max([i[0] for i in self.classes_learn_dataset.values()])
        max_y = max([i[1] for i in self.classes_learn_dataset.values()])

        return max_x, max_y

    def random_points(self, max_x, max_y):
        return [(random.randint(10, max_x), random.randint(10, max_y)) for _ in range(self.K)]


def main():
    a = StateValues()
    a.hierarchical_clusterization()
    a.k_means()

if __name__ == '__main__':
    main()
