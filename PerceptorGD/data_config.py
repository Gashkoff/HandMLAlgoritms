SURNAMES: dict[int:str] = {
    0: "Мельник",
    1: "Нартов",
    2: "Сафронов",
    3: "Саша",
    4: "Скобкин",
    5: "Стрельников",
    6: "Андрей",
    7: "Валерия",
    8: "Егор",
    9: "Ксения",
    10: "Любасова",
    11: "Мария",
    12: "Скорский",
    13: "Фачкин",
    14: "Циганаш",
}

DATASET: dict[int:list[list[int]]] = {
    0: [
        [71, 29, 71, 29, 73, 94, 105, 70, 82, 84, 70, 108, 97, 74, 42, 41, 39, 31, 33, 39],
        [66, 30, 57, 21, 76, 104, 104, 70, 87, 79, 68, 93, 83, 68, 50, 28, 38, 41, 18, 38],
        [65, 26, 69, 30, 72, 85, 98, 70, 78, 84, 70, 107, 100, 77, 34, 44, 35, 21, 34, 35],
    ],
    1: [
        [70, 27, 70, 28, 80, 103, 110, 77, 92, 91, 77, 108, 101, 79, 38, 40, 39, 32, 31, 39],
        [63, 28, 56, 22, 78, 107, 106, 74, 93, 87, 72, 95, 90, 72, 46, 28, 39, 40, 23, 39],
        [58, 22, 64, 28, 69, 87, 96, 68, 81, 84, 68, 99, 97, 70, 28, 45, 36, 23, 38, 36],
    ],
    2: [
        [72, 30, 71, 29, 73, 94, 108, 73, 82, 82, 71, 108, 92, 71, 43, 42, 36, 34, 33, 36],
        [68, 31, 61, 24, 81, 108, 110, 76, 92, 83, 75, 98, 85, 74, 52, 29, 36, 41, 15, 36],
        [58, 22, 67, 31, 76, 88, 99, 73, 83, 89, 74, 104, 105, 79, 27, 49, 37, 15, 39, 37],
    ],
    3: [
        [74, 30, 74, 30, 81, 101, 111, 76, 88, 87, 75, 109, 99, 78, 37, 40, 36, 29, 30, 36],
        [67, 31, 58, 22, 82, 107, 104, 75, 90, 82, 71, 96, 86, 73, 51, 30, 40, 37, 14, 40],
        [65, 25, 70, 31, 76, 91, 100, 74, 85, 91, 75, 108, 107, 82, 30, 46, 39, 18, 35, 39],
    ],
    4: [
        [68, 28, 68, 27, 72, 93, 102, 68, 81, 79, 66, 100, 90, 69, 35, 36, 34, 30, 28, 34],
        [62, 28, 55, 21, 73, 98, 99, 70, 84, 77, 67, 91, 79, 66, 47, 26, 35, 38, 17, 35],
        [62, 23, 67, 28, 68, 80, 93, 67, 75, 82, 69, 100, 97, 73, 26, 47, 33, 15, 36, 33],
    ],
    5: [
        [83, 34, 83, 35, 88, 114, 129, 87, 102, 102, 88, 128, 115, 89, 49, 51, 47, 38, 40, 47],
        [79, 36, 70, 27, 92, 121, 126, 86, 102, 90, 85, 107, 91, 82, 58, 41, 44, 45, 15, 44],
        [68, 26, 76, 35, 81, 96, 110, 81, 92, 102, 85, 121, 120, 90, 36, 57, 45, 19, 45, 45],
    ],
    6: [
        [72, 29, 72, 29, 79, 103, 111, 75, 92, 92, 77, 112, 106, 82, 37, 38, 37, 32, 33, 37],
        [70, 32, 63, 25, 83, 110, 113, 77, 93, 88, 77, 107, 97, 81, 49, 27, 36, 40, 21, 36],
        [10, 4, 10, 4, 12, 13, 15, 12, 12, 12, 11, 15, 14, 12, 5, 5, 4, 3, 3, 4],
    ],
    7: [
        [72, 29, 71, 29, 75, 95, 107, 71, 82, 81, 69, 107, 94, 73, 37, 39, 33, 31, 32, 33],
        [61, 28, 54, 21, 74, 103, 102, 68, 88, 80, 68, 88, 84, 68, 45, 27, 38, 38, 20, 38],
        [59, 23, 66, 29, 68, 83, 96, 67, 76, 82, 67, 103, 97, 71, 28, 47, 32, 21, 39, 32],
    ],
    8: [
        [69, 26, 69, 26, 71, 90, 104, 69, 79, 79, 68, 103, 89, 69, 40, 41, 34, 31, 32, 34],
        [59, 26, 52, 19, 75, 106, 99, 72, 93, 87, 68, 92, 89, 67, 49, 26, 43, 43, 23, 43],
        [62, 23, 65, 27, 69, 85, 98, 68, 78, 84, 69, 102, 97, 71, 29, 49, 35, 22, 40, 35],
    ],
    9: [
        [70, 28, 69, 27, 78, 102, 108, 73, 89, 87, 72, 106, 99, 77, 34, 34, 33, 32, 30, 33],
        [61, 29, 53, 20, 79, 104, 101, 72, 87, 81, 69, 93, 86, 73, 44, 22, 33, 37, 15, 33],
        [64, 24, 69, 28, 76, 94, 103, 72, 85, 87, 71, 105, 101, 76, 28, 42, 33, 24, 35, 33],
    ],
    10: [
        [65, 27, 64, 26, 66, 85, 97, 61, 72, 70, 62, 91, 81, 65, 34, 33, 29, 27, 26, 29],
        [57, 27, 52, 22, 66, 86, 92, 59, 70, 63, 60, 80, 69, 61, 38, 26, 28, 30, 13, 28],
        [48, 20, 45, 18, 56, 69, 71, 49, 57, 55, 49, 67, 62, 54, 28, 23, 26, 19, 11, 26],
    ],
    11: [
        [55, 23, 53, 22, 66, 81, 88, 63, 72, 71, 62, 86, 80, 64, 28, 29, 26, 23, 24, 26],
        [53, 25, 47, 19, 69, 87, 88, 63, 74, 68, 61, 81, 73, 64, 37, 23, 29, 28, 12, 29],
        [54, 22, 56, 24, 67, 80, 87, 64, 72, 75, 84, 90, 80, 68, 25, 33, 27, 19, 27, 27],
    ],
    12: [
        [75, 30, 74, 30, 81, 98, 113, 82, 88, 90, 82, 116, 100, 83, 46, 45, 39, 31, 30, 29],
        [9, 3, 10, 4, 9, 11, 14, 9, 10, 10, 9, 14, 12, 9, 4, 5, 3, 3, 4, 3],
        [9, 3, 9, 3, 9, 11, 12, 9, 9, 9, 9, 13, 11, 9, 4, 4, 3, 3, 3, 3],
    ],
    13: [
        [74, 30, 73, 29, 75, 97, 112, 73, 85, 84, 75, 109, 94, 76, 42, 42, 35, 34, 32, 35],
        [66, 31, 60, 24, 79, 104, 108, 75, 88, 80, 73, 97, 83, 71, 49, 32, 36, 40, 17, 36],
        [68, 25, 73, 30, 75, 87, 103, 75, 81, 87, 77, 112, 102, 81, 34, 47, 34, 21, 36, 34],
    ],
    14: [
        [82, 35, 80, 34, 85, 109, 128, 82, 94, 92, 83, 122, 104, 83, 46, 48, 38, 38, 37, 38],
        [20, 8, 21, 9, 21, 28, 31, 23, 27, 29, 23, 35, 33, 24, 13, 14, 14, 9, 13, 14],
        [80, 33, 86, 38, 96, 112, 128, 91, 100, 106, 94, 133, 126, 101, 38, 56, 42, 24, 42, 42],
    ],
}