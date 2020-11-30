import math
import cv2
import numpy as np

TEMP = "tmp/"


def saveFile(file, fileName):
    """
    Сохранение временных файлов
    :param file: Файл для сохранение
    :param fileName: Имя файла
    """
    cv2.imwrite(TEMP + fileName, file)


def saveFileFinal(file, fileName):
    """
    Сохранение файла в основную директорию
    :param file: Файл для сохранение
    :param fileName: Имя файла
    """
    cv2.imwrite(fileName, file)


def canny(img, low_threshold, high_threshold):
    """
    Детектор Кэнни
    :param img: Изображение для обработки
    :param low_threshold: Нижний порог
    :param high_threshold: Высокий порог
    :return: Изображение
    """
    return cv2.Canny(img, low_threshold, high_threshold)


def get_hough_lines(img, rho=1, theta=np.pi / 180, threshold=90, min_line_len=130, max_line_gap=6):
    """
    Получить линии Хафа
    :param img: Изображение для обработки
    :param rho: rho
    :param theta: theta
    :param threshold: Порог
    :param min_line_len: Минимальная длина линий
    :param max_line_gap: Максимальное расстояние между линиями
    :return: Линии Хафа
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines


def draw_lines(img, lines, height, color=None, thickness=2):
    """
    Отрисовка линий на изображении
    :param img: изображение, на котором нужно рисовать
    :param lines: линии для отрисовки
    :param height: высота изображения
    :param color: цвет линии
    :param thickness: толщина линий
    :return: Новое изображение
    """
    # print(height)
    if color is None:
        color = [0, 0, 255]
        color2 = [0, 255, 0]
    vertical_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            vertical = (math.sqrt(abs(y2 ** 2 - y1 ** 2)) > math.sqrt(abs(x2 ** 2 - x1 ** 2))) > 0
            # print(vertical, x1, y1, x2, y2)
            if vertical:
                vertical_lines.append(line)
            # Отрисовка образующих
            # cv2.line(img, (x1, y1), (x2, y2), color2, thickness)
    min_y = height

    xCoordGroup = []
    delList = []
    for i in range(len(vertical_lines)):
        if not xCoordGroup:
            xCoordGroup.append(i)
        else:
            changed = False
            for a in range(len(xCoordGroup)):
                group = xCoordGroup[a]
                midC = (vertical_lines[i][0][2] + vertical_lines[i][0][0]) / 2
                midG = (vertical_lines[group][0][2] + vertical_lines[group][0][0]) / 2
                if (abs(midG - midC) < 100) and \
                        ((vertical_lines[i][0][1] - vertical_lines[i][0][3])
                         * (vertical_lines[group][0][1] - vertical_lines[group][0][3]) > 0):
                    minY2 = min(vertical_lines[i][0][3], vertical_lines[group][0][3])
                    minY1 = min(vertical_lines[i][0][1], vertical_lines[group][0][1])
                    vertical_lines[group][0][0] = x1 = (vertical_lines[i][0][0] + vertical_lines[group][0][0]) / 2
                    vertical_lines[group][0][1] = y1 = (vertical_lines[i][0][1] + vertical_lines[group][0][1]) / 2
                    vertical_lines[group][0][2] = x2 = (vertical_lines[i][0][2] + vertical_lines[group][0][2]) / 2
                    vertical_lines[group][0][3] = y2 = (vertical_lines[i][0][3] + vertical_lines[group][0][3]) / 2
                    if x2 != x1:
                        if y1 < y2:
                            k = (y2 - y1) / (x2 - x1)
                            c = y1 - k * x1
                            vertical_lines[group][0][1] = minY1
                            vertical_lines[group][0][0] = (minY1 - c) / k
                            delList.append(i)
                            changed = True
                            break
                        else:
                            k = (y2 - y1) / (x2 - x1)
                            c = y1 - k * x1
                            vertical_lines[group][0][3] = minY2
                            vertical_lines[group][0][2] = (minY2 - c) / k
                            delList.append(i)
                            changed = True
                            break
                    else:
                        if y1 < y2:
                            vertical_lines[group][0][3] = minY1
                            delList.append(i)
                            changed = True
                            break
                        else:
                            vertical_lines[group][0][1] = minY2
                            delList.append(i)
                            changed = True
                            break
            if not changed:
                xCoordGroup.append(i)

    i = 0
    for toDel in delList:
        vertical_lines.pop(toDel - i)
        i += 1

    for line in vertical_lines:
        for x1, y1, x2, y2 in line:
            if y2 < min_y:
                min_y = y2
            if y1 < min_y:
                min_y = y1

    mode = False
    if len(vertical_lines) == 1:
        mode = True
    else:
        for line in vertical_lines:
            for x1, y1, x2, y2 in line:
                if x2 != x1:
                    k = (y2 - y1) / (x2 - x1)
                    c = y1 - k * x1
                    if y2 > y1:
                        y2 = int(height)
                        x2 = int((y2 - c) / k)
                        y1 = int(min_y)
                        x1 = int((y1 - c) / k)
                    else:
                        y1 = int(height)
                        x1 = int((y1 - c) / k)
                        y2 = int(min_y)
                        x2 = int((y2 - c) / k)
                else:
                    if y2 > y1:
                        y2 = int(height)
                        y1 = int(min_y)
                    else:
                        y1 = int(height)
                        y2 = int(min_y)
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    return img, mode


def get_aoi(img):
    """
    Получение область обработки изображения
    :param img: image to analyze
    :return: image with deleted uninteresting area
    """
    rows, cols = img.shape[:2]
    mask = np.zeros_like(img)

    left_bottom = [cols * 0.1, rows]
    right_bottom = [cols * 0.95, rows]
    left_top = [cols * 0.45, rows * 0.4]
    right_top = [cols * 0.55, rows * 0.4]

    vertices = np.array([[left_bottom, left_top, right_top, right_bottom]], dtype=np.int32)

    if len(mask.shape) == 2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,) * mask.shape[2])
    return cv2.bitwise_and(img, mask)
