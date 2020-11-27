import math

import cv2
import numpy as np

TEMP = "tmp/"


def saveFile(file, fileName):
    """
    :param file: file to save
    :param fileName: name of file
    """
    cv2.imwrite(TEMP + fileName, file)


def saveFileFinal(file, fileName):
    """
    :param file: file to save
    :param fileName: name of file
    Save file to main directory
    """
    cv2.imwrite(fileName, file)


def canny(img, low_threshold, high_threshold):
    """
    :param img: image to analyze
    :param low_threshold: low threshold
    :param high_threshold: high threshold
    :return: image with canny algorithm used
    """
    return cv2.Canny(img, low_threshold, high_threshold)


def get_hough_lines(img, rho=1, theta=np.pi / 180, threshold=90, min_line_len=130, max_line_gap=6):
    """
    :param img: image to analyze
    :param rho: rho
    :param theta: theta
    :param threshold: threshold
    :param min_line_len: minimal length of line
    :param max_line_gap: maximum gap between lines
    :return: hough lines
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines


def draw_lines(img, lines, height, color=None, thickness=2):
    """
    :param img: image to draw on
    :param lines: lines to draw
    :param height: height of image
    :param color: color of lines
    :param thickness: thickness of lines
    :return: new image
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
                if (abs(midG - midC) < 90) and \
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
