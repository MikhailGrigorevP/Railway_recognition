import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

SOURCE = "source/"
TEMP = "tmp/"
IMG_NUM = "3"
IMG_TYPE = ".jpg"


def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


def get_hough_lines(img, rho=1, theta=np.pi / 180, threshold=20, min_line_len=20, max_line_gap=300):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines


def draw_lines(img, lines, color=None, thickness=2):
    if color is None:
        color = [255, 0, 0]
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    return img


def get_aoi(img):
    rows, cols = img.shape[:2]
    mask = np.zeros_like(img)

    left_bottom = [cols * 0.1, rows]
    right_bottom = [cols * 0.95, rows]
    left_top = [cols * 0.4, rows * 0.6]
    right_top = [cols * 0.6, rows * 0.6]

    vertices = np.array([[left_bottom, left_top, right_top, right_bottom]], dtype=np.int32)

    if len(mask.shape) == 2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,) * mask.shape[2])
    return cv2.bitwise_and(img, mask)


def saveFile(file, fileName):
    cv2.imwrite(TEMP + fileName, file)


def toHLS(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)


def toHSV(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)


def grayScale(image):
    """
    :param image: Исходное изображение (в формате BGR)
    :return: Изображение, преобразованное из формата BGR в оттенки серого
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    saveFile(gray, IMG_NUM + '_gray.bmp')
    return gray


def darken(image, gamma=1.0):
    """
    :param image: Исходное изображение (в формате GRAY)
    :param gamma: V_out = A * V_in^gamma
    :return: чёрно-белое изображение с гамма-коррекцией
    """
    table = np.array([((i / 255.0) ** gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    darkened = cv2.LUT(image, table)
    saveFile(darkened, IMG_NUM + '_darkened.bmp')
    return darkened


def colorMask(img, low_thresh, high_thresh):
    """
    :param img: Исходное изображение
    :param low_thresh: Низкий порог
    :param high_thresh: Высокий порог
    :return: Наложить на кадр цветовой фильтр в заданном диапазоне
    """
    assert (0 <= low_thresh.all() <= 255)
    assert (0 <= high_thresh.all() <= 255)
    return cv2.inRange(img, low_thresh, high_thresh)


def colorSelection(image, darkened, desc):
    """
    :param desc: Описание изображения маски
    :param image: Исходное изображение
    :param darkened: Затемнённое изображение
    :return: Изображение под маской
    """
    white_masks = \
        colorMask(image, np.array([150, 150, 150], dtype=np.uint8), np.array([255, 255, 255], dtype=np.uint8))
    # mask = cv2.bitwise_or(white_masks, white_masks)
    mask = white_masks
    masked = cv2.bitwise_and(darkened, darkened, mask=mask)
    saveFile(masked, IMG_NUM + '_masked_' + desc + '.bmp')
    return masked


def gaussianBlur(image, kernelSize=11):
    """
    :param kernelSize: Размер ядра
    :param image: Исходное изображение
    :return: Изображение с размытие по Гауссу
    """
    blurred = cv2.GaussianBlur(image, (kernelSize, kernelSize), 0)
    saveFile(blurred, IMG_NUM + '_blurred.bmp')
    return blurred


def cannyDetector(image):
    """
    :param image: Исходное изображение
    :return: Применённый детектор Кэнни
    """
    detectedEdges = canny(image, low_threshold=70, high_threshold=140)
    saveFile(detectedEdges, IMG_NUM + '_canny.bmp')
    return detectedEdges


def areaOfInterest(image):
    """
    :param image: Исходное изображение
    :return:
    """
    aoi = get_aoi(image)
    saveFile(aoi, IMG_NUM + '_AOI.bmp')
    return aoi


def houghTransformLineDetection(origImage, image):
    """
    :param image: Исходное изображение
    :return:
    """
    line = get_hough_lines(image)
    hough = draw_lines(origImage, line)
    saveFile(hough, IMG_NUM + '_hought.bmp')
    return hough


def showImg(images):
    """
    :param images: Кортеж из изображения и его описания
    :return: Вывод изображений. Установка ожидания отклика
    """
    for img, text in images:
        cv2.imshow(text, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(fileName):
    image = cv2.imread(fileName)
    imageHLS = toHLS(image)
    gray = grayScale(image)
    darkened = darken(gray, 1.5)
    masked = colorSelection(image, darkened, "RGB")
    masked_hls = colorSelection(imageHLS, darkened, "HLS")
    blurred = gaussianBlur(masked)
    detected = cannyDetector(blurred)
    aoi = areaOfInterest(detected)
    hough = houghTransformLineDetection(image, aoi)

    # Show images

    images = [(image, 'Original image'), (imageHLS, 'HLS image'), (gray, 'Gray image'), (darkened, 'Darkened image'),
              (masked, 'Masked RGB image'), (masked, 'Masked HLS image'), (blurred, 'Blurred image'),
              (detected, 'Canny detected image'), (aoi, 'Area detected image'), (hough, 'Hough image')]
    showImg(images)
    # hls = toHLS(darkened)


if __name__ == '__main__':
    main(SOURCE + IMG_NUM + IMG_TYPE)
