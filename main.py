# import libraries
# for reading files
import os
# for timing
import time
# to analyze type
import filetype
# for graphic interface
import sys
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QListWidgetItem
import design

from additionalFunction import *

# Для компиляции файла из .ui в .py
# pyuic5 main.ui -o main.py

# Компиляция .exe файла
# pip install pyinstaller
# pyinstaller main.py --onefile --noconsole [-exclude названиеБиблиотеки]

# Константы
SOURCE = "source/"
TEMP = "tmp/"
IMG_NUM = "3"
IMG_TYPE = ".jpg"

os.makedirs("tmp/", exist_ok=True)


class railWayAnalyzer:

    @staticmethod
    def toContrast(image, contrast=127):
        """
        Увеличение контраста изображения с высоким порогом
        :param image: image
        :param contrast: contrast
        :return: image with adjusted contrast (for bright images)
        """
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)
        saveFile(buf, IMG_NUM + '_c.bmp')

        ret, threshold_image = cv2.threshold(image, 170, 255, cv2.THRESH_BINARY)
        saveFile(threshold_image, IMG_NUM + '_t.bmp')

        return threshold_image

    @staticmethod
    def adjustContrast(image, contrast=127):
        """
        Увеличение контраста изображения
        :param image: image
        :param contrast: contrast
        :return: image with adjusted contrast
        """
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        buf = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)

        saveFile(buf, IMG_NUM + '_contrast.bmp')

        return buf

    @staticmethod
    def toThreshold(image, spec):
        """
        Установление порога
        :param image: image
        :param spec: image has lots of white
        :return: binary image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if spec:
            ret, threshold_image = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
        else:
            ret, threshold_image = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
        saveFile(threshold_image, IMG_NUM + '_bin.bmp')

        return threshold_image

    @staticmethod
    def grayScale(image):
        """
        Перевод изображения в оттенки серого
        :param image: Исходное изображение (в формате BGR)
        :return: Изображение, преобразованное из формата BGR в оттенки серого
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        saveFile(gray, IMG_NUM + '_gray.bmp')
        return gray

    @staticmethod
    def colorMask(img, low_thresh, high_thresh):
        """
        Цветовой фильтр
        :param img: Исходное изображение
        :param low_thresh: Низкий порог
        :param high_thresh: Высокий порог
        :return: Наложить на кадр цветовой фильтр в заданном диапазоне
        """
        assert (0 <= low_thresh.all() <= 255)
        assert (0 <= high_thresh.all() <= 255)
        return cv2.inRange(img, low_thresh, high_thresh)

    @staticmethod
    def gaussianBlur(image, kernelSize=11):
        """
        Применение размытия по Гауссу
        :param kernelSize: Размер ядра
        :param image: Исходное изображение
        :return: Изображение с размытие по Гауссу
        """
        blurred = cv2.GaussianBlur(image, (kernelSize, kernelSize), 0)
        saveFile(blurred, IMG_NUM + '_blurred.bmp')
        return blurred

    @staticmethod
    def cannyDetector(image, low=150, high=200):
        """
        Детектор Кэнни
        :param high: Высокий порог
        :param low: Нижний порог
        :param image: Исходное изображение
        :return: Применённый детектор Кэнни
        """
        detectedEdges = canny(image, low_threshold=low, high_threshold=high)
        saveFile(detectedEdges, IMG_NUM + '_canny.bmp')

        return detectedEdges

    @staticmethod
    def useSobel(image, kernel=3):
        """
        Ядро собеля
        :param kernel: Ядро матрицы
        :param image: Исходное изображение
        :return: Применённый детектор Кэнни
        """

        sobel = cv2.Sobel(image, cv2.CV_8U, 1, 0, ksize=kernel)
        saveFile(sobel, IMG_NUM + '_sobel.bmp')

        abs_sobel64f = np.absolute(sobel)
        sobel_x = np.uint8(abs_sobel64f)

        return sobel_x

    @staticmethod
    def areaOfInterest(image):
        """
        Вычисление зоны обработки
        :param image: Исходное изображение
        :return: Зона обработки
        """
        aoi = get_aoi(image)
        saveFile(aoi, IMG_NUM + '_AOI.bmp')
        return aoi

    @staticmethod
    def houghTransformLineDetection(origImage, image, mode=False):
        """
        Трансформация Хафа
        :param mode:
        :param origImage:
        :param image: Исходное изображение
        :return:
        """
        line = get_hough_lines(image)
        hough, mode2 = draw_lines(origImage, line, height=image.shape[0])

        # if mode:
        #     line = get_hough_lines(image, rho=1, theta=np.pi/180, threshold=80, min_line_len=150, max_line_gap=9)
        #     hough, mode = draw_lines(origImage, line, height=image.shape[0])
        saveFile(hough, IMG_NUM + '_hought.bmp')
        return hough, mode2

    @staticmethod
    def whitePercent(image):
        """
        Вычисление процента белого цвета
        :param image: image
        :return: calculate percent of white pixels
        """
        f = 131 * (127 + 127) / (127 * (131 - 127))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        test = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)
        test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
        ret, test = cv2.threshold(test, 220, 255, cv2.THRESH_BINARY)
        nonzero = cv2.countNonZero(test)
        imgSize = test.shape[0] * test.shape[1]
        return nonzero / float(imgSize) * 100

    def __init__(self, fileName, window):
        self.start = time.time()

        # Чтение изображения
        image = cv2.imread(fileName)
        window.addListWidgetButton(QListWidgetItem('Original'))

        # Выделение имени изображения
        name = fileName.split("/")
        name = name[len(name)-1].split(".")
        self.name = name[0]

        # Количество белого
        res = self.whitePercent(image)

        spec = False
        if res > 60.0:
            spec = True
            self.name = "winter/" + self.name
        else:
            self.name = "classic/" + self.name

        try_to_analyze = True
        mode = False

        # Обработка изображения
        while try_to_analyze:

            if mode:
                custom = image.copy()

                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            custom[np.all(custom == (i, j, k), axis=-1)] = (250, 250, 250)

                contrast = self.adjustContrast(custom, 120)
                window.addListWidgetButton(QListWidgetItem('Contrast'))
            else:
                contrast = self.adjustContrast(image, 120)
                window.addListWidgetButton(QListWidgetItem('Contrast'))

            threshold = self.toThreshold(contrast, spec)
            window.addListWidgetButton(QListWidgetItem('Thresholded'))

            blurred = self.gaussianBlur(threshold)
            window.addListWidgetButton(QListWidgetItem('Blurred'))

            sobel = self.useSobel(blurred)
            window.addListWidgetButton(QListWidgetItem('Sobel'))

            detected = self.cannyDetector(sobel)
            window.addListWidgetButton(QListWidgetItem('Canny'))

            aoi = self.areaOfInterest(detected)
            window.addListWidgetButton(QListWidgetItem('AOI'))

            hough, modeF = self.houghTransformLineDetection(image, aoi, mode)
            window.addListWidgetButton(QListWidgetItem('Hought'))

            saveFileFinal(hough, self.name+".jpeg")

            if modeF and not mode:
                mode = True
                try_to_analyze = True
            else:
                try_to_analyze = False
                image = cv2.imread("tmp/3_hought.bmp")
                window.drawImage(image)

        self.end = time.time()
        print("Ended in: ", self.end - self.start)


class railWayWindow(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        # Доступ к переменным, методам и т.д. в файле design.py
        super().__init__()
        # Инициализация дизайна
        self.setupUi(self)
        # Связь нажатия кнопки
        self.fileButton.clicked.connect(self.openFile)
        self.listWidget.itemClicked.connect(self.setImage)

    def setImage(self, item):
        """
        Установка изображения на экран приложения
        :param item: нажатый элемент
        """
        image = cv2.imread(TEMP + IMG_NUM + "_canny.bmp")
        text = str(item.text())
        if text == "Original":
            image = self.image
        elif text == "HLS":
            image = cv2.imread(TEMP + IMG_NUM + "_hls.bmp")
        elif text == "Gray":
            image = cv2.imread(TEMP + IMG_NUM + "_gray.bmp")
        elif text == "Thresholded":
            image = cv2.imread(TEMP + IMG_NUM + "_bin.bmp")
        elif text == "Contrast":
            image = cv2.imread(TEMP + IMG_NUM + "_contrast.bmp")
        elif text == "Masked":
            image = cv2.imread(TEMP + IMG_NUM + "_masked_RGB.bmp")
        elif text == "Blurred":
            image = cv2.imread(TEMP + IMG_NUM + "_blurred.bmp")
        elif text == "Canny":
            image = cv2.imread(TEMP + IMG_NUM + "_canny.bmp")
        elif text == "AOI":
            image = cv2.imread(TEMP + IMG_NUM + "_AOI.bmp")
        elif text == "Hought":
            image = cv2.imread(TEMP + IMG_NUM + "_hought.bmp")
        elif text == "Sobel":
            image = cv2.imread(TEMP + IMG_NUM + "_sobel.bmp")
        else:
            image = self.image

        self.drawImage(image)
        self.label.update()
        pass

    def drawImage(self, image):
        """
        Рисование изображения
        :param image: Исходное изображение
        """
        self.mw.setFixedSize(image.shape[1], image.shape[0])
        image = QtGui.QImage(image.data,
                             image.shape[1],
                             image.shape[0],
                             QtGui.QImage.Format_RGB888).rgbSwapped()
        pm = QtGui.QPixmap.fromImage(image)
        self.label.setPixmap(pm)

    def openFile(self):
        """
        Открытие файла
        """
        # На случай, если в списке уже есть элементы
        self.listWidget.clear()
        fileName = QtWidgets.QFileDialog.getOpenFileName(self, "Выберите файл")[0]
        if fileName:
            self.image = cv2.imread(fileName)
            if filetype.is_image(fileName):
                self.listWidget.addItem(QListWidgetItem('Файл успешно прочитан как изображение'))
                railWayAnalyzer(fileName, self)
            else:
                self.listWidget.addItem(QListWidgetItem('Файл не является изображением'))

    def addListWidgetButton(self, string):
        self.listWidget.addItem(string)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = railWayWindow()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
