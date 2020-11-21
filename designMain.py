# for read file os
import os
import sys

# to analyze type
import filetype
# for QT
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QListWidgetItem

import design
from additionalFunction import *

# for computer vision
# To compile QT.ui file to Python code use:
# pyuic5 untitled.ui -o designMain.py
# Create exe
# pyinstaller designMain.py --onefile --noconsole

# constants

SOURCE = "source/"
TEMP = "tmp/"
IMG_NUM = "3"
IMG_TYPE = ".jpg"

os.makedirs("tmp/", exist_ok=True)


class railWayAnalyzer:

    @staticmethod
    def toHLS(image):
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        saveFile(hls, IMG_NUM + '_hls.bmp')
        return hls

    def toContrast(self, image, contrast=127):
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)
        saveFile(buf, IMG_NUM + '_c.bmp')

        ret, threshold_image = cv2.threshold(image, 170, 255, cv2.THRESH_BINARY)
        saveFile(threshold_image, IMG_NUM + '_t.bmp')

        return threshold_image

    def adjustContrast(self, image, contrast=127):
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        buf = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)

        saveFile(buf, IMG_NUM + '_contrast.bmp')

        return buf

    def toThreshold(self, image, spec):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if spec:
            ret, threshold_image = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
        else:
            ret, threshold_image = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
        saveFile(threshold_image, IMG_NUM + '_bin.bmp')

        return threshold_image

    @staticmethod
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

    def grayScale(self, image):
        """
        :param image: Исходное изображение (в формате BGR)
        :return: Изображение, преобразованное из формата BGR в оттенки серого
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        saveFile(gray, IMG_NUM + '_gray.bmp')
        return gray

    @staticmethod
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

    def colorSelection(self, image, darkened, desc):
        """
        :param desc: Описание изображения маски
        :param image: Исходное изображение
        :param darkened: Затемнённое изображение
        :return: Изображение под маской
        """
        white_masks = \
            self.colorMask(image, np.array([150, 150, 150], dtype=np.uint8), np.array([255, 255, 255], dtype=np.uint8))
        # mask = cv2.bitwise_or(white_masks, white_masks)
        mask = white_masks
        masked = cv2.bitwise_and(darkened, darkened, mask=mask)
        saveFile(masked, IMG_NUM + '_masked_' + desc + '.bmp')
        return masked

    @staticmethod
    def gaussianBlur(image, kernelSize=11):
        """
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
        :param high:
        :param low:
        :param image: Исходное изображение
        :return: Применённый детектор Кэнни
        """

        detectedEdges = canny(image, low_threshold=low, high_threshold=high)
        saveFile(detectedEdges, IMG_NUM + '_canny.bmp')

        return detectedEdges

    @staticmethod
    def useSobel(image, kernel=3):
        """
        :param kernel:
        :param image: Исходное изображение
        :return: Применённый детектор Кэнни
        """

        grad_x = cv2.Sobel(image, cv2.CV_8U, 1, 0, ksize=kernel)
        saveFile(grad_x, IMG_NUM + '_sobel.bmp')

        abs_sobel64f = np.absolute(grad_x)
        sobel_x = np.uint8(abs_sobel64f)

        return sobel_x

    @staticmethod
    def areaOfInterest(image):
        """
        :param image: Исходное изображение
        :return:
        """
        aoi = get_aoi(image)
        saveFile(aoi, IMG_NUM + '_AOI.bmp')
        return aoi

    @staticmethod
    def houghTransformLineDetection(origImage, image):
        """
        :param origImage:
        :param image: Исходное изображение
        :return:
        """
        line = get_hough_lines(image)

        hough = draw_lines(origImage, line, height=image.shape[0])
        saveFile(hough, IMG_NUM + '_hought.bmp')
        return hough

    def __init__(self, fileName, window):
        image = cv2.imread(fileName)

        name = fileName.split("/")
        name = name[len(name)-1].split(".")
        self.name = name[0]

        f = 131 * (127 + 127) / (127 * (131 - 127))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        test = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)
        test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
        ret, test = cv2.threshold(test, 220, 255, cv2.THRESH_BINARY)
        nonzero = cv2.countNonZero(test)
        imgSize = test.shape[0] * test.shape[1]
        res = nonzero / float(imgSize) * 100

        window.addListWidgetButton(QListWidgetItem('Original'))

        spec = False
        if res > 60.0:
            spec = True
            self.name = "winter/" + self.name
        else:
            self.name = "classic/" + self.name

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

        hough = self.houghTransformLineDetection(image, aoi)
        window.addListWidgetButton(QListWidgetItem('Hought'))

        image = cv2.imread("tmp/3_hought.bmp")

        saveFileFinal(hough, self.name+".jpeg")

        window.drawImage(image)


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
        self.mw.setFixedSize(image.shape[1], image.shape[0])
        image = QtGui.QImage(image.data,
                             image.shape[1],
                             image.shape[0],
                             QtGui.QImage.Format_RGB888).rgbSwapped()
        pm = QtGui.QPixmap.fromImage(image)
        self.label.setPixmap(pm)

    def openFile(self):
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
