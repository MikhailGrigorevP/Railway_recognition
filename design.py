# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtWidgets


class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        """
        :param MainWindow: UI window
        """
        MainWindow.setObjectName("RailWay finder")
        MainWindow.resize(683, 686)

        self.mw = MainWindow

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.label.setScaledContents(True)
        self.verticalLayout.addWidget(self.label)

        self.listWidget = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget.setObjectName("listWidget")
        self.verticalLayout.addWidget(self.listWidget)

        self.fileButton = QtWidgets.QPushButton(self.centralwidget)
        self.fileButton.setObjectName("fileButton")
        self.verticalLayout.addWidget(self.fileButton)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        """
        :param MainWindow: UI window
        """
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("RailWay finder", "RailWay finder"))
        self.label.setText(_translate("RailWay finder", "Здесь будет ваше изображение"))
        self.fileButton.setText(_translate("RailWay finder", "Выберите файл"))
