# -*- coding: utf-8 -*-
# @Time    : 2023/8/4 9:50
# @Author  : XianYang
# @Email   : xy_mts@163.com
# @File    : p.py

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.openimg = QtWidgets.QPushButton(self.centralwidget)
        self.openimg.setGeometry(QtCore.QRect(230, 780, 151, 41))
        self.openimg.setStyleSheet("background-color: lightblue;")
        self.openimg.setObjectName("openimg")

        self.exit = QtWidgets.QPushButton(self.centralwidget)
        self.exit.setGeometry(QtCore.QRect(710, 780, 151, 41))
        self.exit.setObjectName("exit")
        self.noticetxt = QtWidgets.QTextBrowser(self.centralwidget)
        self.noticetxt.setGeometry(QtCore.QRect(390, 20, 361, 81))
        self.noticetxt.setObjectName("noticetxt")
        self.noticetxt.setStyleSheet("border: 2px solid green;")
        self.noticetxt.setStyleSheet("background-color: rgba(255, 132, 139, 0);")

        self.showimg = QtWidgets.QLabel(self.centralwidget)
        self.showimg.setGeometry(QtCore.QRect(73, 141, 451, 491))
        self.showimg.setText("")
        self.showimg.setObjectName("showimg")

        self.showlogo = QtWidgets.QLabel(self.centralwidget)
        self.showlogo.setGeometry(QtCore.QRect(870, 290, 170, 120))
        logo = r'F:\XY_mts\yolov5-bone_age\my_bone_age\MyQT\logo.png'
        self.jpg = QtGui.QPixmap(logo).scaled(200, 200)
        # self.showlogo.setPixmap(self.jpg)


        self.gif_label = QtWidgets.QLabel(self.centralwidget)
        self.gif_label.setScaledContents(True)
        self.gif_label.setGeometry(QtCore.QRect(530, 350, 100, 100))
        gif_path = r'F:\XY_mts\yolov5-bone_age\my_bone_age\MyQT\6.gif'
        self.movie = QtGui.QMovie(gif_path)
        # self.gif_label.setMovie(movie)
        # movie.start()

        # self.bone = QtWidgets.QLabel(self.centralwidget)
        # self.bone.setScaledContents(True)
        # self.bone.setGeometry(QtCore.QRect(173, 141, 451, 491))

        self.showres = QtWidgets.QTextBrowser(self.centralwidget)
        self.showres.setGeometry(QtCore.QRect(640, 290, 401, 241))
        self.showres.setObjectName("showres")
        self.showres.setStyleSheet("background-color: rgba(255, 132, 139, 0);color: red;")

        self.run_now = QtWidgets.QPushButton(self.centralwidget)
        self.run_now.hide()
        self.run_now.setGeometry(QtCore.QRect(510, 780, 151, 41))
        self.run_now.setObjectName("run_now")
        self.run_now.setStyleSheet("background-color: yellow;")

        self.boy = QtWidgets.QRadioButton(self.centralwidget)
        self.boy.hide()
        self.boy.setGeometry(QtCore.QRect(410, 770, 89, 16))
        self.boy.setObjectName("boy")
        self.girl = QtWidgets.QRadioButton(self.centralwidget)
        self.girl.hide()
        self.girl.setGeometry(QtCore.QRect(410, 810, 89, 16))
        self.girl.setObjectName("girl")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1124, 23))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menu.menuAction())

        self.retranslateUi(MainWindow)
        self.exit.clicked.connect(MainWindow.close)  # type: ignore
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


        self.openimg.clicked.connect(self.openImage)
        self.run_now.clicked.connect(self.runs)
        self.boy.clicked.connect(self.choose_boy)
        self.girl.clicked.connect(self.choose_girl)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.openimg.setText(_translate("MainWindow", "choose img"))
        self.exit.setText(_translate("MainWindow", "exit"))
        self.noticetxt.setHtml(_translate("MainWindow",
                                          "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                          "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                          "p, li { white-space: pre-wrap; }\n"
                                          "</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
                                          "<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:16pt; color:#ff0000;\">Joint laboratory bone age detection system</span></p>\n"
                                          "<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-style:italic;color:#39ff14;\">Welcome to use bone age detection interface.</span></p>\n"
                                          "</body></html>"))
        self.showres.setHtml(_translate("MainWindow",
                                        "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                        "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                        "p, li { white-space: pre-wrap; }\n"
                                        "</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
                                        "<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">After uploading the detection image, please view the results here.</p>\n"
                                        "<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
                                        "<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.run_now.setText(_translate("MainWindow", "run now"))
        self.boy.setText(_translate("MainWindow", "boy"))
        self.girl.setText(_translate("MainWindow", "girl"))
        self.menu.setTitle(_translate("MainWindow", "阿里嘎多美羊羊桑"))

    def openImage(self):  # 选择本地图片上传
        global imgName

        # 弹出一个文件选择框，第一个返回值imgName记录选中的文件路径+文件名，第二个返回值imgType记录文件的类型
        imgName, imgType = QFileDialog.getOpenFileName(self.centralwidget, "打开图片", "", "All Files(*);;*.png;;*.jpg")
        jpg = QtGui.QPixmap(imgName).scaled(self.showimg.width(),
                                            self.showimg.height())  # 通过文件路径获取图片文件，并设置图片长宽为label控件的长款
        self.showimg.setPixmap(jpg)  # 在label控件上显示选择的图片
        if imgName:
            self.gif_label.setMovie(self.movie)
            self.movie.start()
            self.boy.show()
            self.girl.show()

    def choose_boy(self):
        global sex
        sex = 'boy'
        self.run_now.show()

    def choose_girl(self):
        global sex
        sex = 'girl'
        self.run_now.show()

    def runs(self):
        from my_bone_age.MyQT.qtrun import rr
        res = rr(imgName, sex)

        self.gif_label.setMovie(None)
        self.gif_label.deleteLater()

        self.showres.setText(res)
        self.showlogo.setPixmap(self.jpg)




