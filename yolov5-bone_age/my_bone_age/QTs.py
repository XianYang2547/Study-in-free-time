
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
from MyQT import untitled

if __name__ == '__main__':
    app = QApplication(sys.argv)  # 创建应用程序对象
    MainWindow = QMainWindow()  # 创建主窗口
    MainWindow.setStyleSheet("#MainWindow{border-image:url(../my_bone_age/MyQT/background.jpg)}")
    MainWindow.setFixedSize(1200,900)
    ui = untitled.Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()  # 显示主窗口
    sys.exit(app.exec_())  # 在主线程中退出






