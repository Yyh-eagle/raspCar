import sys
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QGridLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QKeyEvent


class FullScreenNumberDisplay(QWidget):
    def __init__(self, numbers):
        """
        初始化全屏数字显示窗口。

        :param numbers: 包含6个数字的列表，例如 [1, 2, 3, 2, 3, 1]
        """
        super().__init__()
        self.numbers = numbers  # 传入的数字列表
        self.initUI()

    def initUI(self):
        """
        初始化用户界面。
        """
        # 设置窗口为全屏
        self.showFullScreen()

        # 设置背景为白色
        self.setStyleSheet("background-color: white;")

        # 使用网格布局来排列数字和加号
        layout = QGridLayout(self)

        # 设置字体
        font = QFont()
        font.setPointSize(200)  # 字体大小

        # 将数字和加号均匀分布在屏幕上
        for i, number in enumerate(self.numbers):
            label = QLabel(str(number), self)
            label.setAlignment(Qt.AlignCenter)  # 文字居中
            label.setFont(font)
            label.setStyleSheet("color: black;")  # 文字颜色为黑色

            # 计算位置：2行 x 4列（数字和加号交替）
            row = i // 3  # 行号
            col = i % 3   # 列号
            layout.addWidget(label, row, col)

            # 在数字之间插入加号
            if i == 1:  # 在第3个数字后插入加号
                plus_label = QLabel("+", self)
                plus_label.setAlignment(Qt.AlignCenter)
                plus_label.setFont(font)
                plus_label.setStyleSheet("color: black;")
                layout.addWidget(plus_label, row, col + 1)

        # 设置布局
        self.setLayout(layout)

    def keyPressEvent(self, event: QKeyEvent):
        """
        处理键盘事件，支持 Ctrl+C 退出。
        """
        if event.key() == Qt.Key_C and event.modifiers() == Qt.ControlModifier:
            self.close()  # 关闭窗口
            QApplication.quit()  # 退出应用


def display_numbers(numbers):
    """
    显示数字数组的全屏窗口。

    :param numbers: 包含6个数字的列表，例如 [1, 2, 3, 2, 3, 1]
    """
    # 检查列表长度
    if len(numbers) != 6:
        print("请确保传入的数字列表长度为6。")
        return

    # 创建应用
    app = QApplication(sys.argv)
    window = FullScreenNumberDisplay(numbers)
    window.show()

    # 运行应用
    sys.exit(app.exec_())


