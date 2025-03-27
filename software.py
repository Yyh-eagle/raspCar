import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, pyqtSlot#信号发送，信号接收
from PyQt5.QtGui import QImage, QPixmap#图像画布，matplotlib画布
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import rclpy
from cv_bridge import CvBridge
from std_msgs.msg import Float64
from Image_gongchuang import ImagePublisher  ############

#线程1 ros2正常运作
class Ros2Thread(QThread):
    image_signal  = pyqtSignal(np.ndarray, np.ndarray)
    data_signal = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.Node = None

    def run(self):
        rclpy.init()
        self.node = ImagePublisher("Image") 
        self.node.publisher_ = self.node.create_publisher(Float64, '/usb2/e_x_filter', 10)

        # 处理图像数据
        while rclpy.ok():
            frame1 = self.node.usb1.read()
            frame2 = self.node.usb2.read()
            self.image_signal.emit(frame1, frame2)

            if hasattr(self.node, 'e_x'):
                self.data_signal.emit(self.node.e_x)

            rclpy.spin_once(self.node, timeout_sec=0.1)  # 控制ROS循环频率
        rclpy.shutdown()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.init_ros2()
        self.data_buffer = []

    def init_ui(self):
        self.setWindowTitle("工创赛检测软件")
        self.setGeometry(100, 100, 1600, 950)
        # 图像显示区域
        self.image_label1 = QLabel()
        self.image_label2 = QLabel()

        # Matplotlib绘图区域
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.line, = self.ax.plot([], [])
        self.ax.set_title("e_x Filter Data")
        self.ax.set_xlabel("time")
        self.ax.set_ylabel("data")

        # 布局
        img_layout = QHBoxLayout()
        img_layout.addWidget(self.image_label1)
        img_layout.addWidget(self.image_label2)

        main_layout = QVBoxLayout()
        main_layout.addLayout(img_layout)
        main_layout.addWidget(self.canvas)

        self.setLayout(main_layout)

    def init_ros2(self):
        self.ros_thread = Ros2Thread()
        self.ros_thread.image_signal.connect(self.update_images)
        self.ros_thread.data_signal.connect(self.update_plot)
        self.ros_thread.start()

         # 定时器处理数据更新
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_images)
        self.timer.start(50)  # 每50毫秒更新一次

    def process_images(self):
        # 可以根据需要处理或更新数据
        pass
    #图像处理信号接收处
    @pyqtSlot(np.ndarray, np.ndarray)
    def update_images(self, frame1, frame2):
        # 转换并显示USB1图像
        h, w, c = frame1.shape
        qimg1 = QImage(frame1.data, w, h, QImage.Format_RGB888).rgbSwapped()
        self.image_label1.setPixmap(QPixmap.fromImage(qimg1))

        # 转换并显示USB2图像
        h, w, c = frame2.shape
        qimg2 = QImage(frame2.data, w, h, QImage.Format_RGB888).rgbSwapped()
        self.image_label2.setPixmap(QPixmap.fromImage(qimg2))

    @pyqtSlot(float)
    def update_plot(self, value):
        # 保持最近100个数据点
        self.data_buffer.append(value)
        if len(self.data_buffer) > 100:
            self.data_buffer.pop(0)
        
        # 更新绘图
        self.line.set_data(np.arange(len(self.data_buffer)), self.data_buffer)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

    def closeEvent(self, event):
        self.ros_thread.terminate()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())