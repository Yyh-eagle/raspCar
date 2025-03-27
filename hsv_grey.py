import cv2
import numpy as np
import time
from line import *
# 初始HSV阈
# 定义全局变量
hough_threshold = 70  # 初始霍夫直线检测阈值
min_line_length = 25  # 初始最小直线长度
lower_h = 0
lower_s = 0
lower_v = 46
upper_h = 180
upper_s = 40
upper_v = 254

lower_ground_gray = np.array([0, 0, 46])   # 地面灰色的HSV阈值下限
upper_ground_gray = np.array([180, 40, 254])   # 地面灰色的HSV阈值上限


def color_detect(frame,lower_ground_yellow,upper_ground_yellow):
    """
    功能：颜色获取
    输入：图像矩阵，颜色选择
    返回值：掩膜矩阵
    """
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)      # 图像从BGR颜色模型转换为HSV模型
    img_blur = cv2.GaussianBlur(hsv_img, (5, 5), 0)
    
    mask = cv2.inRange(img_blur, lower_ground_yellow, upper_ground_yellow)
  
    return mask
def update_lower_h(value):
    global lower_h
    lower_h = value

def update_lower_s(value):
    global lower_s
    lower_s = value

def update_lower_v(value):
    global lower_v
    lower_v = value

def update_upper_h(value):
    global upper_h
    upper_h = value

def update_upper_s(value):
    global upper_s
    upper_s = value

def update_upper_v(value):
    global upper_v
    upper_v = value

def update_hough_threshold(value):
    global hough_threshold
    hough_threshold = value

def update_min_line_length(value):
    global min_line_length
    min_line_length = value
def hough_line_detection(image, mask, threshold, min_line_length):
    """
    功能：霍夫直线检测
    输入：图像矩阵，掩膜矩阵，霍夫直线检测阈值，最小直线长度
    返回值：带有检测直线的图像
    """
    edges = cv2.Canny(mask, 193, 255, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold, minLineLength=min_line_length, maxLineGap=10)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    return image, edges
def main():
    global hough_threshold, min_line_length, lower_h, lower_s, lower_v, upper_h, upper_s, upper_v

    # 创建窗口
    cv2.namedWindow('摄像头')
    cv2.namedWindow('mask_yellow')
    #cv2.namedWindow('hough_lines')
    #cv2.namedWindow('canny_edges')

    # 创建滑动条
    #cv2.createTrackbar('Hough Threshold', 'hough_lines', hough_threshold, 200, update_hough_threshold)
    #cv2.createTrackbar('Min Line Length', 'hough_lines', min_line_length, 200, update_min_line_length)
    cv2.createTrackbar('Lower H', 'mask_yellow', lower_h, 180, update_lower_h)
    cv2.createTrackbar('Lower S', 'mask_yellow', lower_s, 255, update_lower_s)
    cv2.createTrackbar('Lower V', 'mask_yellow', lower_v, 255, update_lower_v)
    cv2.createTrackbar('Upper H', 'mask_yellow', upper_h, 180, update_upper_h)
    cv2.createTrackbar('Upper S', 'mask_yellow', upper_s, 255, update_upper_s)
    cv2.createTrackbar('Upper V', 'mask_yellow', upper_v, 255, update_upper_v)

    # 打开默认摄像头
    cap = cv2.VideoCapture("use_videos/realine.avi")

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    while True:
        time.sleep(0.1)
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            print("无法接收帧（流结束？）")
            break

        # 提高饱和度
        uf = UpFull(frame)
        # 更新HSV阈值
        lower_ground_yellow = np.array([lower_h, lower_s, lower_v])
        upper_ground_yellow = np.array([upper_h, upper_s, upper_v])
        # 获取地面黄色掩膜
        mask_yellow = color_detect(uf, lower_ground_yellow,upper_ground_yellow)
        
        # 应用形态学开闭运算
        processed_mask_yellow = apply_morphological_operations(mask_yellow, kernel_size=9)

        # 霍夫直线检测
        frame_with_lines, edges = hough_line_detection(frame.copy(), processed_mask_yellow, hough_threshold, min_line_length)

        # 显示帧
        cv2.imshow('摄像头', uf)
        cv2.imshow('mask_yellow', processed_mask_yellow)
        #cv2.imshow('hough_lines', frame_with_lines)
        #cv2.imshow('canny_edges', edges)


        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头并关闭窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()