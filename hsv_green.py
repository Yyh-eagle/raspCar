import cv2
import numpy as np
import time
from line import *
# 初始HSV阈
# 定义全局变量

lower_h = 37
lower_s = 10
lower_v = 39
upper_h = 83
upper_s = 220
upper_v = 255



def color_detect(frame,lower,upper):
    """
    功能：颜色获取
    输入：图像矩阵，颜色选择
    返回值：掩膜矩阵
    """
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)      # 图像从BGR颜色模型转换为HSV模型
    img_blur = cv2.GaussianBlur(hsv_img, (5, 5), 0)
    
    mask = cv2.inRange(img_blur, lower, upper)
  
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

def main():
    global hough_threshold, min_line_length, lower_h, lower_s, lower_v, upper_h, upper_s, upper_v

    # 创建窗口
    cv2.namedWindow('摄像头')
    cv2.namedWindow('mask_yellow')



    cv2.createTrackbar('Lower H', 'mask_yellow', lower_h, 180, update_lower_h)
    cv2.createTrackbar('Lower S', 'mask_yellow', lower_s, 255, update_lower_s)
    cv2.createTrackbar('Lower V', 'mask_yellow', lower_v, 255, update_lower_v)
    cv2.createTrackbar('Upper H', 'mask_yellow', upper_h, 180, update_upper_h)
    cv2.createTrackbar('Upper S', 'mask_yellow', upper_s, 255, update_upper_s)
    cv2.createTrackbar('Upper V', 'mask_yellow', upper_v, 255, update_upper_v)

    # 打开默认摄像头
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    while True:
        #time.sleep(0.03)
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            print("无法接收帧（流结束？）")
            break

        # 更新HSV阈值
        lower = np.array([lower_h, lower_s, lower_v])
        upper = np.array([upper_h, upper_s, upper_v])
        # 获取地面黄色掩膜
        mask_yellow = color_detect(frame, lower,upper)
        
        # 应用形态学开闭运算
        processed_mask_yellow = apply_morphological_operations(mask_yellow, kernel_size=9)

       
        # 显示帧
        cv2.imshow('摄像头', frame)
        cv2.imshow('mask_yellow', processed_mask_yellow)
    

        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头并关闭窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()