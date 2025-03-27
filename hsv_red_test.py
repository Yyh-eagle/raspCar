import cv2
import numpy as np
import time

# 定义全局变量
lower_h1 = 0
lower_h2 = 160
lower_s = 103
lower_v = 100
upper_h1 = 15
upper_h2 = 180
upper_s = 214
upper_v = 255

def color_detect(frame, lower_red1, upper_red1, lower_red2, upper_red2):
    """
    功能：颜色获取
    输入：图像矩阵，颜色选择
    返回值：掩膜矩阵
    """
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 图像从BGR颜色模型转换为HSV模型
    img_blur = cv2.GaussianBlur(hsv_img, (5, 5), 0)

    mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    return mask

def update_lower_h1(value):
    global lower_h1
    lower_h1 = value

def update_lower_h2(value):
    global lower_h2
    lower_h2 = value

def update_lower_s(value):
    global lower_s
    lower_s = value

def update_lower_v(value):
    global lower_v
    lower_v = value

def update_upper_h1(value):
    global upper_h1
    upper_h1 = value

def update_upper_h2(value):
    global upper_h2
    upper_h2 = value

def update_upper_s(value):
    global upper_s
    upper_s = value

def update_upper_v(value):
    global upper_v
    upper_v = value

def apply_morphological_operations(mask, kernel_size=9):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    return closing

def main():
    global lower_h1, lower_h2, lower_s, lower_v, upper_h1, upper_h2, upper_s, upper_v

    # 创建窗口
    cv2.namedWindow('摄像头')
    cv2.namedWindow('mask_yellow')

    # 创建滑动条
    cv2.createTrackbar('Lower H1', 'mask_yellow', lower_h1, 180, update_lower_h1)
    cv2.createTrackbar('Lower H2', 'mask_yellow', lower_h2, 180, update_lower_h2)
    cv2.createTrackbar('Lower S', 'mask_yellow', lower_s, 255, update_lower_s)
    cv2.createTrackbar('Lower V', 'mask_yellow', lower_v, 255, update_lower_v)
    cv2.createTrackbar('Upper H1', 'mask_yellow', upper_h1, 180, update_upper_h1)
    cv2.createTrackbar('Upper H2', 'mask_yellow', upper_h2, 180, update_upper_h2)
    cv2.createTrackbar('Upper S', 'mask_yellow', upper_s, 255, update_upper_s)
    cv2.createTrackbar('Upper V', 'mask_yellow', upper_v, 255, update_upper_v)

    # 打开默认摄像头
    cap = cv2.VideoCapture(2)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    while True:
        time.sleep(0.03)
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            print("无法接收帧（流结束？）")
            break

        # 更新HSV阈值
        lower_red1 = np.array([lower_h1, lower_s, lower_v])
        upper_red1 = np.array([upper_h1, upper_s, upper_v])
        lower_red2 = np.array([lower_h2, lower_s, lower_v])
        upper_red2 = np.array([upper_h2, upper_s, upper_v])

        # 获取地面黄色掩膜
        mask_yellow = color_detect(frame, lower_red1, upper_red1, lower_red2, upper_red2)

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