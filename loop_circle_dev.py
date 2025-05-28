import cv2
import  numpy as np
from utils_usb import *


       

def main():

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
       
        # 显示帧
        cv2.imshow('摄像头', frame)


        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头并关闭窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()