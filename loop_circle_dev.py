import cv2
import  numpy as np
from utils_usb import *


def GetCenterColor_usb1_all(frame,color):#只找绿色圆环
    #但是要返回此时最多的颜色

        states= houf_circle(frame)#返回的是偏差值
        print(states)
        if states is not None: #找到圆        
            cv2.circle(frame, (int(states[0]), int(states[1])),int(states[2]) , (0, 255, 0), 2)#在图中画出来
            cv2.circle(frame, (int(states[0]), int(states[1])), 2, (0, 0, 0), -1)

            return [2,states[0],states[1],states[0],states[1]]

        else:
            return []


def houf_circle_all(frame,color):
    """
    霍夫圆检测
    输入frame
    输出圆心坐标,需要对准的颜色
    """
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur_image = cv2.GaussianBlur(gray_frame,(3,3),0)
    #直方图均衡化
    equlized_image = cv2.equalizeHist(blur_image)
    equlized_image = cv2.bilateralFilter(equlized_image, 9, 100, 100)  # d=9, sigmaColor=75, sigmaSpace=75
    
    circles = cv2.HoughCircles(equlized_image, cv2.HOUGH_GRADIENT_ALT, dp=1, 
                          minDist=17, param1=88, param2=0.75,
                          minRadius=60, maxRadius=1000)#改进的霍夫梯度
    
    if circles is not None:
        circles = circles[0,:,:]
        circles_filted = []
        for c in circles:
            
            left = int(c[0]-c[2])
            right = int(c[0]+c[2])
            up = int(c[1]-c[2])
            down =int (c[1]+c[2])
            ROI = GetROI2(frame,left+20,right-20,up+20,down-20)
            mask = color_detect(ROI,color)
           # cv2.imshow("ros",mask_green)
            area = get_area(mask)
            print(area_green)
            if(area>6000):#todo这个值需要重新确定
                cv2.circle(frame, (int(c[0]), int(c[1])), int(c[2]), (0, 255, 0), 2)
                circles_filted.append(c)
        if len(circles_filted)>0:
            # 将 circles_filted 转换为 NumPy 数组
            circles_filted = np.array(circles_filted)
      
       
            #判断是否有多个元素
            if circles_filted.size > 0:
                circles_filted = circles_filted[np.argsort(circles_filted[:, 0])]
        
            if(len(circles_filted)>1):
                if(circles[len(circles)-1][0]-circles[len(circles)-2][0]>=10):
                    circles = np.delete(circles,-1,axis=0)
                elif (circles[1][0]-circles[0][0]>=10):
                    circles = np.delete(circles,0,axis=0)
  
            mean_circle = np.mean(circles_filted, axis=0)
            r_min =min(circles[:,2])
  
            return [mean_circle[0],mean_circle[1],r_min]
        else: 
            return None
    else:
        #print("未检测到圆")
        return None  
       

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