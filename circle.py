import cv2
import numpy as np
import math

#circle detect
def houf_circle(frame):
    """
    霍夫圆检测
    输入frame
    输出圆心坐标
    """
    #灰度化，高斯滤波
    gray_image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur_image = cv2.GaussianBlur(gray_image,(3,3),0)
    #形态学操作
    equlized_image = cv2.equalizeHist(blur_image)
    equlized_image = cv2.bilateralFilter(equlized_image, 9, 100, 100)  # d=9, sigmaColor=75, sigmaSpace=75
    
    circles = cv2.HoughCircles(equlized_image, cv2.HOUGH_GRADIENT_ALT, dp=1, 
                          minDist=17, param1=90, param2=0.83,
                          minRadius=60, maxRadius=500)#改进的霍夫梯度
    
    if circles is not None:
        circles = circles[0,:,:]
        circles = circles[np.argsort(circles[:, 0])]  # 按第二列（索引1）排序
        
        if(len(circles)>1):
            if(circles[len(circles)-1][0]-circles[len(circles)-2][0]>=10):
                circles = np.delete(circles,-1,axis=0)
                #print("成功排除")
            elif (circles[1][0]-circles[0][0]>=10):
                circles = np.delete(circles,0,axis=0)
                #print("成功排除")
        mean_circle = np.mean(circles[:,:3], axis=0)
        r_min =min(circles[:,2])
        #cv2.circle(frame, (int(mean_circle[0]), int(mean_circle[1])),int(r_min) , (0, 255, 0), 2)#在图中画出来
        cv2.rectangle(frame, (int(mean_circle[0]) - 4, int(mean_circle[1]) - 4), (int(mean_circle[0]) + 4, int(mean_circle[1]) + 4), (0, 128, 255), -1)
        return [mean_circle[0],mean_circle[1],r_min]
    else:
        #print("未检测到圆")
        return None           
#标准1：不出现啥都不是的圆
#标准2：尽可能多的出现圆，且稳定有圆
#标准3：最大圆的圆心应该是检测误差最小的


"""
cap = cv2.VideoCapture(0)
while(1):
    ret,frame = cap.read()
    houf_circle(frame)
    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
"""