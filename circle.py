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
    blur_image = cv2.GaussianBlur(gray_image,(5,5),0)
    #形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5), (-1, -1))
    blur_image = cv2.morphologyEx(blur_image, cv2.MORPH_CLOSE, kernel, (-1, -1))
    blur_image = cv2.morphologyEx(blur_image, cv2.MORPH_OPEN, kernel, (-1, -1))
    #cv2.imshow("blur image",frame)
    #改进的霍夫圆检测
    circles = cv2.HoughCircles(blur_image, cv2.HOUGH_GRADIENT_ALT, dp=1, 
                          minDist=20, param1=200, param2=0.9,
                          minRadius=75, maxRadius=500)#改进的霍夫梯度

    if circles is not None:
        mean_circle = np.mean(circles[:,0, :3], axis=0)
        circles = np.round(circles[0, :]).astype("int")
        radius = []
        for (x, y, r) in circles:
            radius.append(r)
        r_min =min(radius)
        cv2.circle(frame, (int(mean_circle[0]), int(mean_circle[1])),int(r_min) , (0, 255, 0), 2)#在图中画出来
        
        cv2.rectangle(frame, (int(mean_circle[0]) - 5, int(mean_circle[1]) - 5), (int(mean_circle[0]) + 5, int(mean_circle[1]) + 5), (0, 128, 255), -1)
        #cv2.putText(frame, "center:("+str(mean_circle[0])+","+str(mean_circle[1])+")", (5,40), cv2.FONT_HERSHEY_SIMPLEX,.9, (0, 0, 255), 2)
        return [mean_circle[0],mean_circle[1],r_min]
    else:
        #print("未检测到圆")
        return None           




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