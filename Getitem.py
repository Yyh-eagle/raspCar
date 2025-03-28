import cv2
import  numpy as np
from utils_usb import *

K_2dcode = 0.5
#找到视野中央的物料




def GetColor_usb1(frame,color_num):
    #都返回，但是要返回此时最多的颜色

    ans =[]
    closest_color = 0
    
    mask_red = color_detect(frame,1)
    mask_green = color_detect(frame,2)
    mask_blue= color_detect(frame,3)

    #得到掩膜的面积用于排除噪声#todo调整所有的hsv参数
    area_green = get_area(mask_green)
    area_blue = get_area(mask_blue)
    area_red = get_area(mask_red)
    
    # 比较面积以决定颜色
    areas = {2: area_green, 3: area_blue, 1: area_red}
    max_color = max(areas, key=areas.get)
    result = np.zeros_like(frame)
    if max_color == 1:
        result[mask_red > 0] = [255, 255, 255]  # 红色为白色
    elif max_color == 2:
        result[mask_green > 0] = [255, 255, 255]  # 绿色色为白色
    elif max_color == 3:
        result[mask_blue > 0] = [255, 255, 255]  # 蓝色为白色

    #获取result中的白色面积
    white_area = cv2.countNonZero(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY))
    #print(f"{white_area=}")
    if(white_area<20000):
        return []
    eros =  ErosAndDia(result)
    kernel = np.ones((10, 10), dtype=np.uint8)
    eros = cv2.dilate(eros, kernel, 1) # 1:迭代次数，也就是执行几次膨胀操作
    #cv2.imshow("eros",eros)    
    state =cnts_draw(frame,eros)# return [x,y,w,h]
    #print(state)
    #获取中心：、
    if len(state)>0:
        closest_color = max_color 
        x,y,w,h = state[0],state[1],state[2],state[3]
        #print(w*h)
        if w*h <=20000:
            return []
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #cv2.circle(frame, (int(x+w/2), int(y+h/2)), int(w/2), (0, 0, 255), 2)
        cv2.putText(frame, str(closest_color), (x+20,y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        ans = [closest_color,x+w/2,y+h/2,x+w/2,y+h/2]   
        return ans
    else:
        return []

def GetCenterColor_usb1(frame):#只找绿色圆环
    #都返回，但是要返回此时最多的颜色

        states= houf_circle(frame)#返回的是偏差值
        print(states)
        if states is not None: #找到圆        
            cv2.circle(frame, (int(states[0]), int(states[1])),int(states[2]) , (0, 255, 0), 2)#在图中画出来
            cv2.circle(frame, (int(states[0]), int(states[1])), 2, (0, 0, 0), -1)

            return [2,states[0],states[1],states[0],states[1]]

        else:
            return []

def houf_circle(frame):
    """
    霍夫圆检测
    输入frame
    输出圆心坐标
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
            mask_green = color_detect(ROI,6)
           # cv2.imshow("ros",mask_green)
            area_green = get_area(mask_green)
            #print(area_green)
            if(area_green>6000):
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
       

