import cv2
import  numpy as np
from utils_usb import *
from circle import houf_circle
K_2dcode = 0.5
#找到视野中央的物料
def GetCenterColor_usb1(frame,kalmen,dT):
    """
    输入：图像帧
    输出：得到目标的中心点坐标centerx，center_y， 目标的颜色
    """
    
    #进行霍夫圆检测
    states= houf_circle(frame)#返回的是偏差值
    ans = []
    if(states is not None):
         #判定当前圆的颜色，用
        left = int(states[0]-states[2])
        right = int(states[0]+states[2])
        up = int(states[1]-states[2])
        down =int (states[1]+states[2])
        ROI = GetROI2(frame,left+5,right-5,up+5,down-5)
        #cv2.imshow("ROI",ROI)
        #得到颜色掩膜
        mask_red = color_detect(frame,"red")
        mask_green = color_detect(frame,"green")
        mask_blue= color_detect(frame,"blue")

        #计算掩膜面积
        area_red = get_area(mask_red)
        area_green = get_area(mask_green)
        area_blue = get_area(mask_blue)
        #result二值化
        areas = {
        'red': area_red,
        'green': area_green,
        'blue': area_blue
        }
        result = np.zeros_like(frame)
        # 按面积排序，找出最大的那个
        closest_color = max(areas, key=areas.get)           
        ans.append(closest_color)
    else:
        ans.append("None")
    k_results = kalmen.KalmenCalculate(states,dT)
    if k_results[0]>1e-2:
        
        ans.append(k_results[0])
        ans.append(k_results[1])
        if(states is None):
            ans.append(0)
            ans.append(0)
        else:
            ans.append(states[0])
            ans.append(states[1])
        
        
        #cv2.circle(frame, (int(k_results[0]), int(k_results[1])),int(states[2]) , (255, 0, 0), 2)#在图中画出来
        cv2.rectangle(frame, (int(k_results[0]) - 5, int(k_results[1]) - 5), (int(k_results[0]) + 5, int(k_results[1]) + 5), (255, 0, 0), -1)#紫色中心点
        #cv2.putText(frame, "center:("+str(int(k_results[0]))+","+str(int(k_results[1]))+")", (5,40), cv2.FONT_HERSHEY_SIMPLEX,.9, (0, 0, 255), 2)
        return ans
    else:
        return []

def GetCenterColor_usb2(frame,kalmen,dT):
    """
    输入：图像帧
    输出：得到目标的中心点坐标centerx，center_y， 
    """
    k_results = []
    closest_color = "None"
    states = []
    #得到颜色掩膜
    mask_red = color_detect(frame,"red")
    mask_green = color_detect(frame,"green")
    mask_blue= color_detect(frame,"blue")

    #计算掩膜面积
    area_red = get_area(mask_red)
    area_green = get_area(mask_green)
    area_blue = get_area(mask_blue)
    #result二值化
    areas = {
    'red': area_red,
    'green': area_green,
    'blue': area_blue
    }
    result = np.zeros_like(frame)
    # 按面积排序，找出最大的那个
    closest_color = max(areas, key=areas.get)           

    # 输出最接近的物料的中心点坐标
    if closest_color == 'red':
        result[mask_red > 0] = [255, 255, 255]  # 红色为白
        eros =  ErosAndDia(result)
        
    elif closest_color == 'green':
        result[mask_green > 0] = [255, 255, 255]  # 黄色为白色
        eros =  ErosAndDia(result)
        
    elif closest_color == 'blue':
        result[mask_blue > 0] = [255, 255, 255]  # 蓝色为白色
        eros =  ErosAndDia(result)
        
    state =cnts_draw(frame,eros,"max")
    if len(state)==0:
        k_results = kalmen.KalmenCalculate(state,dT)
    else:
        
        k_results = kalmen.KalmenCalculate([state],dT)
    #print(k_results)
    if(len(k_results)>0):
        if k_results[6]>1e-2:
            
            e_x = CaculateFeedback_X(K_2dcode,k_results[0])
            
            result = [closest_color,e_x]
            cv2.rectangle(frame, (int(k_results[2]), int(k_results[3])), (int(k_results[4]),int(k_results[5])), (255, 0, 0), 2)#在图中画出来
            cv2.putText(frame, closest_color+'------'+str(e_x), (5,40), cv2.FONT_HERSHEY_SIMPLEX,.9, (0, 0, 255), 2)
            
            return result
        else: return []
    else:return []

#circle detect
def circle_fit(frame):
    """
    拟合圆检测
    输入frame
    输出圆心坐标
    """
    #灰度化，高斯滤波
    gray_image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur_image = cv2.GaussianBlur(gray_image,(5,5),0)
    
    
    #找到椭圆
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5), (-1, -1))
    blur_image = cv2.morphologyEx(blur_image, cv2.MORPH_CLOSE, kernel, (-1, -1))
    blur_image = cv2.morphologyEx(blur_image, cv2.MORPH_OPEN, kernel, (-1, -1))

    #找到边界
    max_cnt = ContourFilter(blur_image)
    if(max_cnt is None):
        return[0,0]
    
    arc_length = cv2.arcLength(max_cnt, True)
    radius = arc_length / (2 * math.pi)
        
    if not (25 < radius and radius < 600):
        print("半径不符合要求")
        return [0,0]
    #拟合椭圆    
    ellipse = cv2.fitEllipse(max_cnt)

    if float(ellipse[1][0]) / float(ellipse[1][1]) > 0.8 and float(ellipse[1][0]) / float(ellipse[1][1]) < 1.2:#e
        corner = cv2.approxPolyDP(max_cnt, 0.02 * arc_length, True)
        cornerNum = len(corner)
        if cornerNum > 4: # 当cornerNum=4时，识别矩形；而cornerNum>4时，识别圆
            cv2.circle(frame, (int(ellipse[0][0]), int(ellipse[0][1])), int(0.25*(ellipse[1][0]+ellipse[1][1])), (0, 255, 0), thickness=2)
            cv2.imshow("hourf", frame)
            return [CaculateFeedback_X(1,int(ellipse[0][0])), CaculateFeedback_Y(1,int(ellipse[0][1]))]

    return [0,0]