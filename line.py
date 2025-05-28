import cv2
import  numpy as np
import math
from utils_usb import *
###################################宏常量定义########################################

USB2_Width = 640
USB2_Height = 480
def apply_morphological_operations(mask, kernel_size=9):
    """
    功能：应用形态学开闭运算
    输入：掩膜矩阵，高斯核大小
    返回值：处理后的掩膜矩阵
    """
    # 确保kernel_size是奇数
    kernel_size = max(1, kernel_size // 2 * 2 + 1)
    
    # 创建结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # 开运算
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 闭运算
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    
    return closing

def UpFull(image,a = 1.95):
    """
    功能：提高饱和度
    输入：图像矩阵，饱和度增强比例a
    返回值：饱和度增强后的rgb图像
    """
  
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 提高饱和度
    hsv[:, :, 1] = hsv[:, :, 1] * a  # 饱和度乘以一个因子 (1.3)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)  # 保证饱和度不超出范围
    output_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return output_image



#
def Get_distance(x0,x1,x2,y1,y2,cx=324,cy=246,K=1/22.4):
    dx = x2 - x1
    dy = y2 - y1
    # 直线方程系数 A, B, C
    A = -dy
    B = dx
    C = dy * x1 - dx * y1
    distance = np.abs(A * cx + B * cy + C) / np.sqrt(A**2 + B**2)
    if(x0<324):
        distance = -distance
        
    return distance*K


def Line_Angle(frame):
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 150, 180, apertureSize=3)

    # 参数说明：1为距离精度，np.pi/180为角度精度，150为阈值
    lines = cv2.HoughLines(edges, 1, np.pi/180, 150)
    thetas = []
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            
            if(theta<=np.pi/3 or theta>=np.pi*2/3):
                
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                #cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)#绿色为检测到的黑线或者其他线
                
                if(theta<=np.pi/3 ):
                    thetas.append(theta+np.pi/2)
                elif(theta>=np.pi*2/3):
                    thetas.append(theta-np.pi/2)
        #实现巡线检测，并返回直线的角度
        if(len(thetas)>0):
            mean_angle = sum(thetas)/len(thetas)
            #print(mean_angle*180/np.pi)
            return mean_angle#返回偏角
        else: return None
    return None
def Line_Angle_out(frame):
    """
    返回x0
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 180, 220, apertureSize=3)

    # 参数说明：1为距离精度，np.pi/180为角度精度，150为阈值
    lines = cv2.HoughLines(edges, 1, np.pi/180, 80)
    select_line = None
    if lines is not None:
        max_x0 = 0
        
        for line in lines:
            
            rho, theta = line[0]#获取线的信息
            
            if(theta<=np.pi/3 or theta>=np.pi*2/3):#避免横线
                
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                
                #print(f"{x0=}")
                if(x0 >=max_x0):
                    max_x0 = x0
                    select_line = line

        if select_line is not None:
            return max_x0

    return 0

def detect_gray_yellow_boundary(frame, kalmen, dT):
    """
    功能：检测灰黄交界线并在图像上绘制无限长的直线
    输入：彩色图像帧, 卡尔曼滤波器实例, 时间间隔dT
    返回值：[angle, distance, filtered_angle, filtered_distance, raw_angle, raw_distance] 或 None
    """
    uf = UpFull(frame)  # 提高饱和度
    
    # 颜色mask提取
    mask_yellow = color_detect(uf, 4)
    mask_gray = color_detect(uf, 5)
    
    # 应用形态学开闭运算
    processed_mask = apply_morphological_operations(mask_yellow)
    cv2.imshow("mask_yellow", processed_mask)
    # 边缘检测
    edges = cv2.Canny(processed_mask, 193, 255, apertureSize=3)
    # 霍夫变换检测直线
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=60)
    
    x0_out  = Line_Angle_out(frame)
    select_line = None
    measurement = None
    angle = 0
    distance = 0
    if lines is not None:
        min_x0 = float('inf')
        for line in lines:
            rho, theta = line[0]  # 获取线的信息
            if theta <= 20/57.3 or theta >= 160/57.3:  # 避免横线
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                if abs(x0-x0_out)<120:
                    #print("排除黑线")
                    continue
                #if x0<240:
                #    continue
 
                if x0 < min_x0:
                    min_x0 = x0
                    select_line = line
        
        if select_line is not None:
            rho, theta = select_line[0]  # 获取线的信息
            # 处理被选中的最左侧的直线
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            distance = Get_distance(x0, x1, x2, y1, y2)

            
            #修正需要的angle
            if theta <= np.pi / 3:
                angle = theta + np.pi / 2
            elif theta >= np.pi * 2 / 3:
                angle = theta - np.pi / 2
            else:
                angle = theta
            
            # 使用卡尔曼滤波器滤波
            measurement = [theta, rho]

    # 如果没有检测到直线，使用卡尔曼滤波器的预测结果
    k_results = kalmen.KalmenCalculate(measurement, dT)
    if k_results is not None:
        filtered_theta = k_results[0]
        filtered_rho = k_results[1]
        a = np.cos(filtered_theta)
        b = np.sin(filtered_theta)
        x0 = a * filtered_rho
        y0 = b * filtered_rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        #cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        filtered_distance = Get_distance(x0, x1, x2, y1, y2)
        #修正需要的angle
        if filtered_theta <= np.pi / 3:
            filtered_angle = filtered_theta + np.pi / 2
        elif filtered_theta >= np.pi * 2 / 3:
            filtered_angle = filtered_theta - np.pi / 2
        else:
            filtered_angle = filtered_theta
            
        return [filtered_angle, filtered_distance, angle, distance]
    
    return None