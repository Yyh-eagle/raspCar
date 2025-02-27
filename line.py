import cv2
import  numpy as np
import math
from utils_usb import *
###################################宏常量定义########################################

USB2_Width = 640
USB2_Height = 480

def UpContrast(image,a =1.5):
    """
    功能：提高对比度
    输入：图像矩阵，对比度增强比例a
    返回值：增强后的rgb图像
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 提高对比度
    hsv[:, :, 2] = cv2.convertScaleAbs(hsv[:, :, 2], alpha=a, beta=0)  # alpha 控制对比度，beta 控制亮度偏移
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)  # 保证明度不超出范围
    # 将图像从 HSV 转换回 BGR
    output_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return output_image
def UpFull(image,a = 1.8):
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

def GetROI(frame, left, right, up, down):
    """
    功能：提取ROI
    输入：图像矩阵,上下左右区域值
    返回值：ROI区域
    """
    # 检查坐标是否在合法范围内
    if left < 0 or right > USB2_Width or up < 0 or down > USB2_Height or left >= right or up >= down:
        raise ValueError("Invalid ROI coordinates")
    
    #mask = np.zeros_like(frame)
    #mask[up:down, left:right] = 255
    # 获取 ROI 区域
    #ROI = cv2.bitwise_and(mask,frame)
    ROI = frame[up:down, left:right]
    return ROI

#霍夫直线检测
def hourf_line(image):
    """
    功能：对二值化图像边缘检测后，直线霍夫检测
    输入：二值化图像，注意修改参数，canny的参数和houf直线检测的参数
    返回值：边缘检测后的图像，直线检测后的直线对象
    """
    # 边缘检测（使用 Canny 算法）
    edges = cv2.Canny(image,60,150)
    cv2.imshow("edges",edges)
    
    # 霍夫变换检测直线
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=150, minLineLength=50, maxLineGap=10)
    
    return ProcessLine(lines)
    
    
#两点式拟合直线
def Fitline(x1,y1,x2,y2,x):
    """
    功能通过两点 (x1, y1) 和 (x2, y2) 拟合直线，并计算给定 y 值对应的 x 坐标。
    
    参数：
    x1, y1: 第一个点的像素坐标
    x2, y2: 第二个点的像素坐标
    y: 给定的 y 值（垂直坐标），我们要找出对应的 x
    
    返回：
    计算得到的 x 值。
    """
    # 计算直线的斜率 m
    if x1 == x2:  # 处理垂直线的特殊情况
        raise ValueError("cannot calculate K")
    m = (y2 - y1) / (x2 - x1)
    
    return y1 +m*(x-x1)

def ProcessLine(lines):
    """
    功能：直线处理函数，将直线的信息处理为控制量
    输入：图像中的直线组
    输出：对应于每个直线的：1.端点坐标 2.图像中心点与ROI坐标之差 3.直线偏转角度 水平为0
    
    """
    lines_info = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            theta = math.atan(((y2-y1)/(x2-x1)))*180/np.pi
            aim_x = Fitline(x1,y1,x2,y2,200)
            lines_info.append([x1,y1,x2,y2,theta,aim_x])
        #print(f"{lines_info[0]=}")
        return lines_info[0]
    else: return None



#基础寻迹代码
def FollowLine(frame,aim_x):
    """
    功能：基础寻迹，灰色和黄色
    输入：图像帧,期望的对齐数据(当基础巡线时保持一个安全的距离，当任务巡线时控制一个较为接近的距离)
    输出：直线偏移角度，直线中心位置
    """
    lines_info =None
    #对比度和饱和度
    uc = UpContrast(frame)
    uf = UpFull(uc)
    #颜色mask提取
    mask_yellow = color_detect(uf,"ground_yellow")
    mask_gray = color_detect(uf,"ground_gray")
    result = np.zeros_like(uf)
    #mask是个单通道二值化图像
    result[mask_yellow > 0] = [255, 255, 255]  # 黄色为白色
    result[mask_gray > 0] = [0, 0, 0]  # 灰色为黑色
    kernel = np.ones((5, 5), dtype=np.uint8)
    erosion = cv2.erode(result, kernel, iterations=1)
    kernel = np.ones((12, 12), dtype=np.uint8)
    dilate = cv2.dilate(erosion, kernel, 2) # 1:迭代次数，也就是执行几次膨胀操作
    ROI = GetROI(dilate,120,520,200,480)
    lines_info = hourf_line(ROI)
    ROI= cv2.cvtColor(ROI,cv2.COLOR_BGR2GRAY)
    #print(lines_info)
    #cv2.imshow("roi",ROI)
    if lines_info is not None:
        e_x = aim_x-lines_info[5]
        print(f"{e_x=}","\n",f"{lines_info[4]=}")
        
        return e_x,lines_info[4]#返回偏差值，以及角度的偏移值
    else:
        return None
    
