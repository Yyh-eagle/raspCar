import cv2
import subprocess
import time
import  numpy as np
import math
import collections
from threading import Thread
from queue import Queue
from collections import deque
from collections import Counter

###################################宏常量定义########################################
FX = 498.043256027757 #相机1的水平焦距
FY = 498.108657044219
CX =315.215468760492
CY =238
USB2_Width = 640
USB2_Height = 480
lower_red1 = np.array([0, 52, 83])      # 红色的HSV阈值下限1
upper_red1 = np.array([15, 255, 255])    # 红色的HSV阈值上限1
lower_red2 = np.array([144, 52, 83])    # 红色的HSV阈值下限2
upper_red2 = np.array([180, 255, 255])   # 红色的HSV阈值上限2
lower_blue = np.array([60,83,38])#蓝色下限
upper_blue = np.array([114, 255, 255])#蓝色上限
lower_green = np.array([37,33,39])#绿色下限
upper_green = np.array([83, 255, 255])#绿色上限
lower_ground_yellow = np.array([19, 13, 113])   # 地面黄色的HSV阈值下限
upper_ground_yellow = np.array([47, 99, 232])   # 地面黄色的HSV阈值上限
lower_ground_gray = np.array([0, 0, 46])   # 地面灰色的HSV阈值下限
upper_ground_gray = np.array([180, 40, 254])   # 地面灰色的HSV阈值上限

lower_ground_green = np.array([35,13,89])   # 地面lv色的HSV阈值下限
upper_ground_green = np.array([74, 197, 255])   # 地面lv的HSV阈值上限
lower_ground_red1 = np.array([0, 52, 83])      # 地面红色的HSV阈值下限1
upper_ground_red1 = np.array([15, 255, 255])    # 地面红色的HSV阈值上限1
lower_ground_red2 = np.array([144, 52, 83])    # 地面红色的HSV阈值下限2
upper_ground_red2 = np.array([180, 255, 255])   # 地面红色的HSV阈值上限2
lower_ground_blue = np.array([60,83,38])   # 地面蓝色的HSV阈值下限
upper_ground_blue = np.array([114, 255, 255])   # 地面蓝色的HSV阈值上限
####################################辅助工具函数#######################################
#1.计时器函数
from contextlib import  contextmanager
@contextmanager
def timer(ind):
    #利用上下文管理器进行时间计数
    start = time.time()
    yield
    end = time.time()
    if ind%50==0:
        print(f"耗时：{end-start:.4f}秒")
#稳定性判别器
class data_define():
    def __init__(self):
        self.ind  =0
    def define(self,data,aim,threshold=40):
        if(abs(data-aim)<=threshold):
            self.ind+=1
        else:
            self.ind=0
        print(self.ind)
        if self.ind >=6:
            return True
        
        else:
            return False

#颜色滤波器
class ColorFilter:
    def __init__(self, window_size=3):
        self.window_size = window_size
        self.window = deque(maxlen=window_size)  # 固定大小的滑动窗口
    
    def update(self, new_color):
        """添加新颜色，并返回滤波后的颜色（众数）"""
        self.window.append(new_color)
        return self.get_mode()
    
    def get_mode(self):
        """计算当前窗口的众数（出现次数最多的颜色）"""
        if not self.window:
            return None
        
        # 统计颜色出现次数
        color_counts = Counter(self.window)
        
        # 返回出现次数最多的颜色
        mode_color = color_counts.most_common(1)[0][0]
        return mode_color

    def reset(self):
        """清空窗口"""
        self.window.clear()



   
   

#4.计数器类
class CallingCounter(object):
    def __init__ (self, func):
        self.func = func# 需要计数的函数
        self.count = 0

    def __call__ (self, *args, **kwargs):
        self.count += 1
        return self.func(*args, **kwargs)
#5.cv图像展示与退出
def ShowCV(name,frame):
    #展示两个摄像头的原始图像
    cv2.imshow(name,frame)
    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False
    else: return True
#6.非阻塞视频流
class VideoStream:
    def __init__(self,src = 0):
        self.name = f"镜头{src}"
        self.stream = cv2.VideoCapture(src)
        # 尝试禁用自动白平衡（AWB）
        self.stream.set(cv2.CAP_PROP_AUTO_WB, 1)  # 0 = 关闭自动白平衡
      
        self.q  =Queue(maxsize=10)
        self.thread = Thread(target=self.update,args=())#生产者线程
        self.thread.daemon =True#守护线程，主程序退出时自动结束
        self.thread.start()#启动线程

    def update(self):
        while 1:
            time.sleep(0.05)
            ret,frame = self.stream.read()#从摄像头读取帧
            if not ret:
                print(self.name,"出现问题")
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()#如果队列已经满，丢弃最旧的帧
                except Queue.Empty:
                    pass
            self.q.put(frame)#将新帧放入队列中

    def read(self):
        return self.q.get()
    
    def Release(self):
        self.stream.release()
#根据颜色转换回数据
def ColorToNum(color):
    if(color == 'red'):
        return 1
    elif(color == 'green'):
        return 2
    elif(color == 'blue'):
        return 3
    else:
        return 0
        


###########################################圆形与中心检测##############################################
def ContourFilter(image):
    minRadius=25
    maxRadius=600
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #寻找轮廓长度>20的轮廓

    if len(contours) == 0:#传递到max函数中的轮廓不能为空
        #print("未找到轮廓")
        return None
    max_cnt = max(contours , key = cv2.contourArea)#找到轮廓中最大的一个

    return max_cnt

def cnts_draw(img,res):
    
    bordered = cv2.copyMakeBorder(res, 2,2,2,2, cv2.BORDER_CONSTANT, value=0)
    canny = cv2.Canny(res,200,250)#Canny边缘检测算法，用来描绘图像中物体的边缘，（100，200为此函数的两个阈值，该阈值越小轮廓的细节越丰富）
    contours, hierarchy=cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#寻找图像轮廓的函数，这里先用Canny算法得到只保留轮廓的图像方便轮廓的找寻

    if len(contours) == 0:#传递到max函数中的轮廓不能为空
        return []
    else:
       
        max_cnt = max(contours , key = cv2.contourArea)#找到轮廓中最大的一个
        #print(cv2.contourArea(max_cnt))
        
        max_cnt_adjusted = max_cnt - [2,2]
        (x,y,w,h) = cv2.boundingRect(max_cnt_adjusted)#找到这个最大轮廓的最大外接矩形，返回的（x，y）为这个矩形右下角的顶点，w为宽度，h为高度
        

        return [x,y,w,h]

#计算掩膜面积
def get_area(mask):
    return np.sum(mask > 0)


#感兴趣的区域
def GetROI_mask(frame, left, right, up, down):
    """
    功能：提取ROI
    输入：图像矩阵,上下左右区域值
    返回值：ROI区域
    """
    # 检查坐标是否在合法范围内
    if left >= right or up >= down:
        raise ValueError("Invalid ROI coordinates")
    left = max(left,0)
    up =max(up,0)
    right = min(USB2_Width,right)
    down = min(USB2_Height,down)
    
    mask = np.zeros_like(frame)
    mask[up:down, left:right] = frame[up:down, left:right]
    #cv2.imshow("mask",mask)
    return mask
#感兴趣的区域
def GetROI2(frame, left, right, up, down):
    """
    功能：提取ROI
    输入：图像矩阵,上下左右区域值
    返回值：ROI区域
    """
    # 检查坐标是否在合法范围内
    if left >= right or up >= down:
        raise ValueError("Invalid ROI coordinates")
    left = max(left,0)
    up =max(up,0)
    right = min(USB2_Width,right)
    down = min(USB2_Height,down)
    return frame[up:down, left:right]
#腐蚀膨胀操作
def ErosAndDia(result):
    kernel = np.ones((5, 5), dtype=np.uint8)
    erosion = cv2.erode(result, kernel, iterations=1)
    kernel = np.ones((10, 10), dtype=np.uint8)
    dilate = cv2.dilate(erosion, kernel, 1) # 1:迭代次数，也就是执行几次膨胀操作
    return dilate


def color_detect(frame,color):
    """
    功能：颜色获取
    输入：图像矩阵，颜色选择
    返回值：掩膜矩阵
    """
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)      # 图像从BGR颜色模型转换为HSV模型
    img_blur = cv2.GaussianBlur(hsv_img, (5, 5), 0)
    #img_bilateral = cv2.bilateralFilter(img_blur, d=5, sigmaColor=150, sigmaSpace=150)
    
    if(color==1):#h红色
        
        mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
    elif(color==4):#地黄
        mask = cv2.inRange(img_blur, lower_ground_yellow, upper_ground_yellow)
    elif(color==5) :#地灰
        mask = cv2.inRange(img_blur, lower_ground_gray, upper_ground_gray)
    elif(color==3):#蓝色
        mask = cv2.inRange(hsv_img, lower_blue, upper_blue)
    elif(color==2):#绿色
        mask = cv2.inRange(hsv_img, lower_green, upper_green)
    elif(color==6):#地绿色
       # print("地绿色")
        mask = cv2.inRange(hsv_img, lower_ground_green, upper_ground_green)
    elif(color==7):#地红色
        mask1 = cv2.inRange(hsv_img, lower_ground_red1, upper_ground_red1)
        mask2 = cv2.inRange(hsv_img, lower_ground_red2, upper_ground_red2)
        mask = cv2.bitwise_or(mask1, mask2)
    elif(color ==8):#地蓝色
        mask = cv2.inRange(hsv_img, lower_ground_blue, upper_ground_blue)
    return mask





def GetCameraPosition(x,y,theta,task_state):
    if(task_state%3==1):
        return Ratio_plate(x,y,theta)
    else:
        return Ratio_Ground(x,y,theta)




def Ratio_Ground(x,y,theta):
    X = (x-326)/22.4
    Y = (y-CY)/22.4

    R = np.array([[np.cos(theta),np.sin(theta)],
                    [np.sin(theta),-np.cos(theta)]])
    X_0 =np.array([X,Y])
    X_World = np.dot(R,X_0)  
    #print(X_World)              
    return X_World

def Ratio_item(x,y,theta):
    X = (x-326)/32.2
    Y = (y-CY)/32.8
    R = np.array([[np.cos(theta),np.sin(theta)],
                    [np.sin(theta),-np.cos(theta)]])
    X_0 =np.array([X,Y])
    X_World = np.dot(R,X_0)  
    #print(X_World)              
    return X_World

def Ratio_plate(x,y,theta):
    X = (x-326)/66
    Y = (y-CY)/70
    R = np.array([[np.cos(theta),np.sin(theta)],
                    [np.sin(theta),-np.cos(theta)]])
    X_0 =np.array([X,Y])
    X_World = np.dot(R,X_0)  
    #print(X_World)              
    return X_World
