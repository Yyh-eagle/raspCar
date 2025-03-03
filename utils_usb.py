import cv2
import subprocess
import time
import  numpy as np
import math
import collections
from threading import Thread
from queue import Queue
from sklearn.cluster import KMeans
###################################宏常量定义########################################

USB2_Width = 640
USB2_Height = 480
lower_red1 = np.array([0, 50, 100])      # 红色的HSV阈值下限1
upper_red1 = np.array([10, 255, 255])    # 红色的HSV阈值上限1
lower_red2 = np.array([170, 50, 100])    # 红色的HSV阈值下限2
upper_red2 = np.array([180, 255, 255])   # 红色的HSV阈值上限2
lower_blue = np.array([100,50,100])#蓝色下限
upper_blue = np.array([124, 255, 255])#蓝色上限
lower_green = np.array([35,43,46])#绿色下限
upper_green = np.array([77, 255, 255])#绿色上限
lower_ground_yellow = np.array([15, 5, 20])   # 地面黄色的HSV阈值下限
upper_ground_yellow = np.array([45, 255, 255])   # 地面黄色的HSV阈值上限
lower_ground_gray = np.array([0, 0, 46])   # 地面灰色的HSV阈值下限
upper_ground_gray = np.array([180, 40, 254])   # 地面灰色的HSV阈值上限


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
#2.稳定判据：
def is_data_stable(sensor_data, aim = 0,threshold=0.25, frame_count=10):
    """
    当传感器返回数据为静态时，通过连续10（可改）帧的稳定判断来确保数据的稳定性，防止对动态目标的误识别,
    输入：传感器数据,期望值，变化阈值，连续判断帧数,
    输出：是否可以将其作为固定目标，（物料和圆环）

    """
   
    # 缓存最近 frame_count 帧的数据
    if not hasattr(is_data_stable, "data_buffer"):
        is_data_stable.data_buffer = collections.deque(maxlen=frame_count)#创建一个队列
    
    # 使用nonlocal来确保在函数内记录和修改max_data的值
    if not hasattr(is_data_stable, "max_data"):  # 初始化max_data
        is_data_stable.max_data = None

    # 更新max_data，记录最大传感器数据
    if is_data_stable.max_data is None:
        is_data_stable.max_data = sensor_data
    else:
        is_data_stable.max_data = max(abs(sensor_data), is_data_stable.max_data)

    # 添加当前传感器数据
    is_data_stable.data_buffer.append(sensor_data)
    
    # 检查数据是否已经积累到足够的帧数
    if len(is_data_stable.data_buffer) < frame_count:
        return False  # 数据还不够稳定
    

    # 计算第 10 帧和第 1 帧之间的偏差
    first_frame = is_data_stable.data_buffer[0]
    tenth_frame = is_data_stable.data_buffer[-1]
    deviation = abs(tenth_frame - first_frame)
    differences = [abs(is_data_stable.data_buffer[i] - aim) for i in range(frame_count-1)]
    
    
    #data_mean = sum(is_data_stable.data_buffer) / len(is_data_stable.data_buffer)#计算均值
    # 如果偏差小于阈值，则认为数据稳定
    if all(diff < threshold*is_data_stable.max_data for diff in differences) and (deviation < threshold*is_data_stable.max_data):#1%误差允许
        return True
    
    return False
#3.状态机打印
def PrintState(State):
    match State:
        case 0: print("状态机0: ","\n","usb1:暂时关闭，无功能","\n","usb2:寻迹，寻找二维码")
        case 1: print("状态机1: ","\n","usb1:在运行至物料区附近时打开，由里程计决定，打开后寻找要对齐的圆形，和颜色","\n","usb2:寻迹，寻找最大的中间物料并对齐")
        case 2 :print("状态机2: ","\n","usb1:在运行至暂存区附近时打开，由里程计决定，打开后对齐中间圆环","\n")


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

   
def CaculateFeedback_X(K,cx):#根据矩形框的中心点来计算反馈量
#K是反馈系数，用于将像素转换为真实的坐标
    return ((USB2_Width/2)-cx)

def CaculateFeedback_Y(K,cy):#根据矩形框的中心点来计算反馈量
#K是反馈系数，用于将像素转换为真实的坐标
    return ((USB2_Height/2)-cy)*K

###########################################圆形与中心检测##############################################
def ContourFilter(image):
    minRadius=25
    maxRadius=600
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:#传递到max函数中的轮廓不能为空
        #print("未找到轮廓")
        return None
    max_cnt = max(contours , key = cv2.contourArea)#找到轮廓中最大的一个
    area = cv2.contourArea(max_cnt)
    if area < (minRadius**2) * math.pi or area > (maxRadius**2) * math.pi:
        #print("轮廓不符合要求")
        return None
    return max_cnt

def cnts_draw(img,res,color):
    """

    功能：在原图像上绘出指定颜色的轮廓。返回卡尔曼滤波状态变量
    img：原图像；res：只剩某颜色的位与运算后的图像；color：指定的颜色
    返回值：颜色的中心
    """
    #这两个参数是设定颜色块的最大半径和最小半径，去除噪声干扰
    #print(color)
    minRadius=100
    maxRadius=1200
    canny = cv2.Canny(res,170,220)#Canny边缘检测算法，用来描绘图像中物体的边缘，（100，200为此函数的两个阈值，该阈值越小轮廓的细节越丰富）
    contours, hierarchy=cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#寻找图像轮廓的函数，这里先用Canny算法得到只保留轮廓的图像方便轮廓的找寻
    
    if len(contours) == 0:#传递到max函数中的轮廓不能为空
        #cv2.imshow('video',img)
        return []
    else:
        
        max_cnt = max(contours , key = cv2.contourArea)#找到轮廓中最大的一个
        cv2.drawContours(img, max_cnt,-1,(0,255,0),2)#在原图上绘制这个最大轮廓
        (x,y,w,h) = cv2.boundingRect(max_cnt)#找到这个最大轮廓的最大外接矩形，返回的（x，y）为这个矩形右下角的顶点，w为宽度，h为高度
        area = cv2.contourArea(max_cnt)
        
        if area < (minRadius**2) * math.pi or area > (maxRadius**2) * math.pi:
        #    print("轮廓不合适")
            return []
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)#在原图上绘制这个矩形
        return [x,y,w,h]

#计算掩膜面积
def get_area(mask):
    return np.sum(mask > 0)


#感兴趣的区域
def GetROI(frame, left, right, up, down):
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

    match color:
        case "red":
            mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)
        case "ground_yellow":mask = cv2.inRange(hsv_img, lower_ground_yellow, upper_ground_yellow)
        case "ground_gray":mask = cv2.inRange(hsv_img, lower_ground_gray, upper_ground_gray)
        case "blue":mask = cv2.inRange(hsv_img, lower_blue, upper_blue)
        case "green":mask = cv2.inRange(hsv_img, lower_green, upper_green)
   
    return mask

#颜色的聚类，目前我还没有想好怎么用
def get_dominant_color(image, k=3):
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(image)
    dominant_colors = kmeans.cluster_centers_
    return dominant_colors


FX = 445.802830580577 #相机1的水平焦距
FY = 446.380440858884
CX =304.242366846951
CY =231.655409240974
#小孔成像原理
def GetWorldPosition(x,y,h):
     # 小孔成像逆变换计算X/Y坐标
    X = (x - CX) * h / FX
    Y = (y - CY) * h / FY
    # Z坐标为固定高度h（假设地面平面Z=h）
    Z = h  
    return (X,Y)
