#import rclpy                            # ROS2 Python接口库
#from rclpy.node import Node             # ROS2 节点类
#from sensor_msgs.msg import Image       # 图像消息类型
#from cv_bridge import CvBridge          # ROS与OpenCV图像转换类
#from std_msgs.msg import Float64        #发布浮点数据
#from rcl_interfaces.msg import ParameterDescriptor

#引入其他函数
import sys
sys.path.append('/home/yyh/dev_ws/src/yyh_image/yyh_image/')

from utils_usb import *
from filter import *
from Code_2D import *
from line import *
from circle import *
from Getitem import *
from Myserial import *
import cv2                              # Opencv图像处理库
import numpy as np                      # Python数值计算库

from pyzbar import pyzbar

"""如何自动化测试？
核心思想就是自动喂给程序一些输入，观测程序的输入和输出是否与预想的程序一致

"""
#创建一个订阅者节点

class ImagePublisher():
    #构造函数，初始化所有变量
    def __init__(self, name):
        #super().__init__(name)#继承  

        #状态机初始化并打印
        #self.task_state=0;PrintState(self.task_state)
        self.task_state=2;PrintState(self.task_state)
        self.IFLine = 0#先不巡线检测
        #镜头初始化
        self.usb1 = VideoStream(0)#机械臂摄像头
        self.usb2 = VideoStream(2)#定位规划摄像头

        #串口同类类
        #self.ser = SerialPort()
        #x,y方向的偏差
        self.e_x = 0
        self.e_y = 0
        self.ab_filter = ab_filter()#一阶滤波器
        self.kalmen_usb1 = KalmanFilter_circle()#圆形的卡尔曼滤波器
        self.kalmen_usb2 = kalman_filter()#实例化kalmen滤波器
        #用于保存小车领取到的任务
        self.task_list = None
      
        self.CarPlate = {1:None,2:None,3:None}
        self.task_id = 1
        self.task_complete = 0#小车对上位机的反馈
        self.ticks = 0
        self.ProcessImage()

        
        
    
    def ProcessImage(self):
        #开始利用状态机控制两个摄像头的不同工作模式
        ind = 0
        while True:
            precTick = self.ticks
            self.ticks = float(cv2.getTickCount())
            self.dT = float((self.ticks - precTick)/cv2.getTickFrequency())
            ind+=1
            frame1 = self.usb1.read()
            frame2 = self.usb2.read()
            #巡线逻辑可以加在这里
            #需要巡线调整之前，在此处
            self.StateControl(ind,frame1,frame2)
            if(not ShowCV("frame",np.hstack((frame1,frame2)))):
                break
            
                
    #状态机控制函数        
   
    def StateControl(self,ind,frame1,frame2):
        
        with timer(ind):
            match self.task_state:
                case 0:
                    self.GoOut(frame1,frame2)
                case 1:
                    self.GetFromPlate(frame1,frame2,1)#从圆环中拿物料
                case 2:
                    self.PutIntoCircle(frame1,frame2)#放到粗加工区
                case 3:
                    self.GetFromCircle(frame1,frame2)#从粗加工区获取
                case 4:
                    self.PutIntoCircle(frame1,frame2)#放到暂存区域
                case 5:
                    self.GetFromPlate(frame1,frame2,2)#从圆环中拿第二批物料
                case 6:
                    self.PutIntoCircle(frame1,frame2)#第二批放到粗加工区
                case 7:
                    self.GetFromCircle(frame1,frame2)#从粗加工区获取第二批
                case 8:
                    self.PutIntoCircle(frame1,frame2)#放到暂存区域码垛
                case 9:
                    self.Return(frame1,frame2)

    #状态机0：
    def GoOut(self,frame1,frame2):
        """
        usb1:暂时关闭，无功能
        usb2:寻迹，寻找二维码
        寻找到二维码后，更新任务
        """
        #二维码
        
        img=cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)#转换成灰度图
        barcodes = pyzbar.decode(img)
        results_2d= Code2D(frame2,barcodes,self.dT,self.kalmen_usb2)
        if(len(results_2d)>0):

            self.e_x = results_2d[1]

            
            if(is_data_stable(self.e_x,0,threshold=0.05)):#如果二维码固定且稳定，说明小车成功对齐了二维码
                self.kalmen_usb2.reinitialize_kf()
                self.task_state=1;PrintState(self.task_state)
            
            #self.publisher_.publish(msg)
            if(self.task_list is None and results_2d[0] is not None):
                self.task_list=StringtoTask(results_2d[0])
                print(self.task_list)
           
           

    def GetFromPlate(self,frame1,frame2,flag):
        """
        usb1 :寻找颜色，对准圆形的中心
        usb2: 寻找颜色，对准方形的中心
        完成抓取任务
        下一状态条件：ifcomplete从串口得到三次放置成功
        下一状态：转移到粗加工区
        """
        if flag == 1:
            task = self.task_list[0:3]
        else:
            task = self.task_list[3:6]
        

        list_usb1 = GetCenterColor_usb1(frame1,self.kalmen_usb1,self.dT)#找目标机械臂
        list_usb2 = GetCenterColor_usb2(frame2,self.kalmen_usb2,self.dT)#找目标侧视镜头
        
        if(len(list_usb2)>0 and len(list_usb1)>0):
            #if(is_data_stable(list_usb2[1],0) and task[self.task_id-1]==list_usb2[0] and is_data_stable(list_usb1[2],0)):#二者能匹配上
            if(is_data_stable(list_usb2[1],0) and task[self.task_id-1]==list_usb2[0]):#二者能匹配上    
                print(f"添加颜色{list_usb2[0]}")
                print(f"串口通信，抓取{list_usb2[0]}的物料")
                
                self.CarPlate[self.task_id] = list_usb2[0]
                print(f"{self.CarPlate=}")
                self.task_id +=1
                self.task_complete  +=1
        if(self.task_complete==3):#状态机切换
            self.kalmen_usb2.reinitialize_kf()#卡尔曼滤波器恢复原状。
            self.task_id = 1#控制计数id恢复为0
            self.task_complete = 0
            self.task_state+=1;PrintState(self.task_state)

        
        #第一步：全部检测，传入图像，反馈目标坐标，
    def PutIntoCircle(self,frame1,frame2):
        """
        功能，对准色环circle，并放置
        usb1:反馈e_x ,e_y,利用两个维度对准
        记录好放置粗加工区的条件
        下一状态条件：ifcomplote从串口得到三次放置成功
        下一状态：按顺序抓取三个物料，代码逻辑同state——2
        """
        
        list_usb1 = GetCenterColor_usb1(frame2,self.kalmen_usb1,self.dT)#找目标机械臂
        
        
        if len(list_usb1)>0 :
            X,Y =GetWorldPosition(list_usb1[1],list_usb1[2],100)
            #print("X:",X,"Y:",Y,"mm")
            cv2.putText(frame2, "center:("+str(int(X))+","+str(int(-Y))+"mm"+")", (5,40), cv2.FONT_HERSHEY_SIMPLEX,.9, (0, 0, 255), 2)
            if list_usb1[0] == 'green' and is_data_stable(list_usb1[2],0):#二者能匹配上
                print(f"添加颜色绿色")
                print(f"串口通信，对准了绿色的圆环")
        """
        串口发一个数，让我进入下一个状态。
        """
    
    def GetFromCircle(self,frame1,frame2):
        """
        从圈中拿物料，需要对齐最中央的物料，绿色，然后任意顺序都可以
        我只反馈中间的数据其余的问题是单片机的活
        结束条件，完成任务指令接受
        """
        list_usb1 = GetCenterColor_usb1(frame1,self.kalmen_usb1,self.dT)#找目标机械臂
        
        
        if(len(list_usb2)>0 and len(list_usb1)>0):
            X,Y =GetWorldPosition(list_usb1[1],list_usb1[2],100)
            #print("X:",X,"Y:",Y,"mm")
            cv2.putText(frame1, "center:("+str(int(X))+","+str(int(-Y))+"mm"+")", (5,40), cv2.FONT_HERSHEY_SIMPLEX,.9, (0, 0, 255), 2)
            if list_usb1[0] == 'green' and is_data_stable(list_usb1[2],0):#二者能匹配上
                print(f"添加颜色绿色")
                print(f"串口通信，对准了绿色的圆环")
        pass
    
    def Return(self,frame1,frame2):
        """
        功能：对准大地边缘，进行回位矫正
        """
        pass
        
    
    def __del__(self):#析构函数
        # 释放摄像头和关闭窗口
        self.usb1.Release()
        self.usb2.Release()
        cv2.destroyAllWindows()



node = ImagePublisher("Image")  # 创建ROS2节点对象并进行初始化
