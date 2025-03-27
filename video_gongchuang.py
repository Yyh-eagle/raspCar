#引入其他函数
import sys
import os
import datetime
sys.path.append('/home/yyh/ros2_ws/src/yyh_image/yyh_image/')

#import matplotlib.pyplot as plt
#from matplotlib.animation import FuncAnimation

from utils_usb import *
from filter import *
from Code_2D import *
from line import *

from Getitem import *
from Myserial import *

import cv2                              
import numpy as np                      
from pyzbar import pyzbar



class Gongchuang():
    #构造函数，初始化所有变量
    def __init__(self, name):
        
        #状态机初始化
        self.task_state=1;PrintState(self.task_state)
    
        #镜头初始化
        self.usb1 = VideoStream('use_videos/usb1.avi')#机械臂摄像头
        
        #绘图初始化
        """
        self.plot_data = {
            'frame': [],
            'raw_x': [],
            'raw_y': [],
            'kf_x': [],
            'kf_y': [],
            'timestamp':  []
        }
        self.setup_realtime_plot()
        """
        #滤波器与误差反馈

        self.ticks = 0#计时变量，用于卡尔曼滤波
        self.kalmen_usb1 = KalmanFilter_circle()#圆形的卡尔曼滤波器
        self.kalmen_line = KalmanFilter_Line()  # 直线的卡尔曼滤波器
        self.ab = ab_filter()
        #任务规划部分
        self.task_list = None
        self.last_task_id = 1
        self.task_id = 1
        
        #处理函数
        self.ProcessImage()


    #核心处理函数
    def ProcessImage(self):
       
        ind = 0#用于效率分析
        while True:
           
            #time.sleep(0.5)
            precTick = self.ticks
            self.ticks = float(cv2.getTickCount())
            self.dT = float((self.ticks - precTick)/cv2.getTickFrequency())
            ind+=1
            
            with timer(ind):
                frame1 = self.usb1.read()
               #FollowLine(frame1,20)
                #print(frame1)
                if(self.task_state==1):
                    self.GetFromPlate(frame1,1)#从圆环中拿物料
                elif(self.task_state==2):
                    self.PutIntoCircle(frame1)#放到粗加工区
                elif(self.task_state==3):
                    self.GetFromCircle(frame1)#从粗加工区获取
                elif(self.task_state==4):
                    self.PutIntoCircle(frame1)#放到暂存区域
                elif(self.task_state==5):
                    self.GetFromPlate(frame1,2)#从圆环中拿第二批物料
                elif(self.task_state==6):
                    self.PutIntoCircle(frame1)#第二批放到粗加工区
                elif(self.task_state==7):
                    self.GetFromCircle(frame1)#从粗加工区获取第二批
                elif(self.task_state==8):
                    self.PutIntoCircle(frame1)#放到暂存区域码垛
                elif(self.task_state==9):
                    self.Return(frame1)
                ShowCV("frame1",frame1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # 关闭所有窗口
        cv2.destroyAllWindows()

           
#状态机1
    def GetFromPlate(self,frame1,flag):
 
        #任务控制
        self.task_list = [1,2,3,1,3,2]
        if flag == 1:
            task = self.task_list[0:3]
        else:
            task = self.task_list[3:6]
        cv2.putText(frame1, str(task[self.task_id-1]), (600,400), cv2.FONT_HERSHEY_SIMPLEX,.9, (0, 0, 0), 2)
        
        #angle_car =Line_Angle(GetROI(frame1,0,200,0,480))
        #print(f"{angle_car=}")
        #得到目标信息
        list_usb1 = GetCenterColor_usb1(frame1,self.kalmen_usb1,self.dT,self.task_id)#找目标机械臂
        #print(list_usb1)
        line = detect_gray_yellow_boundary(frame1,self.kalmen_line,self.dT)
        #if angle is not None:
            #print(angle*57.3)
        angle =1.54
        if(len(list_usb1)>0) and angle is not None :
            #self.record_data(list_usb1)
            #self.update_realtime_plot()
            X,Y =GetWorldPosition(list_usb1[1],list_usb1[2],220,angle)
            cv2.putText(frame1, "center:("+str(int(X))+","+str(int(Y))+"mm"+")", (5,40), cv2.FONT_HERSHEY_SIMPLEX,.9, (0, 0, 255), 2)

            data_num = [ColorToNum(list_usb1[0]),X,Y]
           
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # 空格键触发抓取
            self.task_id +=1


        if(self.task_id==4):#如果三个物料都成功抓取
            
            self.task_id = 1#控制计数id恢复为0
            self.task_state+=1;PrintState(self.task_state)
            self.kalmen_usb1.reinitialize_kf()
        
        #第一步：全部检测，传入图像，反馈目标坐标，
    def PutIntoCircle(self,frame1):
        
        angle = 1.57
        
        #angle = Line_Angle(frame1)  # 直线检测函数
        list_usb1 = GetCenterColor_usb1(frame1,self.kalmen_usb1,self.dT,2)#找机械臂寻找目标
        
        if len(list_usb1)>0 and angle is not None :
            #画图
            #self.record_data(list_usb1)
            #self.update_realtime_plot()
            
            X,Y =GetWorldPosition(list_usb1[1],list_usb1[2],100,angle)
            cv2.putText(frame1, "center:("+str(int(X))+","+str(int(Y))+"mm"+")", (5,40), cv2.FONT_HERSHEY_SIMPLEX,.9, (0, 0, 255), 2)
   
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # 空格键触发全部抓取成功
            self.task_id = 1#控制计数id恢复为0
            self.task_state+=1;PrintState(self.task_state)
            self.kalmen_usb1.reinitialize_kf()


    def GetFromCircle(self,frame1):
        """
        从圈中拿物料，需要对齐最中央的物料，绿色，然后任意顺序都可以
        我只反馈中间的数据其余的问题是单片机的活
        结束条件，完成任务指令接受
        """
        cv2.putText(frame1, "get items", (550,400), cv2.FONT_HERSHEY_SIMPLEX,.9, (0, 0, 0), 2)

        angle = Line_Angle(frame1)  # 直线检测函数
        list_usb1 = GetCenterColor_usb1(frame1,self.kalmen_usb1,self.dT)#找机械臂寻找目标
        
        if len(list_usb1)>0 and angle is not None :
            #画图
            #self.record_data(list_usb1)
            #self.update_realtime_plot()
            
            X,Y =GetWorldPosition(list_usb1[1],list_usb1[2],100,angle)
            cv2.putText(frame1, "center:("+str(int(X))+","+str(int(Y))+"mm"+")", (5,40), cv2.FONT_HERSHEY_SIMPLEX,.9, (0, 0, 255), 2)
            data_num = [ColorToNum(list_usb1[0]),X,Y] 
            
        #    if(is_data_stable(X) and is_data_stable(Y)and list_usb1[0]=="green"):
        #        self.serial.Send_message(data_num,1)
        #    else:
        #        self.serial.Send_message(data_num,0)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # 空格键触发全部抓取成功
            self.task_id = 1#控制计数id恢复为0
            self.task_state+=1;PrintState(self.task_state)
            self.kalmen_usb1.reinitialize_kf()

        #if(len(self.serial.data)>0):
        #    if(self.serial.data[0] == 1):#如果下方上行数据为1
        #        self.task_id = 1#控制计数id恢复为0
        #        self.task_state+=1;PrintState(self.task_state)
        #       self.kalmen_usb1.reinitialize_kf()
            
    
    def Return(self,frame1):
        """
        功能：对准大地边缘，进行回位矫正
        """
        pass
        
    
    def setup_realtime_plot(self):
        """初始化实时绘图窗口"""
        plt.ion()  # 开启交互模式
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # X坐标子图
        self.line_raw_x, = self.ax1.plot([], [], 'r-', label='Raw X', alpha=0.5)
        self.line_kf_x, = self.ax1.plot([], [], 'b-', label='KF X')
        self.ax1.set_title('X Coordinate Comparison')
        self.ax1.set_xlabel('Frame')
        self.ax1.set_ylabel('Pixel')
        self.ax1.grid(True)
        self.ax1.legend()
        
        # Y坐标子图
        self.line_raw_y, = self.ax2.plot([], [], 'g-', label='Raw Y', alpha=0.5)
        self.line_kf_y, = self.ax2.plot([], [], 'm-', label='KF Y')
        self.ax2.set_title('Y Coordinate Comparison')
        self.ax2.set_xlabel('Frame')
        self.ax2.set_ylabel('Pixel')
        self.ax2.grid(True)
        self.ax2.legend()

    def record_data(self, pos):
        """记录当前帧数据"""
        self.plot_data['frame'].append(len(self.plot_data['frame'])+1)#####################
        self.plot_data['raw_x'].append(pos[3])
        self.plot_data['raw_y'].append(pos[4])
        self.plot_data['kf_x'].append(pos[1])
        self.plot_data['kf_y'].append(pos[2])
        self.plot_data['timestamp'].append(time.time())

    def update_realtime_plot(self):
        """更新实时曲线"""
        if len(self.plot_data['frame']) == 0:
            return
        
        # 更新X坐标
        self.line_raw_x.set_data(self.plot_data['frame'], self.plot_data['raw_x'])
        self.line_kf_x.set_data(self.plot_data['frame'], self.plot_data['kf_x'])
        self.ax1.relim()
        self.ax1.autoscale_view()
        
        # 更新Y坐标
        self.line_raw_y.set_data(self.plot_data['frame'], self.plot_data['raw_y'])
        self.line_kf_y.set_data(self.plot_data['frame'], self.plot_data['kf_y'])
        self.ax2.relim()
        self.ax2.autoscale_view()
        
        plt.pause(0.001)

    def __del__(self):#析构函数
        # 释放摄像头和关闭窗口
        self.usb1.Release()

        cv2.destroyAllWindows()



node = Gongchuang("Image")  # 创建ROS2节点对象并进行初始化
