#系统级别函数
import os
import datetime


#引入其他文件中的函数
from utils_usb import *
from filter import *
from Code_2D import *
from line import *
from Getitem import *
from Myserial import *
from show import *
from angle_calculate import *

#计算功能库
import cv2                             
import numpy as np                     
from pyzbar import pyzbar



#订阅节点
class SubscriberNode():
    #构造函数，初始化所有变量
    def __init__(self):
    
        self.iFvideo =0
        self.Init_video()
        self.serial = SerialPort()
        self.Init_chuankou()
        #镜头初始化
        #self.usb1 = VideoStream("use_videos/usb.avi")
        self.usb1 = VideoStream('/dev/camera_arm')
        self.usb2 = VideoStream('/dev/camera_front')

        self.ticks = 0#计时变量，用于卡尔曼滤波
        self.kalmen_line = KalmanFilter_Line()  # 直线的卡尔曼滤波器#todo直线卡尔曼滤波调参数
        self.ab_line = ab_filter()
        self.ab_x = ab_filter()
        self.ab_y = ab_filter()
        #任务规划部分
        #状态机初始
        self.task_state=0
        self.task_list = None
        self.last_task_id = 1
        self.task_id = 1
        
        self.index = 0#目标丢失计数#todo考虑那些状态变量不需要零阶保持
        #主循环
        self.ProcessImage()
      
    #核心处理函数
    def ProcessImage(self):
       
        while True:
    
            self.serial.receive()#串口接收数据
            #卡尔曼曼滤波计数
            precTick = self.ticks
            self.ticks = float(cv2.getTickCount())
            self.dT = float((self.ticks - precTick)/cv2.getTickFrequency())
            #读取图像
            frame1 = self.usb1.read()
            frame2 =self.usb2.read()
            frame2 = cv2.flip(frame2, -1)
            #录视频
            if(self.iFvideo ==1):
                if frame1 is not None and self.writers['usb1'] is None:
                    self.init_writers(frame1)
                # 录制 frame1
                if frame1 is not None and self.writers['usb1'] is not None:
                    self.writers['usb1'].write(frame1)  # 将 frame1 写入视频

            #巡线逻辑，在任何时候都要巡线
            self.Follow_Line(frame1)#一直巡线
            #总任务处理逻辑
            if(self.task_state==0):
                self.GoOut(frame1,frame2)
            elif(self.task_state<9):   
                self.Mainloop(frame1,frame2)#从圆环中拿物料
            else:
                self.Return(frame1)      
            #每次循环结束后更新串口      
      
            self.Update_chuankou()   


            #展示逻辑
            self.putText(frame1,frame2)
            Frame = np.hstack((frame1,frame2))
            cv2.imshow("Frame", Frame)
            cv2.moveWindow("Frame", 0, 20)  # 将窗口移动到屏幕左上角

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        exit(0) #程序退出#todo验证程序可否退出    


    #重要的巡线检测功能
    def Follow_Line(self,frame1):
        Line = detect_gray_yellow_boundary(frame1,self.kalmen_line,self.dT)#灰黄线 
        #巡线
        if Line is not None:
            if Line[2] is not None:
                Line[3]=self.ab_line.ab_filte(Line[3])
                self.H = Line[3]
                self.line_flag = 1
                aim  = caculate_angle()#
                #aim  = caculate_angle(self.serial.D_yaw1,self.serial.D_yaw2)
                
            else:
                self.line_flag =0
               
        else:
            self.line_flag =0
            


            

#出发识别二维码
    def GoOut(self,frame1,frame2):
 
        results_2d= Code2D(frame2,pyzbar.decode(frame2))
        if(len(results_2d)>0):
            
            #任务获取函数
            if(self.task_list is None and results_2d[0] is not None):
                self.task_list=text_to_array(results_2d[0])
                #display_numbers(self.task_list)
                self.QRcode =1
                #self.stop_2d()#确定最后第二个摄像头是否要打开
                self.task_state=1;
       
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # 空格键触发抓取
            self.task_state =1        


        
#状态机1-8
    def Mainloop(self,frame1,frame2):
        
        #任务控制
      
        task = self.task_list
        cv2.putText(frame2, "aim_color"+str(task[self.task_id-1]), (400,75), cv2.FONT_HERSHEY_SIMPLEX,.9, (0,0,255), 2)
        
        self.Okseize = 0.0
        self.IFloop = 0.0
        ################################执行圆形检测##########################################
  
        #################################抓取圆盘任务
        if(self.task_state%4==1):#从旋转圆盘中抓取物料
            list_usb1 = GetColor_usb1(frame1,task[self.task_id-1])#找目标机械臂,为什么传入
         
            if(self.serial.IfArrive>0):
            
                
                #一开始看不见，只要出现就是转过来的
                if(len(list_usb1)>0):#是否检测到目标
                    self.index = 0

                    X,Y =GetCameraPosition(list_usb1[3],list_usb1[4],1.204,self.task_state)
                    self.object_X = X
                    self.object_Y = Y
                    self.object_Z = 0
                    
                    
                    if(if_data_stable(list_usb1[3])):#是否有稳定的目标存在
                        self.Is_aim_FLAG = list_usb1[0]
                        self.IFloop =1.0
                        
                        if(data_stable(list_usb1[3],324) and list_usb1[0]==task[self.task_id-1]):#如果稳定了
                            self.Okseize = 1.0
                            self.IFloop =0.0
                        else:
                            self.Okseize = 0.0
                    else:
                        self.IFloop =0.0
                else:
                    self.index+=1
                    self.Is_aim_FLAG = self.task_list[self.task_id-1]#零阶保持     
            else:
                pass
            
        elif(self.task_state%4==3):#抓起来物料
            list_usb1 = GetColor_usb1(frame1,2)#找只找绿色，
            #此时可以什么都不做，根据之前放进去的位置开环抓取物料
        else:#最难的任务，将物料放在物料或者圆环上面
            list_usb1 = GetCenterColor_usb1(frame1)#着重修改这个函数
            #if(self.serial.IfArrive>0):#到了之后
            self.Okseize = 0.0
            if self.serial.IfArrive>0:
          
                #一开始看不见，只要出现就是转过来的
                if(len(list_usb1)>0):#是否检测到目标
                    self.index = 0
                    X,Y =GetCameraPosition(list_usb1[3],list_usb1[4],1.204,self.task_state)#假设都是直的
                    
                    self.object_X = X
                    self.object_Y = Y
                    self.object_Z = 0
                    
                    if(if_data_stable(list_usb1[3])):#是否有稳定的目标存在#todo调整参数
                        self.Is_aim_FLAG = list_usb1[0]
                        self.IFloop =1.0
                        
             
                        if(is_data_stable(list_usb1[3],324) and list_usb1[0]==task[self.task_id-1]):#如果稳定了
                            self.Okseize = 1.0
                            self.IFloop =0.0
                        else:
                            self.Okseize = 0.0
                    else:
                        self.IFloop =1.0
                else:
                    self.index+=1
                    self.Is_aim_FLAG = self.task_list[self.task_id-1]#零阶保持     
            else:
                pass

        #目标丢失后的处理
        if(self.index >=10):
            self.Is_aim_FLAG =0
       

       
        if(int(self.serial.Ifseize)==0):
            self.task_id = 1
        elif(int(self.serial.Ifseize)==1):
            self.task_id = 2
        elif(int(self.serial.Ifseize)==2):
            self.task_id = 3
        elif(int(self.serial.Ifseize)==3):
            self.task_id = 4
        elif(int(self.serial.Ifseize)==4):
            self.task_id = 5
        elif(int(self.serial.Ifseize)==5):
            self.task_id = 6
        
        if self.task_id == 4 and self.last_task_id ==3:
            self.task_state+=1


  
        
      
    def Return(self,frame1):
        """
        功能：对准大地边缘，进行回位矫正
        """
        pass

   
        
        

    #############################################串口与全局变量#############################################
    #初始化所有全局变量
    def Init_chuankou(self):
        self.Is_aim_FLAG = 0
        self.line_flag =0
        self.QRcode = 0
        self.color = 0
        self.H = 0#摄像头的垂直距离
        self.IFloop = 0#
        self.Car_Yaw = 0.0
        self.object_X =0.0
        self.object_Y =0.0
        self.object_Z =0.0
        self.Okseize = 0.0
        self.Task_Data2 = 0.0
        self.Task_Data3 = 0.0
        self.Task_Data4 = 0.0
        self.Task_Data5 = 0.0
        self.Task_Data6 = 0.0

        

    #更新所有全局变量，并发送串口
    def Update_chuankou(self):
  
        self.object_X = self.ab_x.ab_filte(self.object_X)
        self.object_Y = self.ab_y.ab_filte(self.object_Y)
        
        self.datanum = [self.Is_aim_FLAG,self.line_flag,self.QRcode,self.color,self.H,self.IFloop,self.Car_Yaw,self.object_X,self.object_Y,self.object_Z,self.Okseize,self.Task_Data2,self.Task_Data3,self.Task_Data4,self.Task_Data5,self.Task_Data6]
        #print("串口发送数据",self.datanum)
        self.serial.Send_message(self.datanum)

    #打印信息
    def putText(self,frame1,frame2):
        cv2.putText(frame2, "this_state: "+str(self.task_state), (400,25), cv2.FONT_HERSHEY_SIMPLEX,.9, (0, 0, 255), 2)
        cv2.putText(frame1, "Is_aim_FLAG:"+str(self.Is_aim_FLAG), (0,25), cv2.FONT_HERSHEY_SIMPLEX,.9, (0, 255, 0), 2)
        cv2.putText(frame1, "IFloop:"+str(self.IFloop), (0,75), cv2.FONT_HERSHEY_SIMPLEX,.9, (0, 255, 0), 2)
        cv2.putText(frame1, "OKseize:"+str(self.Okseize), (0,125), cv2.FONT_HERSHEY_SIMPLEX,.9, (0, 255, 0), 2)
        cv2.putText(frame2, "Ifseize:"+str(self.serial.Ifseize), (0,25), cv2.FONT_HERSHEY_SIMPLEX,.9, (0, 255, 0), 2)
        cv2.putText(frame2, "Ifarrive:"+str(self.serial.IfArrive), (0,75), cv2.FONT_HERSHEY_SIMPLEX,.9, (0, 255, 0), 2)
        cv2.putText(frame1, "object("+str(int(self.object_X*100)/100)+","+str(int(self.object_Y*100)/100)+")", (350,25), cv2.FONT_HERSHEY_SIMPLEX,.9, (0, 0, 255), 2)

    ##############################################视频录制#############################################

    def Init_video(self):
        #视频初始化
        #生成录制视频的目录
        self.video_dir = os.path.expanduser("~/yyh_image/saved_videos")
        os.makedirs(self.video_dir, exist_ok=True)#生成目录
  
        # 生成带时间戳的文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_paths = {
            'usb1': os.path.join(self.video_dir, f'usb1_{timestamp}.avi'),
        }
        
        # 视频写入器（稍后初始化）
        self.writers = {
            'usb1': None,
        }
     
        
    #初始化视频写入器
    def init_writers(self, frame1):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 实例化视频编码器
        # 初始化usb1的写入器
        if frame1 is not None and self.writers['usb1'] is None:#第一帧，必须要求self.writers['usb1'] is None
            
            self.writers['usb1'] = cv2.VideoWriter(
                self.video_paths['usb1'], 
                fourcc, 
                20.0,  # 帧率（根据实际情况调整）
                (640, 480)
            )


    def stop_2d(self):

        self.usb2.Release()
        self.frame2 = None
        cv2.destroyAllWindows()

  
    #析构函数
    def __del__(self):
        # 释放摄像头和关闭窗口
        self.usb1.Release()
        cv2.destroyAllWindows()



node = SubscriberNode()  # 创建ROS2节点对象并进行初始化
