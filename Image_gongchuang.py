#系统级别函数
import os
import datetime
import time

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
    
        self.iFvideo =1
        self.Init_video()
        self.serial = SerialPort()
        self.Init_chuankou()
        #镜头初始化
       # self.usb1 = VideoStream("use_videos/mature.avi")
        self.usb2 = VideoStream('/dev/camera_arm')
        self.usb1 = VideoStream('/dev/camera_front')

        self.ticks = 0#计时变量，用于卡尔曼滤波
        self.kalmen_line = KalmanFilter_Line()  # 直线的卡尔曼滤波器#todo直线卡尔曼滤波调参数
        self.ab_line = ab_filter()
        self.ab_x = ab_filter()
        self.ab_y = ab_filter()
        self.df_1 = data_define()
        self.df_2 = data_define()
        self.color_filter =ColorFilter(window_size=5)
        #任务规划部分
        #状态机初始
        self.mystate  =0
        self.task_state=0
        self.last_task_state = 0
        self.task_list = None
        self.last_task_id = 1
        self.task_id = 1

        self.last_color = 0
        self.if_color_change = 0
        self.index = 0#目标丢失计数#todo考虑那些状态变量不需要零阶保持
        self.flag_complete = 0
        self.start_time=None#计时，用于圆盘区域的第二个目标
        
        #主循环
        self.ProcessImage()
      
    #核心处理函数
    def ProcessImage(self):
 
       #angle = 90+set-(180-aim)
       #      = set-90+aim  
        while True:
            
            self.serial.receive()#串口接收数据

            #计算角度
            
            self.task_state = int(self.serial.IfArrive)#todo是否中途=0#todo怎么进入回家程序
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


            self.Update_chuankou()   
            #展示逻辑
            self.putText(frame1,frame2)
            Frame = np.hstack((frame1,frame2))
            cv2.imshow("Frame", Frame)
            cv2.moveWindow("Frame", 20, 80)  # 将窗口移动到屏幕左上角
            if(self.task_list is not None):
                display_task_window(self.text)
            else:
                display_task_window("NO task")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        exit(0) #程序退出#todo验证程序可否退出    


    #重要的巡线检测功能
    def Follow_Line(self,frame1):
        Line = detect_gray_yellow_boundary(frame1,self.kalmen_line,self.dT)#灰黄线 
        #巡线
        if Line is not None:
            if Line[2] is not None:
                #print(Line[2])
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
            if(results_2d[0] is not None):
                self.text = results_2d[2]
                self.task_list=text_to_array(results_2d[0])
                self.QRcode = array_to_int(self.task_list)

            cv2.putText(frame2, str(results_2d[1]), (320,240), cv2.FONT_HERSHEY_SIMPLEX,.9, (255,255,255), 2)
            self.object_X = (results_2d[1]-324)/22
                #self.stop_2d()#确定最后第二个摄像头是否要打开
 
       
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # 空格键触发抓取
            self.task_state =1        


        
#状态机1-8
    def Mainloop(self,frame1,frame2):
        #状态机切换，目标保持器计数清零
        if(self.last_task_state!=self.task_state):
            #稳定判据清零
            self.df_1.ind = 0
            self.df_2.ind = 0
            #任务计时的基准重置
            self.start_time = time.time()#任务的基准计时
            self.Is_aim_FLAG = 0
            self.if_color_change = 0
            self.flag_complete = 0
            #上一个颜色
            self.last_color =0
            self.IFloop =0.0#是否
            self.loop_flag = 0

 
 
        cv2.putText(frame2, "aim_color"+str(self.task_list[self.task_id-1]), (400,475), cv2.FONT_HERSHEY_SIMPLEX,.9, (0,0,255), 2)
        #清零所有的抓取动作
        self.Okseize = 0
        self.take_flag = 0.0

        # ################################执行巡黑线检测，矫正自身yaw角##########################################
        # angle = Line_Angle(frame1)
        # if angle is not None:#如果寻到线了
        #     self.line_flag = 1
        #     self.Car_Yaw  = angle #零阶保持
        # else:
        #     self.line_flag = 0
    
        #################################具体任务##########################################################
        if(self.task_state==1 or self.task_state==4):#从旋转圆盘中抓取物料
            
            list_usb1 = GetColor_usb1(frame1,self.task_list[self.task_id-1])#识别颜色物块
            if(len(list_usb1)>0):

                self.Is_aim_FLAG = list_usb1[0]#滑动窗口滤波
                self.index = 0
                self.object_X,self.object_Y =GetCameraPosition(list_usb1[3],list_usb1[4],1.2217305,self.task_state)#得到世界坐标

                
                if(self.Is_aim_FLAG!=self.last_color and self.last_color !=0):#产生了变化
                    self.if_color_change = 1#是否颜色改变#能进来，颜色变化了，并且初始颜色不等于0
                    self.start_time= time.time()#从颜色改变开始计时，每次颜色改变都计时！！！！！！！！！！！！！！！！
                

                self.this_time = time.time()

                if self.start_time is not None:
                    time_span = self.this_time-self.start_time
                else :
                    time_span = 0
                

                #识别到颜色改变
                if(self.flag_complete ==0 and self.if_color_change==1):#识别到颜色并下一个物料转动过来
                    self.IFloop =0.0
                    self.flag_complete = 1#标志颜色改变的位
               
                #新物料转动进来
                if(self.flag_complete ==1):
    
                    if(time_span>0.6):
                        self.IFloop =1.0
                    if(time_span>2.5):
                        self.IFloop = 0.0
                        if(self.Is_aim_FLAG==self.task_list[self.task_id-1]):
                            self.Okseize = 1.0
                        else:
                            self.Okseize = 0.0
                    else:
                        self.Okseize = 0.0
                else:
                        self.Okseize = 0.0
            else:
                self.index+=1
                
            if(self.Is_aim_FLAG !=0):
                self.last_color = self.Is_aim_FLAG 


#############################################放置圆环############################################################
        elif(self.task_state==2 or self.task_state==3 or self.task_state==5):#对准圆环，粗加工，暂存区，第二次粗加工

            list_usb1= []
            if(self.serial.color_loop is not None):
                list_usb1 = GetCenterColor_usb1_all(frame1,int(self.serial.color_loop)+5)#着重修改这个函数
                
            
            if(len(list_usb1)>0):#
                X,Y =GetCameraPosition(list_usb1[3],list_usb1[4],1.2217305,self.task_state)#todo看看是否一定会检测到直线
                self.object_X = X
                self.object_Y = Y
                self.Is_aim_FLAG = int(self.serial.color_loop)
                if(int(self.Is_aim_FLAG)!=int(self.last_color) ):#
                    self.loop_flag +=1
                    self.start_time= time.time()#从颜色改变开始计时，
                span_time = time.time()-self.start_time
                cv2.putText(frame1, str(self.last_color), (320,240), cv2.FONT_HERSHEY_SIMPLEX,.9, (255,255,255), 2)
                if((span_time)>=2):#是否有稳定的目标存在#todo调整参数
                    self.IFloop =1.0#闭环
                    if(self.loop_flag == 1):
                        if(span_time>8):#如果稳定了
                            self.take_flag = 1.0
                            self.IFloop =0.0#抓的时候让闭环=0
                        else:
                            self.take_flag = 0.0
                    else:
                        if(span_time>=4):#如果稳定了
                            self.take_flag = 1.0
                            self.IFloop =0.0#抓的时候让闭环=0
                        else:
                            self.take_flag = 0.0
                else:
                    self.take_flag = 0.0
            else:
                self.index+=1
            
            if(self.Is_aim_FLAG!= 0):#没检测到目标，is——aim——flabg
                self.last_color = self.Is_aim_FLAG

    ####################码垛#################################################################
        else:

            list_usb1 = GetColor_usb1_green(frame1,2)#着重修改这个函数#识别色块
            #看见了颜色色块
            if(len(list_usb1)>0):#是否检测到目标
                self.index = 0
                X,Y =GetCameraPosition(list_usb1[3],list_usb1[4],1.2217305,self.task_state)#假设都是直的
                
                self.object_X = X
                self.object_Y = Y
                
                if((time.time()-self.start_time)>3.5 and list_usb1[0] == 2):#是否有稳定的目标存在
                    self.IFloop =1.0
                    self.Is_aim_FLAG = list_usb1[0]
                    if (time.time()-self.start_time)>6.5 and self.df_1.define(list_usb1[3],324,30) and self.df_2.define(list_usb1[4],240,30) and list_usb1[0]==2:#如果稳定了
                        self.Okseize = 1.0
                        self.IFloop =0.0
                    else:
                        self.Okseize = 0.0
                else:
                    self.IFloop =1.0
            else:
                self.index+=1
                self.Is_aim_FLAG = self.task_list[self.task_id-1]#零阶保持     


        
        
        #print(self.task_state)
        ###########################任务控制############################################
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

        #记录上一个状态，以便于控制什么时候是状态机切换
        self.last_task_state = self.task_state
  
        #目标丢失后的处理
        if(self.index >=10):
            self.Is_aim_FLAG =0



      
    def Return(self,frame1):
        """
        功能：对准大地边缘，进行回位矫正
        """
        #回位矫正
        #看线条
        #todo 测试
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
        self.Car_Yaw = 1.57
        self.object_X =0.0
        self.object_Y =0.0
        self.object_Z =0.0
        self.Okseize = 0.0
        self.take_flag = 0.0
        self.Task_Data3 = 0.0
        self.Task_Data4 = 0.0
        self.Task_Data5 = 0.0
        self.Task_Data6 = 0.0

        

    #更新所有全局变量，并发送串口
    def Update_chuankou(self):
  
        self.object_X = self.ab_x.ab_filte(self.object_X)
        self.object_Y = self.ab_y.ab_filte(self.object_Y)
        
        self.datanum = [self.Is_aim_FLAG,self.line_flag,self.QRcode,self.color,self.H,self.IFloop,self.Car_Yaw,self.object_X,self.object_Y,self.object_Z,self.Okseize,self.take_flag,self.Task_Data3,self.Task_Data4,self.Task_Data5,self.Task_Data6]
        #print("串口发送数据",self.datanum)
        self.serial.Send_message(self.datanum)

    #打印信息
    def putText(self,frame1,frame2):
        cv2.putText(frame2, "this_state: "+str(self.task_state), (400,400), cv2.FONT_HERSHEY_SIMPLEX,.9, (0, 0, 255), 2)
        cv2.putText(frame1, "Is_aim_FLAG:"+str(self.Is_aim_FLAG), (0,25), cv2.FONT_HERSHEY_SIMPLEX,.9, (0, 255, 0), 2)
        cv2.putText(frame1, "IFloop:"+str(self.IFloop), (0,75), cv2.FONT_HERSHEY_SIMPLEX,.9, (0, 255, 0), 2)
        cv2.putText(frame1, "OKseize:"+str(self.Okseize), (0,125), cv2.FONT_HERSHEY_SIMPLEX,.9, (0, 255, 0), 2)
        cv2.putText(frame2, "Ifseize:"+str(self.serial.Ifseize), (0,25), cv2.FONT_HERSHEY_SIMPLEX,.9, (0, 255, 0), 2)
        cv2.putText(frame2, "Ifarrive:"+str(self.serial.IfArrive), (0,75), cv2.FONT_HERSHEY_SIMPLEX,.9, (0, 255, 0), 2)
        cv2.putText(frame1, "object("+str(int(self.object_X*100)/100)+","+str(int(self.object_Y*100)/100)+")", (350,25), cv2.FONT_HERSHEY_SIMPLEX,.9, (0, 0, 255), 2)
        cv2.putText(frame2, "angle:"+str(self.Car_Yaw*57.3), (0,125), cv2.FONT_HERSHEY_SIMPLEX,.9, (0, 255, 0), 2)
        cv2.putText(frame1, "IFCOLORCHANGE:"+str(self.if_color_change), (350,75), cv2.FONT_HERSHEY_SIMPLEX,.9, (0, 255, 0), 2)      
        cv2.putText(frame2, "QRcode:"+str(self.QRcode), (0,175), cv2.FONT_HERSHEY_SIMPLEX,.9, (0, 255, 0), 2)
        cv2.putText(frame1, "Color_loop:"+str(self.serial.color_loop), (350,125), cv2.FONT_HERSHEY_SIMPLEX,.9, (0, 255, 0), 2)      
        cv2.putText(frame2, "If_take :"+str(self.take_flag), (0,225), cv2.FONT_HERSHEY_SIMPLEX,.9, (0, 255, 0), 2)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
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


    #析构函数
    def __del__(self):
        # 释放摄像头和关闭窗口
        self.usb1.Release()
        cv2.destroyAllWindows()



node = SubscriberNode()  # 创建ROS2节点对象并进行初始化

 
 
