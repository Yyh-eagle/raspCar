import cv2
import numpy as np
stateSize = 6#滤波器的状态向量的维度，需要估计的变量的数量。
measSize = 4#滤波器的观测向量的维度，能够接收的观测数据的数量
coutrSize = 0#滤波器的控制向量的维度，即控制输入的维度
class kalman_filter():
    """
    功能：一维卡尔曼滤波
    输入：当前测量数据，xywh
    返回值：滤波后的数据xyvxvywh
    """
    def __init__(self):
        
        self.kf = cv2.KalmanFilter(stateSize,measSize,coutrSize)#实例化卡尔曼滤波器，滤波器的各个维度如上定义
        self.Matrix_define()#矩阵初始化
        self.found =False
        self.notFoundCount = 0

    def Matrix_define(self):
        
        #初始化状态向量
        self.state = np.zeros(stateSize, np.float32)#[x,y,v_x,v_y,w,h],簇心位置，速度，高宽
        self.kf.statePost = self.state
        #初始化观测向量
        self.meas = np.zeros(measSize, np.float32)#[z_x,z_y,z_w,z_h]
        #初始化噪声向量
        self.procNoise = np.zeros(stateSize, np.float32)

        #状态转移矩阵初始化
        self.kf.transitionMatrix = np.zeros((stateSize, stateSize),np.float32)
        cv2.setIdentity(self.kf.transitionMatrix)#生成单位矩阵
        #不知道这个矩阵是干啥用的
        self.kf.errorCovPre = np.zeros((stateSize,stateSize),np.float32)
        cv2.setIdentity(self.kf.errorCovPre)
        #测量矩阵初始化
        self.kf.measurementMatrix = np.zeros((measSize,stateSize),np.float32)
        self.kf.measurementMatrix[0,0]=1.0
        self.kf.measurementMatrix[1,1]=1.0
        self.kf.measurementMatrix[2,4]=1.0
        self.kf.measurementMatrix[3,5]=1.0
        #噪声矩阵初始化
        self.kf.processNoiseCov =  np.zeros((stateSize, stateSize), dtype=np.float32)
        cv2.setIdentity(self.kf.processNoiseCov)
        self.kf.processNoiseCov[0,0] = 1e-2
        self.kf.processNoiseCov[1,1] = 1e-2
        self.kf.processNoiseCov[2,2] = 5
        self.kf.processNoiseCov[3,3] = 5
        self.kf.processNoiseCov[4,4] = 1e-2
        self.kf.processNoiseCov[5,5] = 1e-2
        cv2.setIdentity(self.kf.measurementNoiseCov)

    def Measure(self,aim):
        self.meas[0] = aim[0][0] + aim[0][2] / 2
        self.meas[1] = aim[0][1] + aim[0][3] / 2
        self.meas[2] = float(aim[0][2])
        self.meas[3] = float(aim[0][3])

    def _adapt_noise(self):
        # 遮挡时间越长，过程噪声越大
        noise_scale = 1.0 + self.notFoundCount * 0.05
        
        self.kf.processNoiseCov = self.kf.processNoiseCov * noise_scale
    def KalmenCalculate(self,aim,dT):
        """
        传入参数 ：aim 预测列表
        dt时间过程量
        """
        if(len(aim) == 0):#一个也没有
            self.notFoundCount += 1
            if self.notFoundCount >= 50:
                self.reinitialize_kf()

        else:
            
            #测量得到的物体位置
            self.notFoundCount = 0
            #更新测量
            self.Measure(aim)
            #print(self.found)
            #第一次检测
            if not self.found:
                #第一次分配测量量直接到状态变量中
                self.state[0] = self.meas[0]
                self.state[1] = self.meas[1]
                self.state[2] = 0
                self.state[3] = 0
                self.state[4] = self.meas[2]
                self.state[5] = self.meas[3]
                self.kf.statePost = self.state
                self.found = True
            else:
                self.kf.correct(self.meas) #Kalman修正
                #print("修正成功")
        
        self.kf.transitionMatrix[0,2] = dT
        self.kf.transitionMatrix[1,3] = dT
        self.state = self.kf.predict()
        self._adapt_noise()

        #第五第六个变量
        width = self.state[4]
        height = self.state[5]
        x_left = self.state[0] - width/2 #左上角横坐标
        y_left = self.state[1] - height/2  #左上角纵坐标
        x_right = self.state[0] + width/2 #右下角横坐标
        y_right = self.state[1] + height/2#右下角纵坐标
        center_x = self.state[0]
        center_y = self.state[1]
        return [center_x[0],center_y[0],x_left[0],y_left[0],x_right[0],y_right[0],width[0],height[0]]
    

    def reinitialize_kf(self):
        self.Matrix_define()
        self.notFoundCount = 0
        self.found =False
        print("卡尔曼滤波器重置")


class KalmanFilter_circle:
    """
    功能：一维卡尔曼滤波器，适用于圆形检测，反馈坐标为 (x, y, r)
    输入：当前测量数据 (x, y, r)
    返回值：滤波后的数据 [x, y, vx, vy]
    """
    def __init__(self):
        self.state_size = 4  # 状态向量维度: [x, y, vx, vy,]
        self.meas_size = 2   # 测量向量维度: [x, y]
        self.control_size = 0
        
        # 初始化卡尔曼滤波器
        self.kf = cv2.KalmanFilter(self.state_size, self.meas_size, self.control_size)
        self._initialize_matrices()
        
        self.found = False
        self.not_found_count = 0

    def _initialize_matrices(self):
        # 状态向量 (列向量)
        self.state = np.zeros((self.state_size, 1), dtype=np.float32)
        
        # 测量向量 (列向量)
        self.meas = np.zeros((self.meas_size, 1), dtype=np.float32)
        
        # 状态转移矩阵 (设置为单位矩阵并添加速度项)
        self.kf.transitionMatrix = np.eye(self.state_size, dtype=np.float32)
        
        # 测量矩阵 (映射状态到测量)
        self.kf.measurementMatrix = np.zeros((self.meas_size, self.state_size), dtype=np.float32)
        self.kf.measurementMatrix[0, 0] = 1  # x 观测
        self.kf.measurementMatrix[1, 1] = 1  # y 观测
        
        
        # 过程噪声协方差 (调整不同状态的噪声水平)
        self.kf.processNoiseCov = np.eye(self.state_size, dtype=np.float32)
        self.kf.processNoiseCov[0, 0] = 0.5  # x 噪声
        self.kf.processNoiseCov[1, 1] = 0.5  # y 噪声
        self.kf.processNoiseCov[2, 2] = 5   # vx 噪声
        self.kf.processNoiseCov[3, 3] = 5   # vy 噪声
 
        
        # 测量噪声协方差
        self.kf.measurementNoiseCov = np.eye(self.meas_size, dtype=np.float32) * 1
        
        # 初始误差协方差
        self.kf.errorCovPre = np.eye(self.state_size, dtype=np.float32)

    def KalmenCalculate(self, measurement, dt):
        """
        输入: 当前测量值 [x, y] 和时间间隔 dt
        返回: 滤波后的状态 [x, y,vx, vy]
        """
        if measurement is None:
            self.not_found_count += 1
            if self.not_found_count >= 50:
                self.reinitialize_kf()
            else:
                # 目标丢失时增加速度噪声，减少漂移
                self.kf.processNoiseCov[2, 2] *= 1.1
                self.kf.processNoiseCov[3, 3] *= 1.1
                # 限制最大噪声值
                self.kf.processNoiseCov[2, 2] = min(self.kf.processNoiseCov[2, 2], 100)
                self.kf.processNoiseCov[3, 3] = min(self.kf.processNoiseCov[3, 3], 100)
            
        else:
            self.not_found_count = 0
            # 更新测量值
            self.meas[0, 0] = measurement[0]
            self.meas[1, 0] = measurement[1]
            
            # 初始化跟踪器
            if not self.found:
                self._initialize_tracker()
                self.found = True
            else:
                # 修正步骤
                self.kf.correct(self.meas)
        
        # 预测步骤
        self._update_transition_matrix(dt)
        self.state = self.kf.predict()
        
        # 提取状态
        x = self.state[0, 0]
        y = self.state[1, 0]
        vx = self.state[2, 0]
        vy = self.state[3, 0]
   
        return [x, y]

    def _initialize_tracker(self):
        """首次检测到目标时初始化状态"""
        self.kf.statePost = np.zeros((self.state_size, 1), dtype=np.float32)
        self.kf.statePost[0, 0] = self.meas[0, 0]  # x
        self.kf.statePost[1, 0] = self.meas[1, 0]  # y
        # 速度初始化为0，误差协方差重置
        self.kf.errorCovPre = np.eye(self.state_size, dtype=np.float32)

    def _update_transition_matrix(self, dt):
        """更新状态转移矩阵中的时间项"""
        self.kf.transitionMatrix[0, 2] = dt  # x += vx*dt
        self.kf.transitionMatrix[1, 3] = dt  # y += vy*dt

    def reinitialize_kf(self):
        """重置滤波器"""
        self._initialize_matrices()
        self.found = False
        self.not_found_count = 0
        print("卡尔曼滤波器圆形已重置")

class ab_filter():
    """
    功能：一阶多项式滤波
    输入：当前数据
    返回值：滤波后的数据
    """
    def __init__(self,a=0.4):
        self.a = a
        self.last = 0


    def ab_filte(self,now):
        self.last = self.a * now + (1 - self.a) * self.last
        return self.last
