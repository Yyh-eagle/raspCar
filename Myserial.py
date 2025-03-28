import numpy as np 
import serial
import serial.tools.list_ports
import time
import struct

"""
创建一个订阅者节点
"""
    
    
#串口通信类 
class SerialPort():
    def __init__(self):

        self.serial_port = serial.Serial(
            port='/dev/ttyUSB1',#串口号#bug 固定串口
            baudrate=460800,#波特率
            bytesize=serial.EIGHTBITS,#八位字节
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
        )
        self.receive_data = None

        self.Camera_X = None
        self.Camera_Y =None
        self.Camera_Yaw =None
        self.Task_num = None
        self.Ifseize =0.0
        self.D_yaw1 = None#右侧舵机的数据
        self.D_yaw2 = None#左侧舵机的数据
        self.IfArrive = 0.0
        self.Task_data4 = None
        self.Task_data5 = None
        self.data = []
        self.receive()

    def receive(self):
        #print("in")
        data = []
        response =None
        size = self.serial_port.inWaiting()
        
        if  size>0:
            response = self.serial_port.read(44)
            
        if response is not None:
            if response[0]==179 and  response[1]==179 :
                
                data=response
                
                self.parse_packet(data)
                #print("serial receive")
        
        self.serial_port.flushInput()
    
    def Send_message(self,data_num):
        """
        flag ==0:看到目标，不抓取
        flag ==1:目标稳定，抓取
        flag ==2:T265的数据

        """

        transdata = [0xb3,0xb3,data_num[0],data_num[1],data_num[2],data_num[3]]#待发送的数据
        
        for data in data_num[4:]:
            
            data_bytes = struct.pack('f', data)
            transdata.extend(data_bytes)
        
        # 追加结束标志
        transdata.extend([0x5b, 0x5b])
        self.data = data_num
        #print(f"{transdata=}")#打印输出的数据
        byte_data = bytearray(transdata)
        
        self.serial_port.write(byte_data)
    def parse_packet(self,packet):
        # 提取数据部分（去除帧头和帧尾）0
        data = packet[2:-2]
        
        # 解析各浮点数字段（小端模式）
        self.Camera_X = struct.unpack('<f', data[0:4])[0]
        self.Camera_Y = struct.unpack('<f', data[4:8])[0]
        self.Camera_Yaw = struct.unpack('<f', data[8:12])[0]
        self.Task_num = struct.unpack('<f', data[12:16])[0]
        self.Ifseize = struct.unpack('<f', data[16:20])[0]
        self.D_yaw1 = struct.unpack('<f', data[20:24])[0]/57.3
        self.D_yaw2 = struct.unpack('<f', data[24:28])[0]/57.3
        self.IfArrive = struct.unpack('<f', data[28:32])[0]
        self.Task_data4 = struct.unpack('<f', data[32:36])[0]
        self.Task_data5 = struct.unpack('<f', data[36:40])[0]
        self.receive_data = [self.Camera_X,self.Camera_Y,self.Camera_Yaw,self.Task_num,self.Ifseize,self.D_yaw1,self.D_yaw2,self.IfArrive,self.Task_data4,self.Task_data5]
        

    

