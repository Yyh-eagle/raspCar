import numpy as np 
import serial
import serial.tools.list_ports
import time

"""
创建一个订阅者节点
"""

#数据转换为高低两个字节
def data_transform(data):
    data = int(data *1000)
    print(data/10)
    data=int(data)
    if data <0:
        data=data+256*256
    data_high   =data//256
    data_low    =data%256
    return [data_high,data_low]
    
#串口通信类 
class SerialPort():
    def __init__(self):

        self.serial_port = serial.Serial(
            port='/dev/ttyUSB0',#串口号
            baudrate=115200,#波特率
            bytesize=serial.EIGHTBITS,#八位字节
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
        )
        self.data = None
        self.reiceive()
    
    def reiceive(self):
        data = []
        response =None
        size = self.serial_port.inWaiting()
        
        if  size>0:
            response = self.serial_port.read(8)
            #print(response)
        if response is not None:
            if response[0]==179 and  response[1]==179 :
                print("in")
                data=[response[2],response[3],response[4],response[5],response[6]]
                self.data = data
                
        self.serial_port.flushInput()
    
    def Send_message(self,data_num,flag):
        transdata = [0xb3,0xb3,flag]#待发送的数据
        for data in data_num:
            high,low=data_transform(data)
            transdata.append(high)
            transdata.append(low)
        
        transdata.append()
        transdata.append(0x5b)
        print(f"{transdata=}")#打印输出的数据
        self.serial_port.write(transdata)
        time.sleep(0.002)
    

