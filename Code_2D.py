import cv2
import  numpy as np
from utils_usb import *
K_2dcode = 0.5
#二维码与任务领取函数#####################################################################
def Code2D(frame,barcodes,dT,kalmen):

    aim = []
    text = []
    
    for barcode in barcodes:
        # 提取条形码的边界框的位置
        (x, y, w, h) = barcode.rect
        aim.append([x,y,w,h])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)#在图中画出来
        # 条形码数据为字节对象，所以如果我们想在输出图像上画出来，就需要先将它转换成字符串
        barcodeData = barcode.data.decode("utf-8")
        barcodeType = barcode.type
        # 绘出图像上条形码的数据和条形码类型
        text.append(str(barcodeData))#将二维码中的信息解码
    
    k_results = kalmen.KalmenCalculate(aim,dT)
    #print(k_results)
    if k_results[6]>1e-2 and len(k_results)>0:
        
        e_x = CaculateFeedback_X(K_2dcode,k_results[0])
        if(len(text)>0):
            results=[text[0],e_x]
        else:
            results=[None,e_x]
        cv2.putText(frame, str(e_x), (5,40), cv2.FONT_HERSHEY_SIMPLEX,.9, (0, 0, 255), 2)
        cv2.rectangle(frame, (int(k_results[2]), int(k_results[3])), (int(k_results[4]),int(k_results[5])), (255, 0, 0), 2)#在图中画出来
        
        
        
        return results
    else:return []

def StringtoTask(text):#将文本信息转化为任务信息
    Tasks = []
    Num_to_Task={1:"red",2:"blue",3:"green"}
    task_list = list(map(int, text.split()))
    #print(task_list)
    for i in range(len(task_list)):
        Tasks.append(Num_to_Task[task_list[i]])
    return Tasks