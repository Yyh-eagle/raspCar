import cv2
import  numpy as np
from utils_usb import *
K_2dcode = 0.5
#二维码与任务领取函数#####################################################################
def Code2D(frame,barcodes):


    text = []
    
    for barcode in barcodes:
        # 提取条形码的边界框的位置
        (x, y, w, h) = barcode.rect

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)#在图中画出来
        # 条形码数据为字节对象，所以如果我们想在输出图像上画出来，就需要先将它转换成字符串
        barcodeData = barcode.data.decode("utf-8")
        barcodeType = barcode.type
        # 绘出图像上条形码的数据和条形码类型
        text.append(str(barcodeData))#将二维码中的信息解码
    
    
    if(len(text)>0):
        results=[text[0]]
    else:
        results=[]
        
    return results
    

def text_to_array(text):
    # 移除字符串中的 '+' 符号
    cleaned_text = text.replace('+', '')
    
    # 将每个字符转换为整数并存储到数组中
    array = [int(char) for char in cleaned_text]
    
    return array


