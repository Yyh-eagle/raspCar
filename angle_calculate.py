import numpy as np
import math 
L1 = 14
L2 =15
Set  =42/57.3#镜头短边和右侧机械臂之间的角度
def caculate_angle(yaw1=40/57.3,yaw2=140/57.3):
    #print([yaw1*57.3,yaw2*57.3])
    if(yaw1>yaw2):
        return None
    
    theta = (np.pi/2)-0.5*(yaw2-yaw1)

    alpha = np.arccos(L1*np.cos(theta)/L2)
   
    return np.pi-alpha-theta+yaw1

#180-
def Caculate_yaw(aim,angle):
    
    return -aim+angle-Set+np.pi/2


