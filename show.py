import cv2
import numpy as np

def display_task_window(text="NO task"):
    # 获取屏幕分辨率
    screen_width =1280  # 默认值，需根据实际屏幕修改
    screen_height = 800
    
    # 创建纯白图像
    img = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * 255
    
    # 设置字体参数
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 7
    thickness = 9
    color = (0, 0, 0)  # 黑色
    
    # 获取文本尺寸
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # 计算文本位置
    x = (screen_width - text_width) // 2
    y = (screen_height + text_height) // 2
    
    # 绘制文本
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness)
    
    # 创建无边框窗口
    cv2.namedWindow("Task Display", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Task Display", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    #cv2.setWindowProperty("Task Display", cv2.WND_PROP_TOOLBAR, cv2.WINDOW_PROP_AUTOSIZE)
    
    # 显示图像
    cv2.imshow("Task Display", img)
    cv2.waitKey(1)  # 保持窗口响应

# 使用示例
if __name__ == "__main__":
    # 初始显示
    display_task_window()
    
    # 更新显示内容
    display_task_window("Processing Task 1...")
    
    # 保持窗口显示
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()