import cv2
import numpy as np
# 初始化全局变量存储点击坐标
click_points = []


def mouse_callback(event, x, y, flags, param):
    global click_points, img

    # 左键点击事件
    if event == cv2.EVENT_LBUTTONDOWN:
        # 记录坐标
        click_points.append((x, y))
        print(f"点击坐标 (x, y): ({x}, {y})")

        # 在图像上绘制红点和坐标文本
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.putText(img, f'({x}, {y})', (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Image", img)


if __name__ == "__main__":
    # 读取图像（替换为你的图像路径）
    cap = cv2.VideoCapture(0)

    # 创建窗口并绑定鼠标回调函数
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouse_callback)



    while True:
        # 显示图像并等待操作\
        _,img = cap.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 高斯模糊（减少噪声）
        gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            gray_blur,
            cv2.HOUGH_GRADIENT,
            dp=1,  # 累加器分辨率（1=与输入图像相同）
            minDist=50,  # 圆之间的最小距离
            param1=100,  # Canny边缘检测的高阈值
            param2=30,  # 圆心检测阈值（越小检测到的圆越多，但可能包含误检）
            minRadius=10,  # 最小圆半径
            maxRadius=500  # 最大圆半径
        )
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                # 圆心坐标和半径
                x, y, r = circle[0], circle[1], circle[2]
                print(x,y)
                # 绘制圆心
                cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
                # 绘制圆轮廓
                cv2.circle(img, (x, y), r, (0, 0, 255), 2)
        cv2.imshow("Image", img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # 按 ESC 退出
            break

    cv2.destroyAllWindows()