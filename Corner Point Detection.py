# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 20:08:47 2023

@author: lilulu

角点检测代码，
cornerSubPix()对检测到的角点作进一步的优化计算，可使角点的精度达到亚像素级别。

功能形式

void cornerSubPix( InputArray image, InputOutputArray corners, Size
winSize, Size zeroZone, TermCriteria criteria )

参数说明

具体调用形式如下：
void cv::cornerSubPix(
cv::InputArray image, // 输入图像
cv::InputOutputArray corners, // 角点（既作为输入也作为输出）
cv::Size winSize, // 区域大小为 NXN; N=(winSize*2+1)
cv::Size zeroZone, // 类似于winSize，但是总具有较小的范围，Size(-1,-1)表示忽略
cv::TermCriteria criteria // 停止优化的标准
);

第一个参数是输入图像，和cv::goodFeaturesToTrack()中的输入图像是同一个图像。
第二个参数是检测到的角点，既是输入也是输出。

第三个参数是计算亚像素角点时考虑的区域的大小，大小为NN， N=(winSize2+1)。

第四个参数作用类似于winSize，但是总是具有较小的范围，通常忽略（即Size(-1, -1)）。

第五个参数用于表示计算亚像素时停止迭代的标准，可选的值有cv::TermCriteria::MAX_ITER
、cv::TermCriteria::EPS（可以是两者其一，或两者均选），前者表示迭代次数达到了最大次数时停止，
后者表示角点位置变化的最小值已经达到最小时停止迭代。二者均使用cv::TermCriteria()构造函数进行指定。
"""

import cv2
import numpy as np

#原2.bmp 11*8 w*h 31-39：21*14=294 40-48：22*14=308 49-58:16*8=128 59-68:14*8=112 89-98：17*9=153
# 2nd采集：22*14=308 good_picture=[31, 34, 37, 41, 49, 55, 60, 61]
# tool2 guyueju采集 11*8=88 左目 good_picture=[71, 73, 75, 77, 79, 81, 83, 85, 87,89]
w = 11
h = 8

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001) ###############################
# 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
objp = np.zeros((w * h, 3), np.float32)
objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
objpoints = []  # 在世界坐标系中的三维点
imgpoints = []  # 在图像平面的二维点
#img_path = "./save_cail_data/color_55.png"
img_path = "C:/work/datasets/20230923/test_grasp_unlabel/color/color_115.bmp"
# img_path = "C:/work/datasets/20231005_/carema_guyueju/12.png"

img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("cs",gray)
ret, corners = cv2.findChessboardCorners(gray, (w, h), None)

# change order of corners. to left and up point first 
corners_opposite_direction = np.empty_like(corners)

corners_opposite_direction = np.flip(corners, axis = 0)
corners_opposite_direction = corners_opposite_direction.astype(np.float32)
print(corners)
print(len(corners))
print(type(corners))
# print(corners_opposite_direction)
# print(len(corners_opposite_direction))
# print(type(corners_opposite_direction))
#cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)############################################
cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)###############################################

objpoints.append(objp)
imgpoints.append(corners)
# 将角点在图像上显示
cv2.drawChessboardCorners(gray, (w, h), corners, ret)
cv2.imshow("cs2",gray)
cv2.waitKey(0)


# cv2.cornerSubPix(gray, corners_opposite_direction, (5, 5), (-1, -1), criteria)###############################################

# objpoints.append(objp)
# imgpoints.append(corners_opposite_direction)
# # 将角点在图像上显示
# cv2.drawChessboardCorners(gray, (w, h), corners_opposite_direction, ret)
# cv2.imshow("cs2",gray)
# cv2.waitKey(0)


