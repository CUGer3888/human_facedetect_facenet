#test
import cv2
import numpy as np

import os
import matplotlib.pyplot as plt
from PIL import Image

img_path = "face.png"
# 加载图像
def detect(img):
    img = cv2.imread(img_path)
    # 加载人脸检测器
    # detector = dlib.get_frontal_face_detector()

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imshow("gray", gray)
    cv2.waitKey(0)
    # # 2、训练一组人脸
    face_detector = cv2.CascadeClassifier(
        "C:\\Users\\CUGac\\PycharmProjects\\astar\\.venv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt.xml")

    # 3、检测人脸（用灰度图检测，返回人脸矩形坐标(4个角)）
    faces_rect = face_detector.detectMultiScale(gray, 1.05, 3)
    #                                          灰度图  图像尺寸缩小比例  至少检测次数（若为3，表示一个目标至少检测到3次才是真正目标）
    print("人脸矩形坐标faces_rect：", type(faces_rect),faces_rect)

    # 4、遍历每个人脸，画出矩形框
    dst = img.copy()
    for x, y, w, h in faces_rect:
        cv2.rectangle(dst, (x, y), (x + w, y + h), (0, 0, 255), 3)  # 画出矩形框
    #旋转图片

    # 显示
    cv2.imshow("dst", dst)
    cv2.waitKey(0)
detect(img_path)