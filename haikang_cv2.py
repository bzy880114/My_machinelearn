# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 18:10:48 2018

@author: sgs4176
"""

import cv2

# 使用RTSP协议调用网络摄像头，下面是使用海康摄像头的一个例子
# LECHENG video
cap = cv2.VideoCapture('rtsp://admin:FAAB1BLC@10.0.1.223:554/cam/realmonitor?channel=1&subtype=1')
#YSY video
#cap = cv2.VideoCapture('rtsp://admin:QACXNT@10.0.1.110/h264/ch1/main/av_stream')

width = 640
height = 480
cap.set(3, width)
cap.set(4, height)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('./out.avi', fourcc, 10, (width, height))

# 下面注释的代码是调用本地摄像头例子
#cap = cv2.VideoCapture (0)
#print(cap.isOpened())0

while cap.isOpened():
    ret,frame = cap.read()
    frame = cv2.resize(frame, (width, height))
    out.write(frame)
    cv2.imshow("frame",frame)
    key = cv2.waitKey(1)
    if(key == ord('q') ):
        break
cap.release()
out.release()
cv2.destroyAllWindows()
