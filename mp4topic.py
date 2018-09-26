# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 10:00:39 2018

@author: sgs4176
"""


from time import gmtime, strftime
import cv2

 # 录像转换为图片
videoFile = 'E:\Video\input/abc.mp4'

cap = cv2.VideoCapture(videoFile)
cap.set(3,640)
cap.set(4,480)

while(cap.isOpened()):    
    ret, frame = cap.read()
    if ret:                       
        cv2.imshow('test', frame)    
        f = strftime("%Y%m%d%H%M%S.jpg", gmtime())    
        cv2.imwrite('E:\Video\output/'+ f, frame)
    else:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):        
        break       
cap.release
cv2.destroyAllWindows()