# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 11:38:33 2018

@author: sgs4176
"""

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from os import listdir
from SSD_MODEL.ssd import SSD300
from SSD_MODEL.ssd_utils import BBoxUtility
from scipy.misc import imread 
import numpy as np
import cv2

voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',              
 'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',              
 'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant',              
 'Sheep', 'Sofa', 'Train', 'Tvmonitor']

NUM_CLASSES = len(voc_classes) + 1
input_shape = (300,300,3)
model = SSD300(input_shape, num_classes = NUM_CLASSES)
model.load_weights('.\weights_SSD300.hdf5', by_name=True)
bbox_until = BBoxUtility(NUM_CLASSES)

def ssd(img_path):
    global oPath
    
    inputs = []
    images = []
    
    img0 = cv2.imread(img_path)
    img = image.load_img(img_path, target_size=(300, 300))
    img = image.img_to_array(img)
    images.append(imread(img_path))
    inputs.append(img.copy())
    inputs = preprocess_input(np.array(inputs))
    
    preds = model.predict(inputs, batch_size=1, verbose=1)
    results = bbox_until.detection_out(preds)
    
    for i,img in enumerate(images):
        det_lable = results[i][:, 0]
        det_conf = results[i][:, 1]
        det_xmin = results[i][:, 2]
        det_ymin = results[i][:, 3]
        det_xmax = results[i][:, 4]
        det_ymax = results[i][:, 5]
        print('i:{0}, det_lable:{1}, det_conf:{2}'.format(i, det_lable, det_conf))
        top_indices = [i for i , conf in enumerate(det_conf) if conf >= 0.5]
        
        top_conf = det_conf[top_indices]
        top_label_indices = det_lable[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]
        
        for i0 in range(top_conf.shape[0]):
            xmin = int(round(top_xmin[i0] * img.shape[1]))
            ymin = int(round(top_ymin[i0] * img.shape[0]))
            xmax = int(round(top_xmax[i0] * img.shape[1]))
            ymax = int(round(top_ymax[i0] * img.shape[0]))
            score = top_conf[i0]
            lable = int(top_label_indices[i0])
            lable_name = voc_classes[lable - 1]
            print(lable_name,score,xmin,ymin,xmax,ymax)
            fileStr0 = img_path.split('.')[-2]
            fileStr0 = fileStr0.split('/')[-1]
            
            if lable_name == 'Person':
                fileStr = '%s/Person/%s.%d.jpg' %(oPath,fileStr0,i0+1)
                im = img0[ymin:ymax, xmin:xmax]
                cv2.imwrite(fileStr,im)
                print('Person0',fileStr)
                
            if lable_name == 'Car1' or lable_name == 'MotorBike1':
                fileStr = '%s/Car/%s.%d.jpg' %(oPath,fileStr0,i0+1)
                im = img0[ymin:ymax, xmin:xmax]
                cv2.imwrite(fileStr,im)
                print('Car',fileStr)
    
if __name__ == '__main__':
    
    img_path = '20180821085030.jpg'
    
    mPath = 'E:\Video\output'
    oPath = 'E:\Video\detput'
    trainFileList = listdir(mPath)
    m = len(trainFileList)
    print('m:%s' %(m))
    
    for i in range(m):
        fileNameStr = trainFileList[i]
        fileStr = fileNameStr.split('.')[-2]
        print('i:{0},fileNameStr:{1},fileStr:{2}'.format(i,fileNameStr,fileStr))
        fileNameStr = '%s%s' %(mPath,fileNameStr)
        print('step:{0}/{1}'.format(i,m))
        ssd(fileNameStr)