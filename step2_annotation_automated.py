#!/usr/bin/env python
# coding: utf-8

# In[309]:


import matplotlib.pyplot as plt
import sys
import time
import scipy.io as io
import os
from os import listdir
import glob
import time

import cv2
import numpy as np
from opt_flow import draw_flow



import math
from skimage import data
from skimage import filters
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, binary_opening, binary_dilation, disk,remove_small_objects,label
from skimage.color import label2rgb
from skimage.morphology import remove_small_holes, binary_erosion
from skimage.segmentation.boundaries import find_boundaries
from skimage.segmentation import clear_border
from skimage.measure import EllipseModel
from skimage.draw import ellipse
from skimage.filters import threshold_otsu, threshold_local
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage import transform as tf
from scipy.signal import find_peaks
from skimage import filters
from skimage.filters import gaussian
import csv

from skimage import exposure
from skimage.morphology import reconstruction

from skimage.feature import peak_local_max
from scipy import ndimage as ndi

from skimage.segmentation import watershed


        
def MotionVectorCalculation(videoName, outputFolder, fps):
    cap = cv2.VideoCapture(videoName)
    (dirName,videoFileName) = os.path.split(videoName)

    outputPath = outputFolder+'\\' + videoFileName[:-4]
    if not os.path.isdir(outputPath):
        print('The directory is not present. Creating a new one..')
        os.mkdir(outputPath)
    else:
        print('The directory is present.')
    
    videoOut = outputPath+'\\' + videoFileName[:-4] + '_opticalFlow.avi'

    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    videoLen = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    print(videoOut)

    hei, wid = prvs.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    VideoOutput = cv2.VideoWriter(videoOut, fourcc, fps, (2*wid,hei))

    magStack = np.zeros([hei, wid, int(videoLen)-1],dtype =  np.float64)
    angStack = np.zeros([hei, wid, int(videoLen)-1],dtype =  np.float64)

    for ii in range(int(videoLen)-1):
        ret, frame2 = cap.read()
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, .5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        #hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        hsv[...,2] = mag*50
        #print(np.max(mag))
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        vis = draw_flow(next, flow*16)

        magStack[:,:,ii] = mag
        angStack[:,:,ii] = ang
        #bgr = np.concatenate((frame2,bgr),axis=1)
        #bgr = np.concatenate((bgr,vis),axis=1)

        #cv2.imshow('frame2',bgr)
        #k = cv2.waitKey(30) & 0xff
        #if k == 27:
        #    break
        #elif k == ord('s'):
        #    cv2.imwrite('opticalfb.png',frame2)
        #    cv2.imwrite('opticalhsv.png',bgr)

        if ii%50==0:
            print(ii)

        prvs = next

        frame_final = np.concatenate((bgr,vis),axis=1)
        VideoOutput.write(frame_final)

    VideoOutput.release()    
    cap.release()
    cv2.destroyAllWindows()
    
    return magStack, angStack

def CellSegmentation(img_enhance,block_size=25,offset=0.02):
    img_gray=img_enhance[:,:,0]
    
    local_thresh = threshold_local(img_gray, block_size, offset=0.02)
    binary_local = img_gray < local_thresh
  
    cellMask1 = binary_closing(binary_local,disk(1))
    cellMask2 = ndi.binary_fill_holes(cellMask1)
    
    cellMask3 =  remove_small_objects(cellMask2,50)

    cellMask4 = binary_closing(cellMask3,disk(2))
    cellMask5 = ndi.binary_fill_holes(cellMask4)
    cellMask6 = binary_erosion(cellMask5,disk(1))
    cellMask7 = binary_opening(cellMask6,disk(3))

    cellMask8 =  remove_small_objects(cellMask7,200)
    
    cellMask9 = binary_opening(cellMask8,disk(3))

    distance = ndi.distance_transform_edt(cellMask9)
    coords = peak_local_max(distance, min_distance=3,footprint=np.ones((3, 3)), labels=cellMask9)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=cellMask8)
    seg = labels>0
    seg = binary_opening(seg,disk(5))
    
    return seg


# In[319]:


rootDir = 'Z:\\TROPONIN2021\\20210511DataSetAnalysisDebug\\subDataSet\\Plate1\Plate1_Matrigel_s01'


inputVideoNames = glob.glob(rootDir+"\\*.avi")
## Specify input video
##inputVideoNames = inputVideoNames[0:2]
fps = 100
## Specify output
outputFolder = 'Z:\\TROPONIN2021\\20210511DataSetAnalysisDebug\\subDataSetOutput\\'

for videoName in inputVideoNames:
    print(videoName)
    magStack, angStack = MotionVectorCalculation(videoName, outputFolder, fps)
    magMax = np.max(magStack,2)*50
    magMean= np.mean(magStack,2)*50
    (dirName,videoFileName) = os.path.split(videoName)
    outputPath =  outputFolder+'\\' + videoFileName[:-4]
    magMaxImageName =  outputPath+"/"+videoFileName[:-4]+"_max_activity.tif" 
    magMeanImageName =  outputPath+"/"+videoFileName[:-4]+"_mean_activity.tif" 
    cv2.imwrite(magMeanImageName,magMean)
    cv2.imwrite(magMaxImageName,magMax)
               


# In[321]:


img_annotated = cv2.imread('.\\subDataSetOutput\\Plate1_Matrigel_s01_video_ds\\Plate1_Matrigel_s01_frame_0001_annotated.tif')
img_raw = cv2.imread('.\\subDataSetOutput\\Plate1_Matrigel_s01_video_ds\\Plate1_Matrigel_s01_frame_0001.tif')
mag_mean = cv2.imread('.\\subDataSetOutput\\Plate1_Matrigel_s01_video_ds\\Plate1_Matrigel_s01_video_ds_mean_activity.tif')
mag_max = cv2.imread('.\\subDataSetOutput\\Plate1_Matrigel_s01_video_ds\\Plate1_Matrigel_s01_video_ds_max_activity.tif')
img_enhance = exposure.equalize_adapthist(img_raw,kernel_size=25)
print(img_annotated.shape)
print(img_raw.shape)
print(mag_mean.shape)


# In[322]:


binary_local=CellSegmentation(img_enhance,block_size=19,offset=0.2)
plt.imshow(binary_local)


# In[323]:


fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(12, 12))
ax[0][0].imshow(img_raw)
ax[0][0].set_title('Image')
ax[0][1].imshow(img_annotated)
ax[0][1].set_title('Manual annotation')
ax[0][2].imshow(img_enhance)
ax[0][2].set_title('enhanced image')
ax[1][0].imshow(mag_mean*50)
ax[1][0].set_title('movement:mean')
ax[1][1].imshow(mag_max*10)
ax[1][1].set_title('movement:max')
ax[1][2].imshow(binary_local)
ax[1][2].set_title('cell segmentation')
fig.tight_layout()

plt.show()


# In[324]:


print(np.max(mag_mean[:,:,0]))
print(np.median(mag_mean[:,:,0]))
print(np.min(mag_mean[:,:,0]))
print(np.mean(mag_mean[:,:,0]))
print(np.std(mag_mean[:,:,0]))


# In[325]:


movement_local0=mag_mean[:,:,0]>np.mean(mag_mean[:,:,0])+np.std(mag_mean[:,:,0])*1.0
movement_local1 = binary_opening(movement_local0,disk(3))
movement_local2 = remove_small_objects(movement_local1,100)

move_cell = (movement_local2>0).astype(int)+(binary_local>0).astype(int)
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(9, 9))
ax[0][0].imshow(movement_local0)
ax[0][0].set_title('move segmentation 0')
ax[0][1].imshow(movement_local1)
ax[0][1].set_title('move segmentation 1')
ax[1][0].imshow(movement_local2)
ax[1][0].set_title('move segmentation 2')
ax[1][1].imshow(move_cell)
ax[1][1].set_title('move segmentation 2 plus cell segmentation')
fig.tight_layout()

plt.show()


# In[326]:


cells = reconstruction((movement_local2>0).astype(int),move_cell)
seg = np.logical_and(binary_local>0,cells>0)
plt.imshow(seg)


# In[327]:


seg_label=label(seg)
seg_box = seg.astype(int)
print(np.max(seg_box))
props = regionprops(seg_label)
for ii in range(len(props)):
###for ii in range(1):
    seg_box[props[ii].bbox[0]:props[ii].bbox[2],props[ii].bbox[1]:props[ii].bbox[3]] = 1


fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 12))
ax[0].imshow(img_annotated)
ax[0].set_title('Manual annotation')
ax[1].imshow(seg_box)
ax[1].set_title('Automated annotation')
fig.tight_layout()

plt.show()


# In[ ]:




