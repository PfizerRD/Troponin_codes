#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Batch processing for muliple folders within a directory is added
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import time
import scipy.io as io
import os
from os import listdir
import glob
import time
import re
import shutil


rootDir = r'P:\techcenter-omtc\Projects\IMRU_Troponin\210513_Matrigel_Iso_DMSO\Plate1_Mat_Iso\*.tif'

outputFolder = r'Z:\pangj05\TROPONIN2021\20210513DataSetAnalysis'

interFrame = 1
    
if __name__ == "__main__":

    tic = time.time()
    if not os.path.isdir(outputFolder):
        print('The OUTPUT directory is not present. Creating a new one..')
        os.mkdir(outputFolder)
        
    imageNames = sorted(glob.glob(rootDir))
    imageNum = len(imageNames)
    print(imageNum)


    for jj in range(0,len(imageNames),interFrame):

        fileSize = os.path.getsize(imageNames[jj]) 
        ##print('Size of file is', fileSize, 'bytes')
        if fileSize<100000:
            continue

        (dirName,tiffFileName) = os.path.split(imageNames[jj])
        subfolderName = re.split('t\d\d\d\d',tiffFileName)[0]
        frameName0=re.split('_s\d\dt',tiffFileName)[1]
        frameName="frame_"+re.split('_ORG',frameName0)[0]
        
        if jj%200 ==0:
            print(imageNames[jj])
            print(tiffFileName)
            print(frameName0)
            print(frameName)
        
        outputSubFolder = outputFolder+'\\' + subfolderName
        
        if not os.path.isdir(outputSubFolder):
            print('The SUBFOLDER directory is not present. Creating a new one..')
            os.mkdir(outputSubFolder)
            
        videoOut = outputSubFolder +'\\' + subfolderName + '_video_ds.avi'

        outputTiffFolder = outputSubFolder + '\\tiff'
        
        src = imageNames[jj]
        dst = outputTiffFolder+'\\'+frameName+'.tif'
        dst0 = outputSubFolder+'\\'+frameName+'.tif'

        if not os.path.isdir(outputTiffFolder):
            print('The TIFF directory is not present. Creating a new one..')
            os.mkdir(outputTiffFolder)
            shutil.copyfile(src, dst)
            shutil.copyfile(src, dst0)
        else:
            ###print('The TIFF directory is present.')
            shutil.move(src, dst)
            


    toc = time.time()
    print('Total time is: ' + str(toc-tic))




