#!/usr/bin/env python
# coding: utf-8

# In[4]:


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
import multiprocessing
from joblib import Parallel, delayed
import shutil

def data_configuration(subfolder,outputRootDir,ds_time=1):
    print(subfolder)    
    # Downsampling in resolution
    ds_res = 1
    # Downsampling in time

    imageNameRoot =  subfolder  + "\\tiff\\*.tif"
    #B9, C2,C4
    (dirName,folderName) = os.path.split(subfolder)

        
    subfolderOutput = outputRootDir+'\\' + folderName 
    tiffFolderOutput = subfolderOutput+"\\tiff"

    if not os.path.isdir(subfolderOutput):
        print('The OUTPUT directory is not present. Creating a new one..')
        os.mkdir(subfolderOutput)
        os.mkdir(tiffFolderOutput)


    imageNames = sorted(glob.glob(imageNameRoot))
    imageNum = len(imageNames)

    src0 = imageNames[0]

    (junk,outputImageName) = os.path.split(imageNames[0])
    dst0 = subfolderOutput+"\\"+outputImageName

    shutil.copy(src0, dst0)

    mm = 0
    tic = time.time()

    for jj in range(0,imageNum-1,ds_time):
        src = imageNames[jj]

        (junk,outputImageName) = os.path.split(imageNames[jj])
        dst = tiffFolderOutput+"\\"+outputImageName

        shutil.copy(src, dst)
        mm+=1
        if mm%100==0:
            print(mm)
    toc = time.time()
    print(toc-tic)     


if __name__ == "__main__":

    rootDir = r'Z:\pangj05\TROPONIN2021\20210527MavaSubDataSetAutomation\Plate4\Plate4'

    outputRootDir = r'Z:\pangj05\TROPONIN2021\20210527MavaSubDataSetAutomation\Plate4_ds'

    if not os.path.isdir(outputRootDir):
        print('The OUTPUT directory is not present. Creating a new one..')
        os.mkdir(outputRootDir)
        
    subfolders = [os.path.join(rootDir, o) for o in os.listdir(rootDir) 
                        if os.path.isdir(os.path.join(rootDir,o))]
    subfolders = sorted(subfolders)
    ds_time = 2
    tic = time.time()
    Parallel(n_jobs=6,prefer='threads')(delayed(data_configuration)(subfolder,outputRootDir,ds_time=ds_time) for subfolder in subfolders)  
    toc = time.time()
    print('total time is: ' + str(toc-tic))
