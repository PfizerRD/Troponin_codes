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

import multiprocessing
from joblib import Parallel, delayed

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.') and not f.startswith('contractivity'):
            yield f

def folder_creation(outputFolder,imageName):

    (dirName,tiffFileName) = os.path.split(imageName)
    subfolderName = re.split('t\d{3,4}',tiffFileName)[0]

    outputSubFolder = outputFolder+'\\' + subfolderName
    
    if not os.path.isdir(outputSubFolder):
        print('The SUBFOLDER directory is not present. Creating a new one..')
        os.mkdir(outputSubFolder)

    outputTiffFolder = outputSubFolder + '\\tiff'
  
    if not os.path.isdir(outputTiffFolder):
        print('The TIFF directory is not present. Creating a new one..')
        os.mkdir(outputTiffFolder)


def tiff_folder_generation(outputFolder,imageName):
    fileSize = os.path.getsize(imageName) 
    ##print('Size of file is', fileSize, 'bytes')
    if fileSize<100000:
        return

    (dirName,tiffFileName) = os.path.split(imageName)

    subfolderName = re.split('t\d{3,4}',tiffFileName)[0]
    frameName0=re.split('_s\d{2,3}t',tiffFileName)[1]
    frameName="frame_"+re.split('_ORG',frameName0)[0]

    outputSubFolder = outputFolder+'\\' + subfolderName
    
    if not os.path.isdir(outputSubFolder):
        print('The SUBFOLDER directory is not present. Creating a new one..')
        os.mkdir(outputSubFolder)

    outputTiffFolder = outputSubFolder + '\\tiff'

    src = imageName
    dst = outputTiffFolder+'\\'+subfolderName+'_'+frameName+'.tif'

    if not os.path.isdir(outputTiffFolder):
        print('The TIFF directory is not present. Creating a new one..')
        os.mkdir(outputTiffFolder)
        ###shutil.copy(src, dst)
        shutil.move(src,dst)
    else:
        ###print('The TIFF directory is present.')
        shutil.move(src, dst)
        ###shutil.copy(src, dst)

def copy_first_frame(outputFolder,subfolder):
    imageNameRoot =  outputFolder+"\\"+subfolder  + "\\tiff"+"\\*.tif"
    imageNames = sorted(glob.glob(imageNameRoot))
    
    (dirName,tiffFileName) = os.path.split(imageNames[0])
    frameNum=re.split('frame_',tiffFileName)[1]
    frameName="frame_"+frameNum

    src = imageNames[0]
    dst = outputFolder+"\\"+subfolder+"\\"+frameName
    shutil.copy(src, dst)


if __name__ == "__main__":

    tic = time.time()

    rootDir = r'Z:\techcenter-omtc\Projects\RDRU_Mybpc3\Data4PipelineTesting\Basal\*.tif'
    outputFolder = r'Z:\techcenter-omtc\Scratch\pangj05\RDRU_Troponin2021\Pipeline_testing\Basal'

    cpu_num = 2
    if not os.path.isdir(outputFolder):
        print('The OUTPUT directory is not present. Creating a new one..')
        os.mkdir(outputFolder)
        
    imageNames = sorted(glob.glob(rootDir))

    ## have to set to single core to run to avoid folder/directory overwriting conflicts
    Parallel(n_jobs=1,prefer='threads')(delayed(folder_creation)(outputFolder,imageName) for imageName in imageNames)   
    toc1 = time.time()
    print('folder creation time: ' + str(toc1-tic))

    Parallel(n_jobs=cpu_num,prefer='threads')(delayed(tiff_folder_generation)(outputFolder,imageName) for imageName in imageNames)  
    toc2 = time.time()
    print("copy/moving all frames time: " + str(toc2-toc1))
    
    subfolders = sorted(list(listdir_nohidden(outputFolder)))
    Parallel(n_jobs=cpu_num,prefer='threads')(delayed(copy_first_frame)(outputFolder,subfolder) for subfolder in subfolders)  
    toc = time.time()
    print("copy first frame time: " + str(toc-toc2))

    print('Total time is: ' + str(toc-tic))




