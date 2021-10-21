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
from scipy.fftpack import fftn,fftfreq
from skimage.exposure import equalize_adapthist
from skimage.segmentation import clear_border

from skimage import feature

import multiprocessing
from joblib import Parallel, delayed

import pandas

DEFAULT_FRAME_NUM = 600

### STEP00 FUNCTION
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
    frameName0=re.split('_s\d\dt',tiffFileName)[1]
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


### STEP 01 FUNCTION
def video_generation(subfolder,fps=100.0,interFrame=1):
    print(subfolder)    
    # Downsampling in resolution
    ds_res = 1
    # Downsampling in time
    ds_time = interFrame 
##for subfolder in subfolders[0:len(subfolders):1]:
##for subfolder in subfolders[49:72]:
    imageNameRoot =  subfolder  + "\\tiff\\*.tif"
    #B9, C2,C4
    (dirName,videoFileName) = os.path.split(subfolder)

    imageNameRoot0 = dirName
    videoName = videoFileName
    # Output video frame rate (needs to check the true image acqusition frequency and the ds_time defined below)
        
    videoOut = subfolder+'\\' + videoName + '_video_ds.avi'

    imageNames = sorted(glob.glob(imageNameRoot))
    imageNum = len(imageNames)
    
    if imageNum>DEFAULT_FRAME_NUM:
        fps=200.0
    img0 = cv2.imread(imageNames[0])
    img1 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    img = img1[::ds_res,::ds_res]

    hei, wid = img.shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    VideoOutput = cv2.VideoWriter(videoOut, fourcc, fps, (wid,hei))


    mm = 0
    tic = time.time()

    for jj in range(0,imageNum-1,interFrame):
        img0 = cv2.imread(imageNames[jj])
        img1 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        img = img1[::ds_res,::ds_res]
         
        frame_final = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) 

        #frame_final=np.concatenate((frame_final,vis),axis=1) 

        VideoOutput.write(frame_final)
        mm+=1
        if mm%100==0:
            print(mm)
    toc = time.time()
    print(toc-tic)     

    VideoOutput.release()
    cv2.destroyAllWindows()
    toc = time.time()

### STEP 2 FUNCTION (mapping annotation T0 to T20)

def longitudinal_annotation_data_configuration(inputSubfolder,outputSubfolder):
    print("src folder: " + inputSubfolder)
    print("des folder: " + outputSubfolder)    
   
    (inputDirName,inputFolderName) = os.path.split(inputSubfolder)
    (outputDirName,outputFolderName) = os.path.split(outputSubfolder)

    srcCSVname = inputSubfolder+'\\cardiomyocytes_boxes_automated.csv'
    desCSVname = outputSubfolder+'\\cardiomyocytes_boxes_automated.csv'

    shutil.copy(srcCSVname, desCSVname)
   
    desRawImgName = outputSubfolder+'\\frame_001.tif'

    img0 = cv2.imread(desRawImgName)
    img1 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    hei,wid = img1.shape
    
    seg_box = np.zeros((hei, wid))

    boxInfo = pandas.read_csv(desCSVname,header = None) 

    # displaying the contents of the CSV file 
    cellNum = boxInfo.shape[0]

    for tag in range(cellNum):
        ind_x1 = int(boxInfo.iloc[tag][0])
        ###ind_x1 = int(boxInfo.iloc[tag][1])
        ###print(ind_x1)
        ind_x2 = int(boxInfo.iloc[tag][2])
        ###ind_x2 = int(boxInfo.iloc[tag][3])
        ###print(ind_x2)
        ind_y1 = int(boxInfo.iloc[tag][1])
        ###ind_y1 = int(boxInfo.iloc[tag][0])
        ###print(ind_y1)
        ind_y2 = int(boxInfo.iloc[tag][3])
        ###ind_y2 = int(boxInfo.iloc[tag][2])
        ###print(ind_y2)

        startRow = ind_x1
        endRow = ind_x2
        startCol = ind_y1
        endCol = ind_y2
        seg_box[startRow:endRow,startCol]=1
        seg_box[startRow:endRow,endCol]=1
        seg_box[startRow,startCol:endCol]=1
        seg_box[endRow,startCol:endCol]=1
      
    seg_box = seg_box>0  
    seg_box = binary_dilation(seg_box,disk(2))
    seg_box = seg_box.astype(int)*222

    img0[:,:,0] = seg_box
    outputImageAnnoName = outputSubfolder+"\\frame_001_auto_annotated.tif"
    cv2.imwrite(outputImageAnnoName, img0)




if __name__ == "__main__":

    ### step00
    tic = time.time()

    rootDir = r'P:\techcenter-omtc\Projects\IMRU_Troponin\211015_ZSF1_Imaging\Plate6_T20\*.tif'
    outputFolder = r'Z:\pangj05\TROPONIN2021\20211015DataSetAnalysis\Plate6_T20'

    ### MUST BE T0
    rootInputDir = r'Z:\pangj05\TROPONIN2021\20211015DataSetAnalysis\Plate6_T0' ### MUST BE T0
    rootOutputDir = outputFolder

    cpu_num_step00 = 2
    cpu_num_step01 = 4
    cpu_num_step2 = 4


    if not os.path.isdir(outputFolder):
        print('The OUTPUT directory is not present. Creating a new one..')
        os.mkdir(outputFolder)
        
    imageNames = sorted(glob.glob(rootDir))

    ## have to set to single core to run to avoid folder/directory overwriting conflicts
    Parallel(n_jobs=1,prefer='threads')(delayed(folder_creation)(outputFolder,imageName) for imageName in imageNames)   
    toc1 = time.time()
    print('folder creation time: ' + str(toc1-tic))

    Parallel(n_jobs=cpu_num_step00,prefer='threads')(delayed(tiff_folder_generation)(outputFolder,imageName) for imageName in imageNames)  
    toc2 = time.time()
    print("copy/moving all frames time: " + str(toc2-toc1))
    
    subfolders = sorted(list(listdir_nohidden(outputFolder)))
    Parallel(n_jobs=cpu_num_step00,prefer='threads')(delayed(copy_first_frame)(outputFolder,subfolder) for subfolder in subfolders)  
    toc = time.time()
    print("copy first frame time: " + str(toc-toc2))

    print('Total time for STEP00 is: ' + str(toc-tic))

    ### step01
    rootDir = outputFolder
    outputFolder = rootDir

    if not os.path.isdir(outputFolder):
        print('The OUTPUT directory is not present. Creating a new one..')
        os.mkdir(outputFolder)
        
    subfolders = [os.path.join(rootDir, o) for o in os.listdir(rootDir) 
                        if os.path.isdir(os.path.join(rootDir,o))]
    subfolders = sorted(subfolders)
    fps = 100.0;
    interFrame = 1
    tic = time.time()
    Parallel(n_jobs=cpu_num_step01,prefer='threads')(delayed(video_generation)(subfolder,fps,interFrame) for subfolder in subfolders)  
    toc = time.time()
    print('total time for STEP01 is: ' + str(toc-tic))

    ### step2 (mapping annotation from T0 to T20)
 
    inputSubfolders = [os.path.join(rootInputDir, o) for o in os.listdir(rootInputDir) if os.path.isdir(os.path.join(rootInputDir,o))]
    outputSubfolders = [os.path.join(rootOutputDir, o) for o in os.listdir(rootOutputDir) if os.path.isdir(os.path.join(rootOutputDir,o))]

    inputSubfolders = sorted(inputSubfolders)
    outputSubfolders = sorted(outputSubfolders)
    
    tic = time.time()
    for ii in range(len(inputSubfolders)):
    ###for ii in range(2):
        longitudinal_annotation_data_configuration(inputSubfolders[ii],outputSubfolders[ii])
    toc = time.time()
    print('total time is: ' + str(toc-tic))




