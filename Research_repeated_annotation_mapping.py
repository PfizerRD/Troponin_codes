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
import pandas
from skimage.morphology import binary_closing, binary_opening, binary_dilation, disk,remove_small_objects,label

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

    rootInputDir = r'Z:\pangj05\TROPONIN2021\20210616DataSetAnalysis\Pairwise\T0_BeforeDosing_Mavacamten'
    rootOutputDir = r'Z:\pangj05\TROPONIN2021\20210616DataSetAnalysis\Pairwise\T20_AfterDosing_Mavacamten'
 
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
