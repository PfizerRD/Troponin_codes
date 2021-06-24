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

def generate_QC_images(inputSubfolder,rootOutputDir):
    print("src folder: " + inputSubfolder)
   
    (inputDirName,inputFolderName) = os.path.split(inputSubfolder)
    ###(outputDirName,outputFolderName) = os.path.split(outputSubfolder)

    cellFolderNames = glob.glob(inputSubfolder+"/*_registered/")
    ###print(cellFolderNames)
    ###print(len(cellFolderNames))
    for ii in range(len(cellFolderNames)):
        segImgName=cellFolderNames[ii]+"\\frame_0000.jpg"
        traceFigName0 = cellFolderNames[ii].split("_registered")[0]
        cellTagName = traceFigName0.split("_cell_")[1]
        traceFigName = traceFigName0+'_quantification_register_part5_th_70.jpg'
        outputTraceFigName = rootOutputDir +"\\" + inputFolderName + "_cell_" +cellTagName+"_QC.jpg"

        if os.path.exists(traceFigName):
            segImg = cv2.imread(segImgName)
            traceFig = cv2.imread(traceFigName)
          
            outputFig = np.concatenate((segImg,traceFig), axis=0)
            cv2.imwrite(outputTraceFigName, outputFig)
         

 
if __name__ == "__main__":

    rootInputDir = r'Z:\pangj05\TROPONIN2021\20210616DataSetAnalysis\Classic\Plate5'
    rootOutputDir = r'Z:\pangj05\TROPONIN2021\20210616DataSetAnalysis\Classic\QC\Plate5'
 
    inputSubfolders = [os.path.join(rootInputDir, o) for o in os.listdir(rootInputDir) if os.path.isdir(os.path.join(rootInputDir,o))]

    inputSubfolders = sorted(inputSubfolders)
    ###outputSubfolders = sorted(outputSubfolders)
    if not os.path.isdir(rootOutputDir):
        print('The SUBFOLDER directory is not present. Creating a new one..')
        os.mkdir(rootOutputDir)
    
    tic = time.time()
    for ii in range(len(inputSubfolders)):
    ###for ii in range(1):
        generate_QC_images(inputSubfolders[ii],rootOutputDir)
    toc = time.time()
    print('total time is: ' + str(toc-tic))
