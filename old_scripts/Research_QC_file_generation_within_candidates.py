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

def generate_QC_images(inputSubfolder,rootOutputDir1,rootOutputDir2,cm_valid_list):
    print("src folder: " + inputSubfolder)
   
    (inputDirName,inputFolderName) = os.path.split(inputSubfolder)
    ###(outputDirName,outputFolderName) = os.path.split(outputSubfolder)

    cellFolderNames = glob.glob(inputSubfolder+"/*_registered/")
    ###print(cellFolderNames)
    ###print(len(cellFolderNames))
    print('inputFolderName: ' + inputFolderName )
    
    tmp = inputFolderName.split("_")
    tmp1=tmp[3].split('s')
    

    for ii in range(len(cellFolderNames)):
        segImgName=cellFolderNames[ii]+"\\frame_0000.jpg"
        traceFigName0 = cellFolderNames[ii].split("_registered")[0]
        cellTagName = traceFigName0.split("_cell_")[1]
        traceFigName = traceFigName0+'_quantification_register_part5_th_70.jpg'
        outputTraceFigName1 = rootOutputDir1 +"\\" + inputFolderName + "_cell_" +cellTagName+"_QC.jpg"
        outputTraceFigName2 = rootOutputDir2 +"\\" + inputFolderName + "_cell_" +cellTagName+"_QC.jpg"
        cm_name = tmp[2]+"_s"+tmp1[1].lstrip('0')+"_cell"+cellTagName+tmp[0]

        if cm_name in cm_valid_list:
            if os.path.exists(traceFigName):
                print("Good!")
                segImg = cv2.imread(segImgName)
                traceFig = cv2.imread(traceFigName)
                outputFig = np.concatenate((segImg,traceFig), axis=0)
                cv2.imwrite(outputTraceFigName1, outputFig)
        else:
            if os.path.exists(traceFigName):
                print("Bad!")
                segImg = cv2.imread(segImgName)
                traceFig = cv2.imread(traceFigName)
                outputFig = np.concatenate((segImg,traceFig), axis=0)
                cv2.imwrite(outputTraceFigName2, outputFig)

        ###
        ###if os.path.exists(traceFigName):
        ###    segImg = cv2.imread(segImgName)
        ###    traceFig = cv2.imread(traceFigName)
          
        ###    outputFig = np.concatenate((segImg,traceFig), axis=0)
        ###    cv2.imwrite(outputTraceFigName, outputFig)
         

 
if __name__ == "__main__":

    rootInputDir = r'Z:\pangj05\TROPONIN2021\20210616DataSetAnalysis\Pairwise\T20_AfterDosing_Mavacamten'
    rootOutputDir1 = r'Z:\pangj05\TROPONIN2021\20210616DataSetAnalysis\Pairwise\QC\valid_CMs\T20_AfterDosing_Mavacamten'
    rootOutputDir2 = r'Z:\pangj05\TROPONIN2021\20210616DataSetAnalysis\Pairwise\QC\non_valid_CMs\T20_AfterDosing_Mavacamten'

    csvCandidatesName = r'Z:\pangj05\TROPONIN2021\20210616DataSetAnalysis\Pairwise\QC\20210616_mava_pairwise_study_candidates_list_100_200Hz.csv'

    CM_names= pandas.read_csv(csvCandidatesName) 
    CM_names['cell_label_time'] = CM_names['cell_label'] + CM_names['Time']
    candidateList=CM_names['cell_label_time'].tolist()
    ###print(CM_names['cell_label'])
    ###print(CM_names['Time'])
 
    print(candidateList)
    inputSubfolders = [os.path.join(rootInputDir, o) for o in os.listdir(rootInputDir) if os.path.isdir(os.path.join(rootInputDir,o))]

    inputSubfolders = sorted(inputSubfolders)
  
    if not os.path.isdir(rootOutputDir1):
        print('The SUBFOLDER directory is not present. Creating a new one..')
        os.mkdir(rootOutputDir1)

    if not os.path.isdir(rootOutputDir2):
        print('The SUBFOLDER directory is not present. Creating a new one..')
        os.mkdir(rootOutputDir2)
    
    tic = time.time()
    for ii in range(len(inputSubfolders)):
    ###for ii in range(2):
       generate_QC_images(inputSubfolders[ii],rootOutputDir1,rootOutputDir2,candidateList)
    toc = time.time()
    print('total time is: ' + str(toc-tic))
