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
    frameName0=re.split('_s\d{1,3}t',tiffFileName)[1]
    ######frameName0=re.split('_s\dt',tiffFileName)[1]
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

    rootDirAll = [
    r'Q:\Projects\RDRU_Mybpc3\20220510_monolayer_iPSC_imaging_MYBPC3\Export\D12_plate_1\*.tif',
    r'Q:\Projects\RDRU_Mybpc3\20220510_monolayer_iPSC_imaging_MYBPC3\Export\F11_first_half_plate_3\*.tif',
    r'Q:\Projects\RDRU_Mybpc3\20220510_monolayer_iPSC_imaging_MYBPC3\Export\F11_plate_3_second_half\*.tif',
    r'Q:\Projects\RDRU_Mybpc3\20220510_monolayer_iPSC_imaging_MYBPC3\Export\H11_plate_2\*.tif',
    r'Q:\Projects\RDRU_Mybpc3\20220510_monolayer_iPSC_imaging_MYBPC3\Export\Parental_plate_4\*.tif'
    ]

    outPutFolderAll = [
    r'Z:\pangj05\RDRU_MYBPC3_2022\0510_iPSC_monolayer_DataSetAnalysis\D12_plate_1',
    r'Z:\pangj05\RDRU_MYBPC3_2022\0510_iPSC_monolayer_DataSetAnalysis\F11_plate_3_first_half',
    r'Z:\pangj05\RDRU_MYBPC3_2022\0510_iPSC_monolayer_DataSetAnalysis\F11_plate_3_second_half',
    r'Z:\pangj05\RDRU_MYBPC3_2022\0510_iPSC_monolayer_DataSetAnalysis\H11_plate_2',
    r'Z:\pangj05\RDRU_MYBPC3_2022\0510_iPSC_monolayer_DataSetAnalysis\Parental_plate_4'  
    ]

    assert len(rootDirAll) == len(outPutFolderAll)


    tic = time.time()

    for mm in range(len(rootDirAll)):

        tic = time.time()
        rootDir = rootDirAll[mm]
        outputFolder = outPutFolderAll[mm]
        print(rootDir)
        print(outputFolder)

        cpu_num = 6
        if not os.path.isdir(outputFolder):
            print('The OUTPUT directory is not present. Creating a new one..')
            os.mkdir(outputFolder)
            
        imageNames = sorted(glob.glob(rootDir))
        ###imageNames = imageNames0[::2]

        print("image number: " + str(len(imageNames)))
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




