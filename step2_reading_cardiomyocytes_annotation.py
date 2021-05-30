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
from scipy import ndimage as ndi
from skimage.morphology import binary_closing, binary_opening, binary_dilation, disk,remove_small_objects,label
from skimage.morphology import remove_small_holes, binary_erosion
from skimage.measure import label, regionprops
import csv

rootDir = r'Z:\pangj05\TROPONIN2021\20210527DataSetAnalysis\Plate1'
###rootDir ='Z:\\TROPONIN2021\\Data20213024_revisted'

outputFolder = rootDir

if not os.path.isdir(outputFolder):
    print('The OUTPUT directory is not present. Creating a new one..')
    os.mkdir(outputFolder)
    
subfolders = [os.path.join(rootDir, o) for o in os.listdir(rootDir) 
                    if os.path.isdir(os.path.join(rootDir,o))]
  
subfolders=sorted(subfolders)

for subfolder in subfolders:
    print(subfolder)
    imageNameRoot =  subfolder  + "\\*.tif"
    #B9, C2, C4
    (dirName,videoFileName) = os.path.split(imageNameRoot)

    imageNameRoot0 = dirName
    videoName = dirName.split('\\')[-1]
    # Output video frame rate (needs to check the true image acqusition frequency and the ds_time defined below)

    imageNames = sorted(glob.glob(imageNameRoot))

    print(imageNames[0])
    img0 = cv2.imread(imageNames[0])
    cellMask1 = (img0[:,:,2]>np.max(img0[:,:,2])-20)
    cellMask2 = ndi.binary_fill_holes(cellMask1)
    cellMask3 =  remove_small_objects(cellMask2,500)
    cellMask3 = binary_opening(cellMask3,disk(3))
    cellLabel = label(cellMask3)
    features = regionprops(cellLabel)
    
    
    # Set output the csv file 
    csvOutputName = subfolder+"\\"+ "cardiomyocytes_boxes.csv"
    #print(csvOutputName)

    fout = open(csvOutputName, 'w', newline='')

    writer = csv.writer(fout)
    
    fout.flush() 
        
    for ff in features:
        print(ff.bbox)
        writer.writerow((ff.bbox[0],ff.bbox[1],ff.bbox[2],ff.bbox[3]))
            
        fout.flush()
    fout.close()
    
    


