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
from skimage import io

rootDir = r'Z:\pangj05\TROPONIN2021\20210616DataSetAnalysis\Pairwise\Shoh_ManSelect\Mavacamten'
plateName = 'Mavacamten'

outputFolder = r'Z:\pangj05\TROPONIN2021\20210616DataSetAnalysis\Pairwise\Shoh_ManSelect'


imageNameRoot =  rootDir  + "\\*_auto_annotated.tif"


imageNames = sorted(glob.glob(imageNameRoot))

# Set output the csv file 
csvOutputName = outputFolder+"\\"+ "cardiomyocytes_boxes_selected.csv"
print(csvOutputName)

for imageName in imageNames:
    print('imageName (FULL): ' + imageName)
    (dirName,fileName) = os.path.split(imageName)

    print('fileName: ' + fileName)
    view_label = plateName + "_" + fileName.split('_')[0]
    print('view_label: ' + view_label)
    ###img0 = cv2.imread(imageName)
    img1 = io.imread(imageName)
    ###print(img1)
    cellMask1 = (img1[:,:,0]>200)
    ###cellMask2 = ndi.binary_fill_holes(cellMask1)
    cellMask2 =  remove_small_objects(cellMask1,10)
    cellMask3 = binary_opening(cellMask2,disk(2))
    cellLabel = label(cellMask3)
    features = regionprops(cellLabel)
    
    
    ### Set output the csv file 
    ###csvOutputName = subfolder+"\\"+ "cardiomyocytes_boxes.csv"
    ###print(csvOutputName)

    fout = open(csvOutputName, 'a', newline='')

    writer = csv.writer(fout)
    
    fout.flush() 
        
    for ff in features:
        print(ff.bbox)
        writer.writerow((view_label,ff.bbox[0],ff.bbox[1],ff.bbox[2],ff.bbox[3]))
            
        fout.flush()
    fout.close()
    
    


