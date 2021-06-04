#!/usr/bin/env python
# coding: utf-8

# In[12]:


import matplotlib.pyplot as plt
import sys
import time
import scipy.io as io
import os
from os import listdir
import glob
import time

import pandas 

import cv2
import numpy as np
from opt_flow import draw_flow

import gc 

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

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.') and not f.startswith('contractivity'):
            yield f



rootDir = r'Z:\pangj05\TROPONIN2021\20210527DataSetAnalysis\Plate5'

subFolders = sorted(list(listdir_nohidden(rootDir)))
   

# Set output the csv file 
csvOutputName =rootDir+"\\"+"contractivity_relaxation_additionalInfo_"+"_0527_plate5.csv"
#print(csvOutputName)

fout = open(csvOutputName, 'w', newline='')

writer = csv.writer(fout)

writer.writerow( ('subFolder','cellTag', 'total_frame_num') )
    
fout.flush() 

for subFolder in subFolders[0:len(subFolders):1]:
    outputFolder = rootDir+"\\"+subFolder
   
    print(outputFolder)
    
    processFolder = outputFolder
    outputFolder = processFolder
    
    videoName0 =  glob.glob(outputFolder+"\\*_ds.avi")
    ## Specify input video
    videoName = videoName0[0]
    
    ## load the box information
    boxInfoName0 = glob.glob(outputFolder+"\\*boxes.csv")
    boxInfoName = boxInfoName0[0]
    ###print(boxInfoName)

    # reading the CSV file 
    boxCSVSize = os.path.getsize(boxInfoName)
    if boxCSVSize<1:
        continue
        
    # reading the CSV file 
    boxInfo = pandas.read_csv(boxInfoName,header = None) 

    # displaying the contents of the CSV file 
    cellNum = boxInfo.shape[0]
    
    imgFolder = outputFolder+"\\tiff\\*.tif"
    print("folder: ")
    print(imgFolder)
    imgNames = sorted(glob.glob(imgFolder))
    

    for tag in range(cellNum):
    ###for tag in range(1):
       
        writer.writerow((subFolder, tag, len(imgNames)))
            
        fout.flush()
fout.close()




