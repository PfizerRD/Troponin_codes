#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib
matplotlib.use("Agg")
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

from memory_profiler import profile
import multiprocessing
from joblib import Parallel, delayed
###from tqdm import tqdm

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.') and not f.startswith('contractivity'):
            yield f
        
def select_manual_picked(rootDir,outputDir,subFolder,CM_info):
    ###subFolders = sorted(list(listdir_nohidden(rootDir)))
    # Set output the csv file 
    print(subFolder)
    view_label_processed = subFolder.split('Dosing_')[1]
    print("view_label_processed: " + view_label_processed)
    inputFolder = rootDir+"\\"+subFolder  
    
    processFolder = inputFolder
    
    
    videoName0 =  glob.glob(inputFolder+"\\*_ds.avi")
    ## Specify input video
    videoName = videoName0[0]
    
    ## load the box information
    boxInfoName0 = glob.glob(inputFolder+"\\*boxes_automated.csv")
    boxInfoName = boxInfoName0[0]
    ###print(boxInfoName)

    # reading the CSV file 
    boxCSVSize = os.path.getsize(boxInfoName)
    if boxCSVSize<1:
        return 
    boxInfo = pandas.read_csv(boxInfoName,header = None) 

    # displaying the contents of the CSV file 
    cellNum = boxInfo.shape[0]
    
    imgFolder = inputFolder+"\\tiff\\*.tif"
 
    imgNames = sorted(glob.glob(imgFolder))
    frameNum = len(imgNames)

    CM_info_Single = CM_info.loc[CM_info['view_label']==view_label_processed,]

    tmp = view_label_processed.split('_')
    plate_name = tmp[0]
    tmp1 = tmp[1].split('s')
    
    view_name = tmp1[1].lstrip('0')

    fout = open(outputCSVname, 'a', newline='')
    writer = csv.writer(fout)
    fout.flush() 

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
        
        for kk, row in CM_info_Single.iterrows():
            if ind_x1<row['ind_x1'] and ind_x2>row['ind_x1'] and ind_y1<row['ind_y1'] and ind_y2>row['ind_y1']:
                print('view_label: '+ view_label_processed)
                cell_label = plate_name + "_s" + view_name + "_cell"+str(tag)
                print('cell label: ' + cell_label)
                
                writer.writerow([cell_label])
            
                fout.flush()
    fout.close()
 
    
    
if __name__ == "__main__":

        rootDir = r'Z:\pangj05\TROPONIN2021\20210616DataSetAnalysis\Pairwise\T0_BeforeDosing_Mavacamten'
       

        inputCSVname = r'Z:\pangj05\TROPONIN2021\20210616DataSetAnalysis\Pairwise\Shoh_ManSelect\MAVA_cardiomyocytes_boxes_selected.csv'
        outputCSVname = r'Z:\pangj05\TROPONIN2021\20210616DataSetAnalysis\Pairwise\Shoh_ManSelect\Mavacamten_cardiomyocytes_cell_label_selected.csv'

        CM_info= pandas.read_csv(inputCSVname,header=None)    
        CM_info.columns =['view_label', 'ind_x1', 'ind_y1', 'ind_x2','ind_y2']
    
        print(CM_info)

        subFolders = sorted(list(listdir_nohidden(rootDir)))
       
        ###CM_pipeline(rootDir,rotate,relax_th,subFolders)
        Parallel(n_jobs=1,prefer='threads')(delayed(select_manual_picked)(rootDir,outputCSVname,subFolder,CM_info) for subFolder in subFolders)    