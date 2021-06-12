import matplotlib.pyplot as plt
import sys
import time
import scipy.io as io
import os
from os import listdir
import glob
import time

import cv2
import numpy as np
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
import shutil

def cellSegmentation(subfolder,block_size=25,offset=0.02):

    imageNameRoot =  subfolder  + "\\tiff\\*.tif"
        #B9, C2,C4
    (dirName,videoFileName) = os.path.split(subfolder)

    imageNameRoot0 = dirName
    imageNames = sorted(glob.glob(imageNameRoot))

    img0 = cv2.imread(imageNames[0])
    
    img_enhance = equalize_adapthist(img0, kernel_size=block_size)
    img_gray=img_enhance[:,:,0]
    
    local_thresh = threshold_local(img_gray, block_size, offset=offset)
    binary_local = img_gray < local_thresh
  
    cellMask1 = binary_closing(binary_local,disk(1))
    cellMask2 = ndi.binary_fill_holes(cellMask1)
    
    cellMask3 =  remove_small_objects(cellMask2,50)

    cellMask4 = binary_closing(cellMask3,disk(2))
    cellMask5 = ndi.binary_fill_holes(cellMask4)
    cellMask6 = binary_erosion(cellMask5,disk(1))
    cellMask7 = binary_opening(cellMask6,disk(3))

    cellMask8 =  remove_small_objects(cellMask7,200)
   

    edges2 = feature.canny(img0[:,:,0], sigma=1.8)
    
    seg = np.logical_and(1-edges2,cellMask8)
    seg = binary_opening(seg,disk(3))
    seg =  remove_small_objects(seg,800)
    seg =  clear_border(seg)
    ##distance = ndi.distance_transform_edt(cellMask9)
    ###coords = peak_local_max(distance, min_distance=3,footprint=np.ones((3, 3)), labels=cellMask9)
    ###mask = np.zeros(distance.shape, dtype=bool)
    ###mask[tuple(coords.T)] = True
    ###markers, _ = ndi.label(mask)
    ###labels = watershed(-distance, markers, mask=cellMask8)
    ###seg = labels>0
    ###seg = binary_opening(seg,disk(5))
    
    return seg

def movingCellDetection(subfolder,block_size=21,offset=0.015,fps=100.0,interFrame=5):

    ds_time = interFrame 

    imageNameRoot =  subfolder  + "\\tiff\\*.tif"
        #B9, C2,C4
    (dirName,videoFileName) = os.path.split(subfolder)

    imageNameRoot0 = dirName
    imageNames = sorted(glob.glob(imageNameRoot))
    imageNum = len(imageNames)

    fps=100.0
    if imageNum>1500:
        fps=200.0
        
    kk = 0
    stackLen = len(imageNames)//ds_time
    img0 = cv2.imread(imageNames[0])
    img1 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    reg_hei,reg_wid = img1.shape
    imgStack = np.zeros([reg_hei, reg_wid, int(stackLen)+1],dtype =  np.float32)

    for ii in range(0,imageNum,ds_time):
        img0 = cv2.imread(imageNames[ii])
        img1 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        imgStack[:,:,kk]=img1
        kk = kk+1
        
    imgStack.astype(np.float32)
    
    # sample spacing
    T = 1.0 / stackLen*ds_time

    # Number of sample points
    N = stackLen
    xf = fftfreq(N, T)[:N//2]

    freq_range = np.where((np.array(xf)>0.5) & (np.array(xf)<1.5)) ##only check freqency domain from 0.5hz to 1.5hz
    
    imgStackFFT =np.abs(fftn(imgStack, axes=2))
    imgFFTMax= np.max(imgStackFFT[:,:,freq_range[0]],axis=2)
    
    th_log=np.mean(np.log(imgFFTMax))+2.5*np.std(np.log(imgFFTMax))
    
    ###local_thresh = threshold_local(imgFFTMax, block_size, offset)
    
    
    cellMask0 = np.log(imgFFTMax)>th_log 
    
    cellMask1 = binary_closing(cellMask0,disk(1))
    cellMask2 = ndi.binary_fill_holes(cellMask1)
    
    cellMask3 =  remove_small_objects(cellMask2,20)

    cellMask4 = binary_closing(cellMask3,disk(2))
    cellMask5 = ndi.binary_fill_holes(cellMask4)

    seg =  remove_small_objects(cellMask5,30)
    
    return cellMask0,imgFFTMax,seg
    
    
def imreconstruct(marker, mask, SE=disk(3)):
    """
    Description: Constrain mask, continuously expand marker to realize morphological reconstruction, where mask >= marker
    
         Parameter:
                 -marker marker image, single channel / three channel image
                 -mask template image, same as marker
                 -conn Connectivity reconstruction structure element, refer to matlab::imreconstruct::conn parameter, the default is 8 Unicom
    """
    while True:
        marker_pre = marker
        dilation = binary_dilation(marker, SE)
        marker = np.min((dilation, mask), axis=0)
        if (marker_pre == marker).all():
            break
    return marker

def auto_annotation(subfolder):
    print(subfolder)
    cellMask0_mov,_,seg_mov = movingCellDetection(subfolder,block_size=21,offset=0.015)
    seg = cellSegmentation(subfolder,block_size=21,offset=0.015)

    seg_active = imreconstruct(seg_mov,seg,SE=disk(5))
 
    seg_label=label(seg_active)
    seg_box=np.zeros(seg_active.shape,dtype = int)
    props=regionprops(seg_label)
        
    csvOutputName = subfolder+"\\"+ "cardiomyocytes_boxes_automated.csv"
    fout = open(csvOutputName, 'w', newline='')
    writer = csv.writer(fout)
    fout.flush() 
    
    for ii in range(len(props)):
        a = props[ii].bbox[2]-props[ii].bbox[0]
        b = props[ii].bbox[3]-props[ii].bbox[1]
        major = max(a,b)
        minor = min(a,b)
        if major>50:
            startRow = max(1,props[ii].bbox[0]-10)
            endRow = min(seg.shape[0]-1,props[ii].bbox[2]+10)
            startCol = max(1,props[ii].bbox[1]-10)
            endCol = min(seg.shape[1]-1,props[ii].bbox[3]+10)
            seg_box[startRow:endRow,startCol]=1
            seg_box[startRow:endRow,endCol]=1
            seg_box[startRow,startCol:endCol]=1
            seg_box[endRow,startCol:endCol]=1
            writer.writerow((startRow,startCol,endRow,endCol))
            fout.flush()
    
    fout.close()
    
        ###seg_box[props[ii].bbox[0]:props[ii].bbox[2],props[ii].bbox[1]:props[ii].bbox[3]]=1
    seg_box = binary_dilation(seg_box,disk(2))
    seg_box = seg_box.astype(int)*222
    
    origImgName = subfolder+"\\tiff\\frame_001.tif"
    origImg = cv2.imread(origImgName)
    origImg[:,:,0] = seg_box
    outputImageAnnoName = subfolder+"\\frame_001_auto_annotated.tif"
    cv2.imwrite(outputImageAnnoName, origImg)
    
    annotationManName = subfolder+"\\frame_001.tif"
    if os.path.isfile(annotationManName):
        outputImageAnnoComName = subfolder+"\\frame_001_annotationCompare.tif"
        img_man = cv2.imread(annotationManName)
        img_man[:,:,0] = seg_box
        cv2.imwrite(outputImageAnnoComName, img_man)
    

#############################
#############################
#############################

if __name__ == "__main__":

    tic = time.time()

    rootDir = r'Z:\pangj05\TROPONIN2021\20210527MavaSubDataSetAutomation\Plate4_ds'

    outputFolder = rootDir

    if not os.path.isdir(outputFolder):
        print('The OUTPUT directory is not present. Creating a new one..')
        os.mkdir(outputFolder)
            
    subfolders = [os.path.join(rootDir, o) for o in os.listdir(rootDir) if os.path.isdir(os.path.join(rootDir,o))]
    subfolders = sorted(subfolders)

    tic = time.time()
    Parallel(n_jobs=6,prefer='threads')(delayed(auto_annotation)(subfolder) for subfolder in subfolders)  
    toc = time.time()
    print('total time is: ' + str(toc-tic))