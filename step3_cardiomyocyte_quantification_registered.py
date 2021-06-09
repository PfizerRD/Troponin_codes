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
            
def SimilarityComparison(img10, img20):
    img1 = gaussian(img10, sigma=1)
    img2 = gaussian(img20, sigma=1)
   
    reg_hei,reg_wid = img10.shape
    
    N = reg_hei*reg_wid

    norm_term1 = np.sum(np.multiply(img1,img2))*N
    norm_term2 = np.sum(img1)*np.sum(img2)

    denorm_term1 = N*np.sum(np.multiply(img1,img1))-np.sum(img1)*np.sum(img1)
    denorm_term2 = N*np.sum(np.multiply(img2,img2))-np.sum(img2)*np.sum(img2)

    SC = (norm_term1-norm_term2)*(norm_term1-norm_term2)/(denorm_term1*denorm_term2)
    
    return SC


def FitEllipse(cellMask):
    bound = find_boundaries(cellMask)

    bound_ind = np.where(bound>0)
    bound_x = bound_ind[0]
    bound_y = bound_ind[1]
    bound_xy = np.column_stack((bound_x,bound_y))

    ellipse_fit = EllipseModel()
    ellipse_fit.estimate(bound_xy)

    # xc, yc, a, b, theta
    xc,yc,a,b,theta = np.round(ellipse_fit.params, 3)

    img_ellipse = np.zeros(cellMask.shape, dtype=np.uint8)
   
    rr, cc = ellipse(xc,yc,a,b,rotation = theta)
    
    row,col =  cellMask.shape

    ## only keep ellipse within the box region
    rr = rr[np.where(rr<row)]
    cc = cc[np.where(rr<row)]
    
    rr = rr[np.where(cc<col)]
    cc = cc[np.where(cc<col)] 
    img_ellipse[rr, cc] = 1
    return a, b, img_ellipse,theta

    
def SingleCellCropping(imgNames, ind_x1, ind_x2, ind_y1, ind_y2):

    img0 = cv2.imread(imgNames[0])
    img = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    imgSample = img[ind_x1:ind_x2,ind_y1:ind_y2]

    reg_hei,reg_wid = imgSample.shape

    videoLen = len(imgNames)
    regionStack = np.zeros([reg_hei, reg_wid, int(videoLen)-1],dtype =  np.float64)

    for ii in range(int(videoLen)-1):
        frame_rgb =  cv2.imread(imgNames[ii])
        frame = cv2.cvtColor(frame_rgb,cv2.COLOR_BGR2GRAY)
        regionStack[:,:,ii] = frame[ind_x1:ind_x2,ind_y1:ind_y2] 
  
    return imgSample, regionStack


def getLargestCC(segmentation):
    labels = label(segmentation)
    ###assert( labels.max() != 0 ) # assume at least 1 CC
    if labels.max()==0:
        return segmentation
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC


def SingleCellSegmentationFitting(imgSample):
    #cellMask0 = (imgSample>np.mean(imgSample))
    #cellMask0 = imgSample>(0.1*np.median(imgSample)+0.9*np.mean(imgSample))
    ##block_size = 11
    edge_sobel = filters.sobel(imgSample)
    thresh = threshold_otsu(edge_sobel)*0.8
    cellMask0 = edge_sobel>thresh
    
    thresh1 = threshold_otsu(imgSample)
    cmMask = imgSample<thresh1 
    
    ###print(np.sum(cmMask)/(cmMask.shape[0]*cmMask.shape[1]))
    
    if np.sum(cmMask)/(cmMask.shape[0]*cmMask.shape[1])>0.80:
       cmMask=imgSample>thresh1

    cellMask0 = cellMask0 | cmMask

    
    cellMask1 = binary_closing(cellMask0,disk(1))
    cellMask2 = ndi.binary_fill_holes(cellMask1)
    
    cellMask3 =  remove_small_objects(cellMask2,200)

    ###cellMask4 = cellMask3
    cellMask4 = binary_closing(cellMask3,disk(2))
    cellMask5 = ndi.binary_fill_holes(cellMask4)
    cellMask6 = binary_erosion(cellMask5,disk(1))
    cellMask7 = binary_opening(cellMask6,disk(3))

    cellMask8 =  remove_small_objects(cellMask7,400)
    cellMask8[5,:] = 0
    cellMask8[-5,:] = 0
    cellMask8[:,5] = 0
    cellMask8[:,-5] = 0
    
    cellMask9 = clear_border(cellMask8)
    ###cellMask =  remove_small_objects(cellMask9,300)
    
    cellMask = getLargestCC(cellMask9)
    
    bound = find_boundaries(cellMask)
    if np.sum(cellMask)==0:
        return None, None, None, None, None, None
    a, b, ellipseMask,theta = FitEllipse(cellMask)
    majorAxis = max(a,b)
    minorAxis = min(a,b)
    
    return majorAxis, minorAxis, ellipseMask, cellMask, bound, theta


def SingleCellSegmentationBox(imgSample):
    #cellMask0 = (imgSample>np.mean(imgSample))
    #cellMask0 = imgSample>(0.1*np.median(imgSample)+0.9*np.mean(imgSample))
    ##block_size = 11
    edge_sobel = filters.sobel(imgSample)
    thresh = threshold_otsu(edge_sobel)*0.8
    cellMask0 = edge_sobel>thresh
    
    thresh1 = threshold_otsu(imgSample)
    cmMask = imgSample<thresh1 
    
    ###print(np.sum(cmMask)/(cmMask.shape[0]*cmMask.shape[1]))
    
    if np.sum(cmMask)/(cmMask.shape[0]*cmMask.shape[1])>0.80:
       cmMask=imgSample>thresh1

    cellMask0 = cellMask0 | cmMask

    
    cellMask1 = binary_closing(cellMask0,disk(1))
    cellMask2 = ndi.binary_fill_holes(cellMask1)
    
    cellMask3 =  remove_small_objects(cellMask2,200)

    ###cellMask4 = cellMask3
    cellMask4 = binary_closing(cellMask3,disk(2))
    cellMask5 = ndi.binary_fill_holes(cellMask4)
    cellMask6 = binary_erosion(cellMask5,disk(1))
    cellMask7 = binary_opening(cellMask6,disk(3))

    cellMask8 =  remove_small_objects(cellMask7,400)
    cellMask8[5,:] = 0
    cellMask8[-5,:] = 0
    cellMask8[:,5] = 0
    cellMask8[:,-5] = 0
    
    cellMask9 = clear_border(cellMask8)
    ###cellMask =  remove_small_objects(cellMask9,300)
    
    cellMask = getLargestCC(cellMask9)
    
    bound = find_boundaries(cellMask)
    if np.sum(cellMask)==0:
        return None, None, None, None, None, None
    a, b, ellipseMask,theta = FitEllipse(cellMask)
    ###majorAxis = max(a,b)
    ###minorAxis = min(a,b)
    
    xy = np.nonzero(bound)
    a = max(xy[0])-min(xy[0])
    b = max(xy[1])-min(xy[1])
    majorAxis = max(a,b)/2
    minorAxis = min(a,b)/2
    
    return majorAxis, minorAxis, ellipseMask, cellMask, bound, theta


def SeriesCellSegmentationFitting(regionStack,output_display = 0):
    videoLen = regionStack.shape[2]
    majorValues = np.zeros(int(videoLen),dtype =  np.float64)
    minorValues = np.zeros(int(videoLen),dtype =  np.float64)
    ellipseMasks = np.zeros(regionStack.shape,dtype=np.uint8)
    cellMasks = np.zeros(regionStack.shape,dtype=np.uint8)
    bounds = np.zeros(regionStack.shape,dtype=np.uint8)
    thetas = np.zeros(int(videoLen),dtype =  np.float64)
    
    for ii in range(int(videoLen)):
        if output_display:
            print("Frame num: " + str(ii))
        imgSample = regionStack[:,:,ii]
       
        majorAxis, minorAxis, ellipseMask, cellMask, bound, theta = SingleCellSegmentationFitting(imgSample)
        
        if majorAxis is None:
            majorValues[ii] = majorValues[ii-1]
            minorValues[ii] = minorValues[ii-1]
            ellipseMasks[:,:,ii] = ellipseMasks[:,:,ii-1]
            bounds[:,:,ii] = bounds[:,:,ii-1]
            cellMasks[:,:,ii] = cellMasks[:,:,ii-1]
            thetas[ii] = thetas[ii-1]
        else:
            majorValues[ii] = majorAxis
            minorValues[ii] = minorAxis
            ellipseMasks[:,:,ii] = ellipseMask
            bounds[:,:,ii] = bound
            cellMasks[:,:,ii] = cellMask
            thetas[ii] = theta
        
    return majorValues,  minorValues, ellipseMasks, bounds, cellMasks, thetas
        

def SeriesCellSegmentationBox(regionStack,output_display = 0):
    videoLen = regionStack.shape[2]
    majorValues = np.zeros(int(videoLen),dtype =  np.float64)
    minorValues = np.zeros(int(videoLen),dtype =  np.float64)
    ellipseMasks = np.zeros(regionStack.shape,dtype=np.uint8)
    cellMasks = np.zeros(regionStack.shape,dtype=np.uint8)
    bounds = np.zeros(regionStack.shape,dtype=np.uint8)
    thetas = np.zeros(int(videoLen),dtype =  np.float64)
    
    for ii in range(int(videoLen)-1):
        if output_display:
            print("Frame num: " + str(ii))
        imgSample = regionStack[:,:,ii]
       
        majorAxis, minorAxis, ellipseMask, cellMask, bound, theta = SingleCellSegmentationBox(imgSample)
        
        if majorAxis is None:
            majorValues[ii] = majorValues[ii-1]
            minorValues[ii] = minorValues[ii-1]
            ellipseMasks[:,:,ii] = ellipseMasks[:,:,ii-1]
            bounds[:,:,ii] = bounds[:,:,ii-1]
            cellMasks[:,:,ii] = cellMasks[:,:,ii-1]
            thetas[ii] = thetas[ii-1]
        else:
            majorValues[ii] = majorAxis
            minorValues[ii] = minorAxis
            ellipseMasks[:,:,ii] = ellipseMask
            bounds[:,:,ii] = bound
            cellMasks[:,:,ii] = cellMask
            thetas[ii] = theta
        
    return majorValues,  minorValues, ellipseMasks, bounds, cellMasks, thetas    

def SegmentationDisplayOutput(regionStack,bounds,ellipseMasks,videoName,tag,register=0, inter = 1):
    
    (dirName,videoFileName) = os.path.split(videoName)

    if register:
        outputFolder = dirName +'\\' + videoFileName[:-4]+"_segmentation_cell_"+str(tag)+"_registered"
    else:
        outputFolder = dirName +'\\' + videoFileName[:-4]+"_segmentation_cell_"+str(tag)
    
    if not os.path.isdir(outputFolder):
        print('The DISPLAY directory is not present. Creating a new one..')
        os.mkdir(outputFolder)
        
    ###plt.rcParams['figure.figsize'] = [12, 8]
    
    for ind in range(0,regionStack.shape[2],inter):
   
        fig, (ax1,ax2,ax3) = plt.subplots(1,3)

        imgSample = regionStack[:,:,ind]

        ###plt.subplot(131)
        ax1.imshow(imgSample)

        ###ax = plt.subplot(132)
        ax2.imshow(imgSample)

        bound_x, bound_y = np.where(bounds[:,:,ind]>0)

        ax2.plot(bound_y,bound_x,'.r')

        if register == 0:
            ###ax = plt.subplot(133)
            ax3.imshow(imgSample)

            bound_x, bound_y = np.where(bounds[:,:,ind]>0)

            ellipse_bound = find_boundaries(ellipseMasks[:,:,ind])
            ellipse_bound_x, ellipse_bound_y = np.where(ellipse_bound>0)
            ax3.plot(bound_y,bound_x,'.r')
            ax3.plot(ellipse_bound_y,ellipse_bound_x,'.b')
        else:
            ###ax = plt.subplot(133)
            xy = np.nonzero(bounds[:,:,ind])
            if len(xy[0])>0:
                a = max(xy[1])
                b = min(xy[1])
                imgSample[:,a] = 0
                imgSample[:,b] = 0
            ax3.imshow(imgSample)
         
        displayImageName = outputFolder+"\\frame_"+str(ind).zfill(4)+".jpg"

        fig.savefig(displayImageName)
        fig.clf()    # normally I use these lines to release the memory
        plt.close(fig)
        gc.collect()


def SingleCellForceCalculation(videoName, imgNames, ind_x1,ind_x2,ind_y1,ind_y2,tag,display=1):

    imgSample, regionStack = SingleCellCropping(imgNames, ind_x1,ind_x2,ind_y1,ind_y2)
    majorValues, minorValues, ellipseMasks, bounds, cellMasks, thetas = SeriesCellSegmentationFitting(regionStack)

    majorValues_valid = majorValues[(majorValues>np.median(majorValues)*0.5) & (majorValues<np.median(majorValues)*1.5)]
    lenMax = 2*np.percentile(majorValues,98)
    lenMin = 2*np.percentile(majorValues,2)
    lenRelaxation = 2*np.percentile(majorValues,98)

    frames_relax = 2*majorValues>lenRelaxation

    labels_relax = label(frames_relax)

    labelSize = np.zeros(max(labels_relax))
    label_ref = 0
 
    for ii in range(1,max(labels_relax)+1): # what if the longest label happens at end? Does it matter? 
        labelSize[ii-1] = sum(labels_relax==ii)
        if labelSize[ii-1]>5:
            label_ref = ii
            break
            
    if label_ref == 0:
    
        ### assert (label_ref > 0),"NO SUITABLE RELAXATION REFERENCE FRAMES!!"
        inds_ref = np.array(range(np.argmax(majorValues)-2,np.argmax(majorValues)+2))
    else:    
        inds_ref =  np.where(labels_relax == label_ref)[0]

    videoLen = regionStack.shape[2]

    SC_values_ref = np.zeros([len(inds_ref),int(videoLen)],dtype =  np.float64)
    for ii in range(len(inds_ref)):
        img1 = regionStack[:,:,inds_ref[ii]-1]
        for jj in range(int(videoLen)):
            img2 = regionStack[:,:,jj]
            SC_values_ref[ii,jj] = SimilarityComparison(img1,img2)
        
    SC_baseline_ref = np.mean(SC_values_ref,axis=0)
    
    SC_min = np.min(SC_baseline_ref)
    SC_max = np.max(SC_baseline_ref)
    NR = (1-SC_min/SC_max)/(1-lenMin/lenMax)

    ###length = 100*(1-(1-SC_baseline_ref/SC_max)/NR)
    ### Modified by Pangjc to normalize length to the REAL cardiomyocytes length (unit: pixel) 04/21/2021
    length0 = 100*(1-(1-SC_baseline_ref/SC_max)/NR)
    length = lenMin+(lenMax-lenMin)*(length0-np.min(length0))/(np.max(length0)-np.min(length0))
      
    (dirName,videoFileName) = os.path.split(videoName)
    
    if display:

        ###plt.rcParams['figure.figsize'] = [8, 8]
        fig, (ax1,ax2,ax3)= plt.subplots(3,1)

        ###plt.subplot(311)
        ax1.plot(2*majorValues[:-1])
        ax1.set_title('Major Axis length (Unit: pixel)')
        ###plt.subplot(312)
        ax2.plot(frames_relax[-1],'r')
        ax2.plot(inds_ref,np.ones(len(inds_ref)),'b',lw=5)
        ax2.set_title('Relaxation estimation')

        ###plt.subplot(313)
        ax3.plot(2*minorValues[-1],'r')
        ax3.set_title('Minor Axis length (Unit: pixel)')

        displayFigName1 = dirName +'\\' + videoFileName[:-4]+"_segmentation_cell_"+str(tag)+"_quantification_part1.jpg"
        fig.savefig(displayFigName1)
        fig.clf()
        plt.close(fig)
        
        print('max length: ' + str(lenMax))
        print('min length: ' + str(lenMin))
        print('NR: ' + str(NR))
        print('relaxation length: ' + str(lenRelaxation))

        ###plt.rcParams['figure.figsize'] = [8, 8]
        fig, (ax1,ax2) = plt.subplots(2,1)
        ###plt.subplot(211)
        ax1.hist(2*majorValues_valid, bins=30)
        ###plt.subplot(212)
        ax2.hist(2*minorValues, bins=30)
        
        displayFigName2 = dirName +'\\' + videoFileName[:-4]+"_segmentation_cell_"+str(tag)+"_quantification_part2.jpg"
        fig.savefig(displayFigName2)
        fig.clf()    # normally I use these lines to release the memory
        plt.close(fig)
        gc.collect()

        ###plt.rcParams['figure.figsize'] = [12, 8]
        fig, (ax1,ax2) = plt.subplots(2,1)
      
        ###plt.subplot(211)      
        ax1.imshow(SC_values_ref)
        ax1.set_xlim(0,SC_values_ref.shape[1])
        ax1.set_title("Intensity Similairy Comparison for REFERENCE FRAMES")

        ###plt.subplot(212)
        ###plt.rcParams['figure.figsize'] = [12, 8]
        SC_baseline_ref = np.mean(SC_values_ref,axis=0)

        ax2.plot(SC_baseline_ref)
        ax2.plot(inds_ref-1,SC_baseline_ref[inds_ref-1] ,'r',lw=5)
        ax2.set_xlim(0,SC_values_ref.shape[1])

        ax2.set_title("Intensity Similariy PROFILE for REFERENCE FRAMES")
        
        displayFigName3 = dirName +'\\' + videoFileName[:-4]+"_segmentation_cell_"+str(tag)+"_quantification_part3.jpg"
        fig.savefig(displayFigName3)
        fig.clf()
        plt.close(fig)

        speed = np.diff(length)
        maxV = np.max(speed)
        minV = np.min(speed)
        peaks0, _ = find_peaks(speed, height=maxV*0.2,distance=30)
        peaks1, _ = find_peaks(-speed, height=-minV*0.2,distance=30)
        
        ###plt.rcParams['figure.figsize'] = [12, 8]
        fig, (ax1,ax2)= plt.subplots(2,1)
        
        ###plt.subplot(211)
        ax1.plot(length)
        ax1.set_title('LENGTH (Unit: pixel)')
        ###plt.subplot(212)
        ax2.plot(speed)
        ax2.plot(peaks0,speed[peaks0], "x")
        ax2.plot(peaks1,speed[peaks1], "o")
        ax2.set_title('SPEED (Unit: pixel/time)')
   
        displayFigName4 = dirName +'\\' + videoFileName[:-4]+"_segmentation_cell_"+str(tag)+"_quantification_part4.jpg"
        fig.savefig(displayFigName4)
        fig.clf()
        plt.close(fig)
        
    return majorValues, minorValues,length, bounds, cellMasks, regionStack, ellipseMasks, frames_relax, inds_ref,SC_values_ref, thetas


def SingleCellRegistration(regionStack, ellipseMasks, majorValues, majorExtra,minorExtra, rotate=0):
    
    medianMajorValue = np.median(majorValues)
    ###print('median major value: ' + str(medianMajorValue))
    ind_ref = np.argmin(abs(majorValues-medianMajorValue))
    ###print('median index: ' + str(ind_ref))

    #ellipseMask_Ref = ellipseMasks[:,:,1]
    ellipseMask_Ref = ellipseMasks[:,:,ind_ref]
    props = regionprops(ellipseMask_Ref)
    box_prop = props[0]
    y0,x0 = box_prop.centroid

    boxLength = box_prop.major_axis_length + majorExtra
    boxWidth = box_prop.minor_axis_length + minorExtra

    orientation = box_prop.orientation

    if rotate:
        orientation=orientation+np.pi/2
 
    x_M1 = x0 + math.cos(orientation) * 0.5 * (boxLength)
    y_M1 = y0 - math.sin(orientation) * 0.5 * (boxLength)

    x_M2 = x0 - math.cos(orientation) * 0.5 * (boxLength)
    y_M2 = y0 + math.sin(orientation) * 0.5 * (boxLength)

    x_m1 = x0 - math.sin(orientation) * 0.5 * (boxWidth)
    y_m1 = y0 - math.cos(orientation) * 0.5 * (boxWidth)

    x_m2 = x0 + math.sin(orientation) * 0.5 * (boxWidth)
    y_m2 = y0 + math.cos(orientation) * 0.5 * (boxWidth)  

    regionStackRegister = np.zeros([round(boxWidth), round(boxLength), regionStack.shape[2]], dtype = np.float64)
    padValues = np.percentile(regionStack, 70)
    ###regionStackRegister = padValues*np.ones([round(boxWidth), round(boxLength), regionStack.shape[2]],dtype =  np.float64)
    
    src = np.array([[0, boxWidth*0.5], [boxLength*0.5, 0], [boxLength, 0.5*boxWidth], [boxLength*0.5, boxWidth]])
    dst = np.array([[x_M2, y_M2], [x_m2, y_m2], [x_M1, y_M1], [x_m1, y_m1]])

    tform3 = tf.ProjectiveTransform()
    tform3.estimate(src, dst)
    
    print('transformation matrix: ')
    print(tform3.params)
    for ii in range(regionStack.shape[2]):
        regionStackRegister[:,:,ii] = tf.warp(regionStack[:,:,ii], tform3, cval = padValues, output_shape=(round(boxWidth), round(boxLength)))
        
    return regionStackRegister


def SingleCellForceCalculationBox(regionBoxStack,regionStack,videoName,tag,display=0):

    majorValues_box, minorValues_box, ellipseMasks_box, bounds_box, cellMasks_box, thetas_box = SeriesCellSegmentationBox(regionBoxStack)
    
    majorValues, minorValues, ellipseMasks, bounds, cellMasks, thetas = SeriesCellSegmentationFitting(regionStack)

    #majorValues_valid = majorValues[abs(majorValues-majorValues_rough)<3] # Get rid of unstable/wrong estimations of major axis
    majorValues_valid_box = majorValues_box[(majorValues_box>np.median(majorValues_box)*0.5) & (majorValues_box<np.median(majorValues_box)*1.5)]
    
    lenMax_box = 2*np.percentile(majorValues_valid_box,98)
    lenMin_box = 2*np.percentile(majorValues_valid_box,2)
    lenRelaxation_box = 2*np.percentile(majorValues_valid_box,98)
      
    majorValues_valid = majorValues[(majorValues>np.median(majorValues)*0.5) & (majorValues<np.median(majorValues)*1.5)]
    
    lenMax = 2*np.percentile(majorValues,98)
    lenMin = 2*np.percentile(majorValues,2)
    lenRelaxation = 2*np.percentile(majorValues,98)

    frames_relax = 2*majorValues>lenRelaxation

    labels_relax = label(frames_relax)

    labelSize = np.zeros(max(labels_relax))
    label_ref = 0
    for ii in range(1,max(labels_relax)+1): # what if the longest label happens at end? Does it matter? 
        labelSize[ii-1] = sum(labels_relax==ii)
        if labelSize[ii-1]>5:
            label_ref = ii
            break
            
    if label_ref == 0:
    
        ### assert (label_ref > 0),"NO SUITABLE RELAXATION REFERENCE FRAMES!!"
        inds_ref = np.array(range(np.argmax(majorValues)-2,np.argmax(majorValues)+2))
    else:    
        inds_ref =  np.where(labels_relax == label_ref)[0]

    videoLen = regionStack.shape[2]

    SC_values_ref = np.zeros([len(inds_ref),int(videoLen)],dtype =  np.float64)
    for ii in range(len(inds_ref)):
        img1 = regionStack[:,:,inds_ref[ii]]
        for jj in range(int(videoLen)):
            img2 = regionStack[:,:,jj]
            SC_values_ref[ii,jj] = SimilarityComparison(img1,img2)
        
    SC_baseline_ref = np.mean(SC_values_ref,axis=0)
    
    SC_min = np.min(SC_baseline_ref)
    SC_max = np.max(SC_baseline_ref)
    NR = (1-SC_min/SC_max)/(1-lenMin/lenMax)

    ###length = 100*(1-(1-SC_baseline_ref/SC_max)/NR)
    ### Modified by Pangjc to normalize length to the REAL cardiomyocytes length (unit: pixel) 04/21/2021
    ### Modified by Pangjc to adjust to horizontal measurement
    length0 = 100*(1-(1-SC_baseline_ref/SC_max)/NR)
    length = lenMin_box+(lenMax_box-lenMin_box)*(length0-np.min(length0))/(np.max(length0)-np.min(length0))
    
    (dirName,videoFileName) = os.path.split(videoName)
        
    if display:

        ###plt.rcParams['figure.figsize'] = [8, 8]
        fig, (ax1,ax2,ax3) = plt.subplots(3,1)

        ###plt.subplot(311)
        ax1.plot(2*majorValues_box[:-1])
        ax1.set_title('Major Axis length (Unit: pixel)')
        ###plt.subplot(312)
        ax2.plot(frames_relax[-1],'r')
        ax2.plot(inds_ref,np.ones(len(inds_ref)),'b',lw=5)
        ax2.set_title('Relaxation estimation')

        ###plt.subplot(313)
        ax3.plot(2*minorValues_box[-1],'r')
        ax3.set_title('Minor Axis length (Unit: pixel)')

        displayFigName1 = dirName +'\\' + videoFileName[:-4]+"_segmentation_cell_"+str(tag)+"_quantification_register_part1.jpg"
        fig.savefig(displayFigName1)
        fig.clf()
        plt.close(fig)
        
        print('max length: ' + str(lenMax))
        print('min length: ' + str(lenMin))
        print('NR: ' + str(NR))
        print('relaxation length: ' + str(lenRelaxation))

        ###plt.rcParams['figure.figsize'] = [8, 8]
        fig, (ax1,ax2 )= plt.subplots(2,1)
        ###plt.subplot(211)
        ax1.hist(2*majorValues_valid_box, bins=30)
        ###plt.subplot(212)
        ax2.hist(2*minorValues_box, bins=30)
        
        displayFigName2 = dirName +'\\' + videoFileName[:-4]+"_segmentation_cell_"+str(tag)+"_quantification_register_part2.jpg"
        fig.savefig(displayFigName2)
        fig.clf()
        plt.close(fig)

        ###plt.rcParams['figure.figsize'] = [12, 8]
        
        fig, (ax1,ax2) = plt.subplots(2,1)
        
        ax1.imshow(SC_values_ref)
        ax1.set_xlim(0,SC_values_ref.shape[1])
        ax1.set_title("Intensity Similairy Comparison for REFERENCE FRAMES")

        ###plt.subplot(212)
        ###plt.rcParams['figure.figsize'] = [12, 8]
        SC_baseline_ref = np.mean(SC_values_ref,axis=0)

        ax2.plot(SC_baseline_ref)
        ax2.plot(inds_ref-1,SC_baseline_ref[inds_ref-1] ,'r',lw=5)
        ax2.set_xlim(0,SC_values_ref.shape[1])

        ax2.set_title("Intensity Similariy PROFILE for REFERENCE FRAMES")
        
        displayFigName3 = dirName +'\\' + videoFileName[:-4]+"_segmentation_cell_"+str(tag)+"_quantification_register_part3.jpg"
        fig.savefig(displayFigName3)
        fig.clf()
        plt.close(fig)

        speed = np.diff(length)
        maxV = np.max(speed)
        minV = np.min(speed)
        peaks0, _ = find_peaks(speed, height=maxV*0.2,distance=30)
        peaks1, _ = find_peaks(-speed, height=-minV*0.2,distance=30)
        
        fig, (ax1,ax2)= plt.subplots(2,1)
        ###plt.rcParams['figure.figsize'] = [12, 8]
        ###plt.subplot(211)
        ax1.plot(length)
        ax1.set_title('LENGTH (Unit: pixel)')
        ###plt.subplot(212)
        ax2.plot(speed)
        ax2.plot(peaks0,speed[peaks0], "x")
        ax2.plot(peaks1,speed[peaks1], "o")
        ax2.set_title('SPEED (Unit: pixel/time)')
        
        displayFigName4 = dirName +'\\' + videoFileName[:-4]+"_segmentation_cell_"+str(tag)+"_quantification_register_part4.jpg"
        fig.savefig(displayFigName4)
        fig.clf()
        plt.close(fig)
    
    return majorValues_box, minorValues_box,length, bounds_box, cellMasks_box,ellipseMasks_box, frames_relax, inds_ref,SC_values_ref, thetas_box

def estimateTimePoints(videoName, tag, length,speed,peaks,leftBound,rightBound,th = 0.7,register=0):
    A_list = []
    B_list = []
    C_list = []
    D_list = []
    E_list = []
    F_list = []
    J_list = []
    I_list = []
   
    (dirName,videoFileName) = os.path.split(videoName) 
    for ii in range(len(leftBound)):
        if leftBound[ii]<5 or rightBound[ii]>len(speed):
            continue
        maxV = np.max(speed[leftBound[ii]:rightBound[ii]])
        minV = np.min(speed[leftBound[ii]:rightBound[ii]])       
        pos_ind = peaks[ii]+np.argmax(speed[peaks[ii]:rightBound[ii]])
        neg_ind = leftBound[ii]+np.argmin(speed[leftBound[ii]:peaks[ii]])
        A_ind = leftBound[ii]+np.argmin(np.abs(minV*th-speed[leftBound[ii]:neg_ind]))
        C_ind = neg_ind+np.argmin(np.abs(minV*th-speed[neg_ind:pos_ind]))
        
        D_ind = neg_ind+np.argmin(np.abs(maxV*th-speed[neg_ind:pos_ind]))
        F_ind = pos_ind+np.argmin(np.abs(maxV*th-speed[pos_ind:rightBound[ii]]))
        
        length_min = length[peaks[ii]]
        length_base = np.percentile(length[leftBound[ii]:rightBound[ii]],95)
        
        dH = length_base-length_min
        
        I_ind = peaks[ii]+np.argmin(np.abs(dH*th-(length[peaks[ii]:rightBound[ii]]-length_min)))
        
        A_list.append(A_ind)
        B_list.append(neg_ind)
        C_list.append(C_ind)
        D_list.append(D_ind)
        E_list.append(pos_ind)
        F_list.append(F_ind)
        J_list.append(peaks[ii])
        I_list.append(I_ind)
        
    print(A_list)
    print(B_list)
    print(C_list)
    print(D_list)
    print(E_list)
    print(F_list)
    print(J_list)
    print(I_list)
    
    As = np.array(A_list)
    Bs = np.array(B_list)
    Cs = np.array(C_list)
    Ds = np.array(D_list)
    Es = np.array(E_list)
    Fs = np.array(F_list)
    Js = np.array(J_list)
    Is = np.array(I_list)
    
    As = As[:-1]
    Bs = Bs[:-1]
    Cs = Cs[:-1]
    Ds = Ds[:-1]
    Es = Es[:-1]
    Fs = Fs[:-1]
    Js = Js[:-1]
    Is = Is[:-1]
    
    
    if len(As):
        fig, (ax1,ax2) = plt.subplots(2,1)
        ###plt.rcParams['figure.figsize'] = [28, 10]
        ###plt.subplot(211)
        ax1.plot(length)
        ax1.plot(Is,length[Is], "x")
        ax1.plot(Is,length[Is-1], "d")
        ax1.plot(Is,length[Is+1], "d")
        ax1.plot(Js,length[Js], "o")
      
        ax1.set_title('NORMALIZED LENGTH CHANGE')
        ###plt.subplot(212)
        ax2.plot(speed)
        ax2.plot(As,speed[As], "x")
        ax2.plot(Bs,speed[Bs], "x")
        ax2.plot(Cs,speed[Cs], "x")
        ax2.plot(Ds,speed[Ds], "D")
        ax2.plot(Es,speed[Es], "D")
        ax2.plot(Fs,speed[Fs], "D")
        ax2.set_title('NORMALIZED SPEED'+ str(th))
        if register:
            displayFigName5 = dirName +'\\' + videoFileName[:-4]+"_segmentation_cell_"+str(tag)+"_quantification_register_part5_th_"+str(round(th*100)) +".jpg"
        else:
            displayFigName5 = dirName +'\\' + videoFileName[:-4]+"_segmentation_cell_"+str(tag)+"_quantification_part5_th_"+str(round(th*100)) +".jpg"
        print("point detection:")
        print(displayFigName5)
        fig.savefig(displayFigName5)
        fig.clf()
        plt.close(fig)
       
    return As,Bs,Cs,Ds,Es,Fs,Js,Is 
        
def CM_pipeline(rootDir,rotate,relax_th,subFolder):
    ###subFolders = sorted(list(listdir_nohidden(rootDir)))
    # Set output the csv file 
    print(subFolder)
    csvOutputName =rootDir+"\\"+"contractivity_relaxation_features_unreg_relax_percent_"+str(round(relax_th*100))+"_"+subFolder+".csv"
    csvOutputName_r =rootDir+"\\"+"contractivity_relaxation_features_reg_relax_percent_"+str(round(relax_th*100))+"_"+subFolder+".csv"
    #print(csvOutputName)
  
    fout = open(csvOutputName, 'w', newline='')
    fout_r = open(csvOutputName_r, 'w', newline='')

    writer = csv.writer(fout)
    writer_r = csv.writer(fout_r)


    writer.writerow( ('subFolder','cellTag', 'A_frame', 'B_frame','C_frame','D_frame','E_frame','F_frame','J_frame','I_frame',
                      'A_value','A_value_minus','A_value_plus','B_value','B_Value_minus','B_value_plus',
                      'C_value','C_value_minus','C_value_plus','D_value','D_Value_minus','D_value_plus',
                      'E_value','E_value_minus','E_value_plus','F_value','F_Value_minus','F_value_plus',
                      'J_value','J_value_minus','J_value_plus','I_value','I_Value_minus','I_value_plus','total_frame_num') )

    writer_r.writerow( ('subFolder','cellTag', 'A_frame', 'B_frame','C_frame','D_frame','E_frame','F_frame','J_frame','I_frame',
                      'A_value','A_value_minus','A_value_plus','B_value','B_Value_minus','B_value_plus',
                      'C_value','C_value_minus','C_value_plus','D_value','D_Value_minus','D_value_plus',
                      'E_value','E_value_minus','E_value_plus','F_value','F_Value_minus','F_value_plus',
                      'J_value','J_value_minus','J_value_plus','I_value','I_Value_minus','I_value_plus','total_frame_num') )
        
    fout.flush()
    fout_r.flush()    

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
        return 
    boxInfo = pandas.read_csv(boxInfoName,header = None) 

    # displaying the contents of the CSV file 
    cellNum = boxInfo.shape[0]
    
    imgFolder = outputFolder+"\\tiff\\*.tif"
    print("folder: ")
    print(imgFolder)
    imgNames = sorted(glob.glob(imgFolder))
    frameNum = len(imgNames)

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
       
        display = 1
        majorValues1, minorValues1,length1, bounds1, cellMasks1, regionStack1,ellipseMasks1, frames_relax1, inds_ref1, SC_values_ref1, thetas1 = SingleCellForceCalculation(videoName, imgNames,
                                                                                          ind_x1,ind_x2,ind_y1,ind_y2,tag,display)
        if 1:
            SegmentationDisplayOutput(regionStack1,bounds1,ellipseMasks1,videoName,tag,register=0,inter = 50)

        majorExtra = 30
        minorExtra = 25

        speed1 = np.diff(length1)
        maxV1 = np.max(speed1)
        minV1 = np.min(speed1)

        force1 = np.max(length1)-length1
        maxForce1 = np.max(force1)

        peaks1, _ = find_peaks(force1, height=maxForce1*0.3,distance=80)

        half_width1 = 70 # related to sample freqency
        leftBound1 = peaks1-half_width1
        rightBound1 = peaks1+half_width1

        try: # for cases there are no regular pattern detected
            As, Bs, Cs, Ds, Es, Fs, Js, Is = estimateTimePoints(videoName,tag,length1,speed1,peaks1,leftBound1,rightBound1)
        except ValueError:
            As,Bs,Cs,Ds,Es,Fs, Js, Is = [0],[0],[0],[0],[0],[0],[0],[0]
        ###writer.writerow( ('subFolder','cellTag', 'A_frame', 'B_frame','C_frame','D_frame','E_frame','F_frame',
        ###         'A_value','A_value_minus','A_value_plus','B_value','B_Value_minus','B_value_plus',
        ###         'C_value','C_value_minus','C_value_plus','D_value','D_Value_minus','D_value_plus',
        ###         'E_value','E_value_minus','E_value_plus','F_value','F_Value_minus','F_value_plus') )
                
        for jj in range(len(As)):       
            writer.writerow((subFolder, tag, As[jj], Bs[jj], Cs[jj], Ds[jj], Es[jj], Fs[jj],Js[jj],Is[jj],
                          speed1[As[jj]],speed1[As[jj]-1],speed1[As[jj]+1],
                          speed1[Bs[jj]],speed1[Bs[jj]-1],speed1[Bs[jj]+1],
                          speed1[Cs[jj]],speed1[Cs[jj]-1],speed1[Cs[jj]+1],
                          speed1[Ds[jj]],speed1[Ds[jj]-1],speed1[Ds[jj]+1],
                          speed1[Es[jj]],speed1[Es[jj]-1],speed1[Es[jj]+1],
                          speed1[Fs[jj]],speed1[Fs[jj]-1],speed1[Fs[jj]+1],
                          length1[Js[jj]],length1[Js[jj]-1],length1[Js[jj]+1],
                          length1[Is[jj]],length1[Is[jj]-1],length1[Is[jj]+1],frameNum))
    
        ####
        #### registered
        ####
        cellMaskMedian = np.median(cellMasks1,axis=2)
        if np.sum(cellMaskMedian)<500:
            continue
            
        cellStack = SingleCellRegistration(regionStack1, ellipseMasks1, majorValues1, majorExtra,minorExtra,rotate)
       
        try:
            majorValues, minorValues,length, bounds, cellMasks,ellipseMasks, frames_relax, inds_ref,SC_values_ref, thetas = SingleCellForceCalculationBox(cellStack,regionStack1,videoName,tag,display=1)
        except:
            continue

        if 1:
            SegmentationDisplayOutput(cellStack,bounds,ellipseMasks,videoName,tag,register=1,inter = 50)
        
        speed = np.diff(length)
        maxV = np.max(speed)
        minV = np.min(speed)

        force = np.max(length)-length
        maxForce = np.max(force)

        peaks, _ = find_peaks(force, height=maxForce*0.3,distance=80)

        half_width = 70 # related to sample freqency
        leftBound = peaks-half_width
        rightBound = peaks+half_width
        
        try: # for cases there are no regular pattern detected
            As_r, Bs_r, Cs_r, Ds_r, Es_r, Fs_r, Js_r,Is_r = estimateTimePoints(videoName,tag,length,speed,peaks,leftBound,rightBound,th=relax_th,register=1)
        except ValueError:
            As_r,Bs_r,Cs_r,Ds_r,Es_r,Fs_r, Js_r,Is_r = [0],[0],[0],[0],[0],[0],[0],[0]
        
        
        reg_periods = min(len(As_r),len(Bs_r),len(Cs_r),len(Ds_r),len(Es_r),len(Fs_r),len(Js_r),len(Is_r))
        for jj in range(reg_periods-1):
            writer_r.writerow((subFolder, tag, As_r[jj], Bs_r[jj], Cs_r[jj], Ds_r[jj], Es_r[jj], Fs_r[jj], Js_r[jj],Is_r[jj],
                          speed[As_r[jj]],speed[As_r[jj]-1],speed[As_r[jj]+1],
                          speed[Bs_r[jj]],speed[Bs_r[jj]-1],speed[Bs_r[jj]+1],
                          speed[Cs_r[jj]],speed[Cs_r[jj]-1],speed[Cs_r[jj]+1],
                          speed[Ds_r[jj]],speed[Ds_r[jj]-1],speed[Ds_r[jj]+1],
                          speed[Es_r[jj]],speed[Es_r[jj]-1],speed[Es_r[jj]+1],
                          speed[Fs_r[jj]],speed[Fs_r[jj]-1],speed[Fs_r[jj]+1],
                          length[Js_r[jj]],length[Js_r[jj]-1],length[Js_r[jj]+1],
                          length[Is_r[jj]],length[Is_r[jj]-1],length[Is_r[jj]+1],frameNum))
            

        del(cellMasks1)
        del(regionStack1)
        del(bounds1)
        del(ellipseMasks1)
        del(majorValues1)
        del(minorValues1)
        gc.collect()
    fout_r.flush()  
    fout_r.close() 
    fout.flush()
    fout.close()
    
    
if __name__ == "__main__":

        rootDir = r'Z:\pangj05\TROPONIN2021\20210527DataSetAnalysis\Plate2'

        rotate=1 ## rotate=1 if running on vm test1
 
        cpu_num = 12
        relax_th = 0.7

        subFolders = sorted(list(listdir_nohidden(rootDir)))
       
        ###CM_pipeline(rootDir,rotate,relax_th,subFolders)
        Parallel(n_jobs=cpu_num,prefer='threads')(delayed(CM_pipeline)(rootDir,rotate,relax_th,subFolder) for subFolder in subFolders[84:96])    