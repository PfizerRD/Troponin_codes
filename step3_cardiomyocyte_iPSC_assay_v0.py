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

import csv
import cv2
import numpy as np
from opt_flow import draw_flow

from scipy import ndimage as ndi

import math
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, binary_opening, binary_dilation, disk,remove_small_objects,label,binary_erosion
from skimage.color import label2rgb
from skimage import data, color, io, img_as_float

from scipy.signal import find_peaks
from skimage.filters import gaussian

import math
import pandas as pd


from memory_profiler import profile
import multiprocessing
from joblib import Parallel, delayed

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.') and not f.startswith('contractivity'):
            yield f

def getLargestCC(segmentation):
    labels = label(segmentation)
    ###assert( labels.max() != 0 ) # assume at least 1 CC
    if labels.max()==0:
        return segmentation
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

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

def iPSC_pipeline(RootPath,OutputPath,subfolder,ds=1):
   
    subfolder = RootPath + "\\" + subfolder
    (dirName,videoFileName) = os.path.split(subfolder)
    print(subfolder)
    if not os.path.exists(OutputPath):
            os.mkdir(OutputPath)
            print("Directory " , OutputPath ,  " Created ")
    else:    
            print("Directory " , OutputPath ,  " already exists")


    csvOutputName =OutputPath+"\\"+videoFileName+"_endpoints.csv"
  
    fout = open(csvOutputName, 'w', newline='')

    writer = csv.writer(fout)

    writer.writerow( ('subFolder','Optical_A_index', 'Optical_B_index', 'OpticalFlow_A_value','OpticalFlow_B_value','Correlation_diff_A_index','Correlation_diff_B_index','Correlation_diff_A_value','Correlation_diff_B_value','Optical_Flow_baseline(5%)') )
        
    fout.flush()  
    
    imageNameRoot =  subfolder  + "\\tiff\\*.tif"
    
    imageNames = sorted(glob.glob(imageNameRoot))
    imageNum = len(imageNames)
    print(imageNum)
    print(imageNames[0])
    img0 = cv2.imread(imageNames[0])
    frame1 = img0[::ds,::ds,:]
    frame_ref = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    hei, wid = prvs.shape

    magStack = np.zeros([hei, wid, int(imageNum)-1],dtype =  np.float32)
    angStack = np.zeros([hei, wid, int(imageNum)-1],dtype =  np.float32)
    SC_values_ref = np.zeros(int(imageNum)-1)
    
    for ii in range(1,imageNum-1):

        img2 = cv2.imread(imageNames[ii])
        frame20 = img2[::ds,::ds,:]
        frame2 = cv2.cvtColor(frame20,cv2.COLOR_BGR2GRAY)

        next = frame2

        SC_values_ref[ii-1] = SimilarityComparison(frame_ref, frame2)

        ###prvs_s = gaussian(prvs, sigma = 1)
        ###next_s = gaussian(next, sigma = 1)

        prvs_s = prvs
        next_s = next
        ###flow_d = next_s-prvs_s
        flow = cv2.calcOpticalFlowFarneback(prvs_s,next_s, None, .5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        hsv[...,2] = mag*10

        magStack[:,:,ii] = mag

        if ii%100==0:
            print(ii)
        prvs = next
        
    SC_diff = (np.gradient(SC_values_ref[:-10]))

    SC_diff_max = np.max(SC_diff)
    SC_diff_min = np.min(SC_diff)

    SC_diff_pos, _ = find_peaks(SC_diff, height= SC_diff_max*0.85,distance=30)
    SC_diff_neg, _ = find_peaks(-SC_diff, height= -SC_diff_min*0.85,distance=30)

    ### added in 10/19 fixed the issue of uncomplete cycle at the beginning
    if len(SC_diff_neg)>1 and len(SC_diff_pos)>0:
        if abs(SC_diff_neg[0]-SC_diff_pos[0])>abs(SC_diff_neg[1]-SC_diff_pos[0]):
            SC_diff_neg=SC_diff_neg[1:]


    SC_inv_height = np.max(1-SC_values_ref[:-10])
    dist_peak, _ = find_peaks(1-SC_values_ref[:-10], height=  SC_inv_height*0.85,distance=100)
    
    
    ###thresh = threshold_otsu(magSum)
    magSum = np.max(magStack,axis=2)
    thresh = np.percentile(magSum,80)
    mask = magSum>1*thresh

    ###cellMask1 = binary_closing(mask,disk(1))
    ###cellMask2 = ndi.binary_fill_holes(cellMask1)
    ###cellMask3 =  remove_small_objects(cellMask2,200)

    ###cellMask4 = binary_closing(cellMask3,disk(2))
    ###cellMask5 = ndi.binary_fill_holes(cellMask4)
    ###cellMask6 = binary_erosion(cellMask5,disk(1))
    ###cellMask7 = binary_opening(cellMask6,disk(3))

    ###cellMask8 =  remove_small_objects(cellMask7,400)
    ###cellMask9 = clear_border(cellMask8)
    ###cellMask9 = getLargestCC(cellMask9)

    ###mask_label = label(cellMask9)
    mask_label = label(mask)

    ###for jj in range(0,1):# just one object
    mask_region = (mask_label>0)
    mask_region_stack = np.repeat(mask_region[:, :, np.newaxis], magStack.shape[2], axis=2)
    mask_region_size = np.sum(mask_region)
    print(mask_region_size)
    ##plt.imshow(mask_region_stack[:,:,200])
    magStack_mask = np.multiply(magStack, mask_region_stack.astype(int))
    flow_trace = np.sum(magStack_mask,axis=0)
    flow_trace = np.sum(flow_trace, axis=0)/mask_region_size

    A_list = []
    B_list = []

    half_width1 = 7 # related to sample freqency
    leftBound1 = SC_diff_pos-half_width1
    ###leftBound1 = dist_peak-half_width1
    rightBound1 = SC_diff_pos+half_width1
    ###rightBound1 = dist_peak

    half_width2 = 11 # related to sample freqency
    leftBound2 = SC_diff_neg-half_width2
    rightBound2 = SC_diff_neg+half_width2
    ###leftBound2 = dist_peak+20
    ###rightBound2 = dist_peak+half_width2


    for ii in range(len(leftBound1)):
        if leftBound1[ii]<5 or rightBound1[ii]>len(flow_trace)-20:
            continue

        maxV = np.max(flow_trace[leftBound1[ii]:rightBound1[ii]])
        neg_ind = leftBound1[ii]+np.argmax(flow_trace[leftBound1[ii]:rightBound1[ii]])
        A_list.append(neg_ind)

    for ii in range(len(leftBound2)):
        if leftBound2[ii]<5 or rightBound2[ii]>len(flow_trace)-20:
            continue

        maxV = np.max(flow_trace[leftBound2[ii]:rightBound2[ii]])
        pos_ind = leftBound2[ii]+np.argmax(flow_trace[leftBound2[ii]:rightBound2[ii]])
        B_list.append(pos_ind)

    As = np.array(A_list)
    Bs = np.array(B_list)
    print(As)
    print(Bs)

    reg_periods = min(len(As),len(Bs),len(SC_diff_pos),len(SC_diff_neg))
    for mm in range(reg_periods):
        ## writer.writerow( ('subFolder','Optical_A_index', 'Optical_B_index', 'OpticalFlow_A_value','OpticalFlow_B_value','Correlation_A_index','Correlation_B_index','Correlation_A_value','Correlation_B_value') )
        writer.writerow((videoFileName,  As[mm],Bs[mm],flow_trace[As[mm]],flow_trace[Bs[mm]],SC_diff_pos[mm], SC_diff_neg[mm],SC_diff[SC_diff_pos[mm]],SC_diff[SC_diff_neg[mm]],np.percentile(flow_trace,5)))

    fout.flush()
    fout.close()


    fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1)
    plt.rcParams['figure.figsize'] = [28, 32]
            ###plt.subplot(211)

    alpha = 0.8
    img_hsv = color.rgb2hsv(frame1)
    color_mask = np.zeros(frame1.shape)
    color_mask[...,0] = mask_region.astype(float)*255
    color_mask[...,1] = mask_region.astype(float)*255
    #color_mask[...,2] = magMask.astype(np.float)*255
    color_mask_hsv = color.rgb2hsv(color_mask)

    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha
    img_masked = color.hsv2rgb(img_hsv)


    ax1.imshow(img_masked)
    ax2.plot(flow_trace[1:-10],linewidth=2)
    if len(As)>0:
        ax2.plot(As,flow_trace[As], "x",markersize=8)
    if len(Bs)>0:
        ax2.plot(Bs,flow_trace[Bs], "d", markersize=8)
    ax2.plot(dist_peak,flow_trace[dist_peak],'*', markersize=8)
    ax2.plot(SC_diff_pos,flow_trace[SC_diff_pos],'*', markersize=8)
    ax2.plot(SC_diff_neg,flow_trace[SC_diff_neg],'*', markersize=8)


    ax3.plot(1-SC_values_ref[:-10],linewidth=2)
    ax3.plot(dist_peak,1-SC_values_ref[dist_peak],"*", markersize=8)

    ax4.plot(abs(SC_diff[1:-10]),linewidth=2)
    ax4.plot(SC_diff_pos,abs(SC_diff[SC_diff_pos]),"o", markersize=8)
    ax4.plot(SC_diff_neg,abs(SC_diff[SC_diff_neg]),"x", markersize=8)
    ax4.plot(dist_peak,abs(SC_diff[dist_peak]),"*")
    
    displayFigureName2 = OutputPath+"\\"+videoFileName+"_result.png"
    print(displayFigureName2)
    fig.savefig(displayFigureName2)
    fig.clf()
    plt.close(fig)

    
    
    
if __name__ == "__main__":

    ds = 2

    RootPath = 'Y:\\RDRU_MYBPC3_2021\\Pilot20211011\\IPSC_Plate1'

    OutputPath = 'Y:\\RDRU_MYBPC3_2021\\Pilot20211011_plate1_output_update'

    subfolders = list(listdir_nohidden(RootPath))
 
    cpu_num = 4
 

    subFolders = sorted(list(listdir_nohidden(RootPath)))
    ###for mm in range(1,5):
    ###    subfolder = subFolders[mm]
    ###    iPSC_pipeline(RootPath,OutputPath,subfolder,ds)
    Parallel(n_jobs=cpu_num,prefer='threads')(delayed(iPSC_pipeline)(RootPath,OutputPath,subfolder,ds) for subfolder in subFolders)   



