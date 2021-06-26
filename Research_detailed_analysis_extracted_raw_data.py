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


if __name__ == "__main__":

    fileName = r'Z:\pangj05\TROPONIN2021\20210616DataSetAnalysis\Pairwise\T0_BeforeDosing_DMSO_s02_video_ds_cell_0.npz'
    npzfile=np.load(fileName)
    print(sorted(npzfile.files))
    print(npzfile['similarity'])
    print(len(npzfile['similarity']))
