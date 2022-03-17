import numpy as np
import glob
import os
import matplotlib.pyplot as plt
###%matplotlib inline
from scipy.signal import find_peaks
from scipy.signal import peak_widths

import pandas as pd
from scipy.signal import butter,filtfilt
###%matplotlib inline

import matplotlib
matplotlib.use('tkAgg')



def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = fs*0.5
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# https://www.py4u.net/discuss/14342
def loops_fill(arr):
    out = arr.copy()
    for row_idx in range(1,out.shape[0]):
            if np.isnan(out[row_idx]):
                out[row_idx] = out[row_idx-1]
    return out

def landmark_detection(flow_trace,corr_trace,cord_trace,beat_count):
    starts, contracts, relaxs, ends = [], [], [], []
  
    
    balances,balances_right = balance_point_detection(corr_trace,beat_count)
    
    dist_inter = round(len(corr_trace)/(len(balances)+1))
    
    leftWid = round(dist_inter*0.8)
    rightWid = round(dist_inter*0.8)
    
    contracts = local_extreme_detection(flow_trace,corr_trace,balances,leftWid,0)
    relaxs = local_extreme_detection(flow_trace,corr_trace,balances,0,rightWid)
    
    starts = starts_detection(flow_trace,contracts)
    
    ends= ends_detection(flow_trace,relaxs)
    
    return starts, contracts, balances,relaxs,ends

def balance_point_detection(corr_trace,beat_count):
    corr_trace_inv = 1-corr_trace
    corr_trace_inv = (corr_trace_inv-np.min(corr_trace_inv))/(np.max(corr_trace_inv)-np.min(corr_trace_inv))
    dist_inter = round(len(corr_trace)/(beat_count+1))
    
    balances_inv, prop_inv = find_peaks(corr_trace_inv, height= 1.0*0.8,distance=dist_inter)
    ###if balances.size>0:
    results_inv = peak_widths(corr_trace_inv,balances_inv)
    peak_wid_inv = (np.median(results_inv[0]))
    
    corr_trace_norm = (corr_trace-np.min(corr_trace))/(np.max(corr_trace)-np.min(corr_trace))
    balances_norm, prop_norm = find_peaks(corr_trace_norm, height= 1.0*0.8,distance=dist_inter) 
    results_norm = peak_widths(corr_trace_norm,balances_norm)
    peak_wid_norm = (np.median(results_norm[0]))    
    
    balances = balances_inv
    balances_right= results_inv[3].astype(int)
    if peak_wid_inv>peak_wid_norm and peak_wid_inv>0.6*dist_inter: # the peak width is too large, then do not need to reverse the correlation curve
        balances = balances_norm
        balances_right=results_norm[3].astype(int)

    ###print("peak width inv: " + str(peak_wid_inv))
    ###print("peak width norm: " + str(peak_wid_norm))
    ###print("dist: " + str(dist_inter))
    ###print("blances: " + str(balances))
    ###print("right: " + str(balances_right))
    return balances,balances_right
    

def local_extreme_detection(flow_trace,corr_trace, balances, leftWid, rightWid):
    res = []
    dist_inter = round(len(corr_trace)/(len(balances)+1))
    for bp in balances:
        leftP = max(0,bp-leftWid)
        rightP = min(len(flow_trace)-1,bp+rightWid)
        extremPs, _ = find_peaks(flow_trace[leftP:rightP],distance=dist_inter)
       
        if extremPs.size>0:
            res.append(extremPs[0]+leftP+1)
            ###res.extend(extremPs+leftP+1)
        else:
            res.append(None)
    return np.array(res)
    
def starts_detection(flow_trace,contracts):
    res = []
    count = 0
    for con in contracts:
        count = count + 1
        if not con:
            res.append(None)
            continue
        else:
            kk = 2
            lp_o = con-kk
            lp_l = lp_o-1
            lp_r = lp_o+1
            while lp_l>0: 
                lp_o = lp_o-1
                lp_l = lp_o-1
                lp_r = lp_o+1
                
                if flow_trace[lp_r]>flow_trace[lp_o] and flow_trace[lp_o]<flow_trace[lp_l]:
                    res.append(lp_o)
                    break
            if len(res)<count:
                res.append(None)
    return res

def ends_detection(flow_trace,relaxs):
    res = []
    count = 0
    for rel in relaxs:
        count = count+1
        if not rel:
            res.append(None)
            continue
        else:
            kk = 2
            rp_o = rel+kk
            rp_l = rp_o-1
            rp_r = rp_o+1
            while rp_r<len(flow_trace)-2: 
                rp_o = rp_o+1
                rp_l = rp_o-1
                rp_r = rp_o+1
                
                
                right_dist = abs(flow_trace[rp_r]-flow_trace[rp_o])
                
                if flow_trace[rp_l]>flow_trace[rp_o] and flow_trace[rp_o]<flow_trace[rp_r]:
                    res.append(rp_o)
                    break
            if len(res)<count:
                res.append(None)
                
    return res

def display_freq_componet(flow_trace,corr_trace,cord_trace,freq):
    
    LEN  = len(flow_trace)
    
    flow_fft = np.fft.fft(flow_trace)/LEN           
    flow_fft = flow_fft[range(int(LEN/2))] # Exclude sampling frequency

    corr_fft = np.fft.fft(corr_trace)/LEN           
    corr_fft = corr_fft[range(int(LEN/2))] # Exclude sampling frequency

    cord_fft = np.fft.fft(cord_trace)/LEN  
    cord_fft = cord_fft[range(int(LEN/2))] # Exclude sampling frequency
        
    
    fig, axs = plt.subplots(2,3,figsize=(15,8))
    axs[0,0].plot(flow_trace)
    axs[0,0].set_title('optical trace')

    axs[0,1].plot(corr_trace)
    axs[0,1].set_title('corr trace')

    axs[0,2].plot(abs(cord_trace))
    axs[0,2].set_title('corr diff trace')

    axs[1,0].plot(freq[1:200],abs(flow_fft[1:200]))
    axs[1,0].set_title('optical trace FFT')

    axs[1,1].plot(freq[1:200],abs(corr_fft[1:200]))
    axs[1,1].set_title('correlation trace FFT')

    axs[1,2].plot(freq[1:200],abs(cord_fft[1:200]))
    axs[1,2].set_title('correlation difftrace FFT')
    
def landmark_visualization(flow_trace0,flow_trace,corr_trace,cord_trace,starts,contracts,balances,relaxs,ends):
    
    cordd_trace = np.diff(cord_trace, prepend=cord_trace[0])
    
    fig, axs = plt.subplots(4,figsize=(18,8))
    
    axs[0].plot(flow_trace0)
    axs[0].plot(starts,flow_trace0[starts],">", markersize=8)
    axs[0].plot(contracts,flow_trace0[contracts],"o", markersize=8)
    axs[0].plot(balances,flow_trace0[balances],"d", markersize=8)
    axs[0].plot(relaxs,flow_trace0[relaxs],"x", markersize=8)
    axs[0].plot(ends,flow_trace0[ends],"<", markersize=8)
    axs[0].set_title('optical trace')
    
    axs[1].plot(flow_trace)
    axs[1].plot(starts,flow_trace[starts],">", markersize=8)
    axs[1].plot(contracts,flow_trace[contracts],"o", markersize=8)
    axs[1].plot(balances,flow_trace[balances],"d", markersize=8)
    axs[1].plot(relaxs,flow_trace[relaxs],"x", markersize=8)
    axs[1].plot(ends,flow_trace[ends],"<", markersize=8)
    axs[1].set_title('smoothed optical trace')

    axs[2].plot(1-corr_trace)
    axs[2].plot(starts,1-corr_trace[starts],">", markersize=8)
    axs[2].plot(contracts,1-corr_trace[contracts],"o", markersize=8)
    axs[2].plot(balances,1-corr_trace[balances],"d", markersize=8)
    axs[2].plot(relaxs,1-corr_trace[relaxs],"x", markersize=8)
    axs[2].plot(ends,1-corr_trace[ends],"<", markersize=8)
    axs[2].set_title('inverse of corr trace')

    axs[3].plot(abs(cord_trace))
    axs[3].plot(starts,abs(cord_trace[starts]),">", markersize=8)
    axs[3].plot(contracts,abs(cord_trace[contracts]),"o", markersize=8)
    axs[3].plot(balances,abs(cord_trace[balances]),"d", markersize=8)
    axs[3].plot(relaxs,abs(cord_trace[relaxs]),"x", markersize=8)
    axs[3].plot(ends,abs(cord_trace[ends]),"<", markersize=8)
    axs[3].set_title('corr diff trace')
    
    return fig
    
def pair_markers(starts, contracts, balances, relaxs, ends):
        
    starts1, contracts1, balances1,relaxs1,ends1 = [],[],[],[],[]
        
    for ii in range(len(starts)):
        if starts[ii] and contracts[ii] and balances[ii] and relaxs[ii] and ends[ii]:
            starts1.append(starts[ii])
            contracts1.append(contracts[ii])
            balances1.append(balances[ii])
            relaxs1.append(relaxs[ii])
            ends1.append(ends[ii])

    ###### using list comprehension to remove None values in list        
    ###starts = [i for i in starts if i]
    ###contracts = [i for i in contracts if i]
    ###balances = [i for i in balances if i]
    ###relaxs = [i for i in relaxs if i]
    ###ends = [i for i in ends if i]

    return starts1, contracts1, balances1, relaxs1, ends1


if __name__ == "__main__":
    
    rootFolder = r'../MYBPC3_dataset/20220217_iCell_smMol_DataSetAnalysis/Plate_Well_manual_correction/*.npz'
    fileNames = sorted(glob.glob(rootFolder))
    fileNum = len(fileNames)

    for file in fileNames:

        ### load data and preprocessing (truncation and fill up NaN values)
        (dirName,filename) = os.path.split(file)

        data = np.load(file)
        flow_trace0 = data['flow_trace']
        flow_trace1 = data['flow_trace_smooth']
        corr_trace1 = data['corr_trace']
        cord_trace1 = data['cord_trace']
        starts=data['starts']
        contracts=data['contracts']
        balances = data['balances']
        relaxs = data['relaxs']
        ends = data['ends']

        print('dirName: ' + dirName)
        print('filename: ' + filename)
        fig = landmark_visualization(flow_trace0,flow_trace1,corr_trace1,cord_trace1,starts,contracts,balances,relaxs,ends)      

        print("annotation 'starts'")
        aa = plt.ginput(-1,timeout=-1)
        starts_m=[round(a[0]) for a in aa]
        print("finished 'starts'")

        print("annotation 'contracts'")
        bb = plt.ginput(-1,timeout=-1)
        contracts_m = [round(b[0]) for b in bb]
        print("finished 'contracts'")

        print("annotation 'balances'")
        cc = plt.ginput(-1,timeout=-1)
        balances_m = [round(c[0]) for c in cc]
        print("finished 'balances'")

        print("annotation 'relaxs'")
        dd = plt.ginput(-1,timeout=-1)
        relaxs_m = [round(d[0]) for d in dd]
        print("finished 'relaxs'")

        print("annotation 'ends'")
        ee = plt.ginput(-1,timeout=-1)
        ends_m = [round(e[0]) for e in ee]
        print("finished 'ends'")

        os.system('clear')
        plt.show()
        plt.close('all')

        annotationManualFileName = dirName + '/' + filename
        np.savez(annotationManualFileName, flow_trace=flow_trace0, flow_trace_smooth=flow_trace1, corr_trace=corr_trace1,
                 cord_trace=cord_trace1,
                 starts=starts_m, contracts=contracts_m, balances=balances_m, relaxs=relaxs_m, ends=ends_m,
                 starts_old=starts, contracts_old=contracts, balances_old=balances, relaxs_old=relaxs, ends_old=ends,
                 manual=1)


        filename1 = filename.split('_feature')[0]
        baseline = [np.percentile(flow_trace1,5)]*len(ends_m)
        zipped = list(zip([filename1]*len(ends_m),starts_m, contracts_m, relaxs_m, ends_m, flow_trace1[contracts_m],flow_trace1[relaxs_m],baseline))
        df = pd.DataFrame(zipped, columns=['subFolder','Optical_A_index', 'Optical_B_index','Optical_C_index','Optical_D_index','OpticalFlow_A_value',
                                           'OpticalFlow_B_value', 'OpticalFlow_baseline'])
        
        csvfilename = dirName+'/'+filename1+'_endpoints.csv'
        print(csvfilename)
        df.to_csv(csvfilename, index=False)

        if 1:
            LEN = len(flow_trace1)
            Fs = 100
            ### Filtering out noises

            values = np.arange(int(LEN / 2))
            timePeriod = LEN / Fs
            freq = values / timePeriod

            # Estimate the beating count
            flow_fft1 = np.fft.fft(flow_trace1) / LEN
            flow_fft1 = flow_fft1[range(int(LEN / 2))]  # Exclude sampling frequency
            freq_ind = np.argmax(abs(flow_fft1[1:200]))
            beat_count = freq[1 + freq_ind] * LEN / Fs


            fig_m = landmark_visualization(flow_trace0, flow_trace1, corr_trace1, cord_trace1,starts_m, contracts_m, balances_m, relaxs_m, ends_m)
            outputFigName = dirName + '/' + filename[:-4] + '_annotation.png'
            fig_m.savefig(outputFigName)
            fig_m.clf()
            plt.close(fig_m)
