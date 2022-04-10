from bitalino import BITalino
import time
from pandas import read_csv
from matplotlib import pyplot
import pandas as pd
from scipy import signal
import numpy as np
import streamlit as st
from collections import deque

def collect_data(macAddress,run_time=90,channels=[3],samplingRate=100,nSamples=1,saveData=False,outputFileName='ppgData',is_streamlit = False):
    # Connect to BITalino
    print("------  Collecting Data")
    device = BITalino(macAddress)
    ppg_data = []
    times = []

    if is_streamlit:
        progress_bar = st.progress(0)
        time_reaiming = st.text(f"Time Reamining: ~{run_time}s")


    # Start Acquisition
    device.start(samplingRate, channels)
    start = time.time()
    end = time.time()
    sec_count = 0
    while (end - start) < run_time + 2:
        '''
        The return value is a matrix like below
        [
            [sequence_num,digital1_val,digital2_val,digital3_val,digital4_val,..analog_channels],

        ]
        '''
            
        times.append(end-start)
        samples = device.read(nSamples)
        ppg_data.extend(samples)
        end = time.time()
        sec_count+=1

            

        if sec_count == 500:
            print("Time Reamining:  ~", int(run_time - (end - start)), "s")
            if is_streamlit:
                progress_bar.progress(min(100,int(100*(end - start)/run_time)))
                time_reaiming.text(f"Time Reamaining:  ~ {int(run_time - (end - start))}s")
            sec_count = 0
    
            

    # Stop acquisition
    device.stop()
    # Close connection
    device.close()

    channel4_data = []
    for data in ppg_data[0:run_time*100]:
        channel4_data.append(data[-1])
    # ppg_vs_time = zip(times,channel4_data)

    if saveData:
            with open(f'PPG_Data/{outputFileName}.csv','w') as file:
                count = 0
                for ppg_time,ppg_value in ppg_vs_time:
                 count+=1
                 file.write(f'{ppg_time},{ppg_value}\n')
                 if count == run_time*100:
                  break
            print('------  Collected Data') 

    if is_streamlit:
        time_reaiming.empty()
        progress_bar.empty()     
    
    return channel4_data[0:run_time*100]

def plot_time_series_ppg(ppg_data=None,path='',outputFileName = 'ppgData'):
    if not (ppg_data or path):
        raise Exception('Please provide data or path to csv file')
    if path: data = read_csv(path,header=0, index_col=0)
    else: data = ppg_data

    data.plot(legend=None)
    pyplot.title(outputFileName)
    pyplot.xlabel('Time')
    pyplot.ylabel('PPG Value')
    pyplot.savefig(f'PPG_Data/{outputFileName}.png')
    pyplot.show()
    

def plot_time_series_cpg(path='',outputFileName='cpgData'):
    if path: data = read_csv(path,header=None)
    else: raise Exception('Please provide path to cpg data')
    data[0] = pd.to_datetime(data[0], errors='coerce')
    start = data[0][0]
    data[0] = data.apply(lambda row: (row[0] - start).total_seconds(), axis = 1)
    print(data.head())
    data.plot(x=0,y=1,legend=None)
    data.to_csv('CPG_Data/saminul_extended2_secondstimestamp.csv',index=False)
    pyplot.title(outputFileName)
    pyplot.xlabel('Time')
    pyplot.ylabel('CPG Value')
    pyplot.savefig(f'CPG_Data/{outputFileName}.png')
    pyplot.show()

def clean_data(data, doPlot = False, filter = True):
    print("------  Cleaning Data")
    fc_low = 0.1 # cutoff frequency
    fc_high = 18 # cutoff frequency
    fs = 100 # sampling frequency
    wn = [fc_low/(0.5*fs), fc_high/(0.5*fs)]
    sos = signal.butter(4, wn, btype='bandpass', analog=False, output='sos') # 4th order band-pass digital butterworth filter with cut-off freqs of 0.1hz and 18hz

    sig = data
    if filter:
        filtered_data = signal.sosfilt(sos, sig)
    else:
        filtered_data = sig

    all = filtered_data
    all_divided = np.split(all, np.arange(1000, len(all), 1000))
    incompletes_removed = [a for a in all_divided if a.shape[0] == 1000]
    divided_data = np.stack(incompletes_removed)

    def normalize(x):
        return (x - np.mean(x))/np.std(x)

    normalized_data = np.apply_along_axis(normalize, 1, divided_data)

    if doPlot:
        pyplot.plot(normalized_data[0])
        pyplot.show()
    return normalized_data

if __name__ == '__main__':
    # Windows : "XX:XX:XX:XX:XX:XX"
    # Mac OS :  "/dev/tty.BITalino-XX-XX-DevB" or "/dev/tty.BITalino-DevB" 
    macAddress = "/dev/tty.BITalino-DevB"
    name = '3_SAM_Test'
    outputFileName = f'{name}_PPG_Data'
    ppg_vs_time = collect_data(macAddress,outputFileName=outputFileName)
    plot_time_series_ppg(path=f'PPG_Data/{outputFileName}.csv',outputFileName = outputFileName)

