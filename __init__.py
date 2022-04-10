from bitalino import BITalino
import time
from pandas import read_csv
from matplotlib import pyplot
import pandas as pd


def collect_data(macAddress,run_time=33,channels=[3],samplingRate=100,nSamples=1,saveData=True,outputFileName='ppgData'):
    # Connect to BITalino
    device = BITalino(macAddress)
    ppg_data = []
    times = []
    # Start Acquisition
    device.start(samplingRate, channels)
    start = time.time()
    end = time.time()
    while (end - start) < run_time:
        '''
        The return value is a matrix like below
        [
            [sequence_num,digital1_val,digital2_val,digital3_val,digital4_val,..analog_channels],

        ]
        '''
        times.append(end-start)
        ppg_data.extend(device.read(nSamples))
        end = time.time()
    # Stop acquisition
    device.stop()
    # Close connection
    device.close()

    channel4_data = []
    for data in ppg_data:
        channel4_data.append(data[-1])
    ppg_vs_time = zip(times,channel4_data)

    if saveData:
            with open(f'PPG_Data/{outputFileName}.csv','w') as file:
                count = 0
                for ppg_time,ppg_value in ppg_vs_time:
                 count+=1
                 file.write(f'{ppg_time},{ppg_value}\n')
                 if count == 3000:
                  break
            print('Collected Data') 
    return ppg_vs_time

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

if __name__ == '__main__':
    # Windows : "XX:XX:XX:XX:XX:XX"
    # Mac OS :  "/dev/tty.BITalino-XX-XX-DevB" or "/dev/tty.BITalino-DevB" 
    macAddress = "/dev/tty.BITalino-DevB"
    name = '3_SAM_Test'
    outputFileName = f'{name}_PPG_Data'
    ppg_vs_time = collect_data(macAddress,outputFileName=outputFileName)
    plot_time_series_ppg(path=f'PPG_Data/{outputFileName}.csv',outputFileName = outputFileName)

