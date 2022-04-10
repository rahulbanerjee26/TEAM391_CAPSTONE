import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from numpy import genfromtxt

def normalize(x):
  return (x - np.mean(x))/np.std(x)



noisy_signal = genfromtxt('CPG_Data/saminul_extended2_secondstimestamp.csv', delimiter=',',encoding="utf8")
signal_time_stamp = noisy_signal[:,0]
noisy_signal = noisy_signal[:,1]

fc_low = 0.1 # cutoff frequency
fc_high = 18 # cutoff frequency
fs = 70 # sampling frequency
wn = [fc_low/(0.5*fs), fc_high/(0.5*fs)]
sos = signal.butter(4, wn, btype='bandpass', analog=False, output='sos')
filtered_signal = signal.sosfilt(sos, noisy_signal)

normalized_signal = np.apply_along_axis(normalize, 0, filtered_signal)

fig = plt.figure()
ax = plt.axes()
ax.plot(signal_time_stamp[2000:3000], normalized_signal[2000:3000],label = 'clean');

fig1 = plt.figure()
ax2 = plt.axes()
ax2.plot(signal_time_stamp[2000:3000], noisy_signal[2000:3000],label = 'noisy')

plt.show()


