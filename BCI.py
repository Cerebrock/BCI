import os; os.chdir(r'./CyKit/Py3') 
import sys
import cyPyWinUSB as hid
import queue
from cyCrypto.Cipher import AES
from cyCrypto import Random
import threading
import pickle

from ctypes import windll, Structure, c_long, byref
import ctypes

from pyriemann.stats import pairwise_distance
from pyriemann.estimation import Covariances
from sklearn.manifold import TSNE

import numpy as np
import pandas as pd
from collections import deque
from time import sleep
from IPython.display import clear_output
from IPython.display import Audio, display

import scipy.signal
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from scipy.fftpack import fft
from scipy.signal import firwin, remez, kaiser_atten, kaiser_beta
from scipy.signal import butter, lfilter

from pyriemann.estimation import XdawnCovariances, Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.viz import plot_confusion_matrix
from pyriemann.classification import MDM, TSclassifier
from pyriemann.estimation import Covariances


from catboost import CatBoostClassifier
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold

class POINT(Structure):
    _fields_ = [("x", c_long), ("y", c_long)]

class WinMouse:
    def __init__(self):
        user32 = ctypes.windll.user32
        self.width = user32.GetSystemMetrics(0)
        self.height = user32.GetSystemMetrics(1)

    def click(self, x, y):
        ctypes.windll.user32.SetCursorPos(x, y)
        ctypes.windll.user32.mouse_event(2, 0, 0, 0, 0) # left down
        ctypes.windll.user32.mouse_event(4, 0, 0, 0, 0) # left up

    def move_mouse(self, x, y):
        ctypes.windll.user32.SetCursorPos(x, y)
    
    @staticmethod
    def get_mouse_position():
        pt = POINT()
        windll.user32.GetCursorPos(byref(pt))
        return pt.x, pt.y
        
class BluetoothIO(threading.Thread):
    def __init__(self, eeg_queue, gyro_queue):
        super().__init__()
        self.hid = None
        self.delimiter = ", "
        self.eeg_queue = eeg_queue
        self.gyro_queue = gyro_queue
        devicesUsed = 0
    
        for device in hid.find_all_hid_devices():
            if device.product_name == 'EEG Signals':
                devicesUsed += 1
                self.hid = device
                self.hid.open()
                self.serial_number = device.serial_number
                device.set_raw_data_handler(self.dataHandler)                   
        if devicesUsed == 0:
            os._exit(0)
        sn = self.serial_number
        
        # EPOC+ in 16-bit Mode.
        k = ['\0'] * 16
        k = [sn[-1],sn[-2],sn[-2],sn[-3],sn[-3],sn[-3],sn[-2],sn[-4],sn[-1],sn[-4],sn[-2],sn[-2],sn[-4],sn[-4],sn[-2],sn[-1]]
        
        # EPOC+ in 14-bit Mode.
        #k = [sn[-1],00,sn[-2],21,sn[-3],00,sn[-4],12,sn[-3],00,sn[-2],68,sn[-1],00,sn[-2],88]
        
        self.key = str(''.join(k))
        self.cipher = AES.new(self.key.encode("utf8"), AES.MODE_ECB)

    def dataHandler(self, data):
        join_data = ''.join(map(chr, data[1:]))
        data = self.cipher.decrypt(bytes(join_data,'latin-1')[0:32])
        
        if str(data[1]) == "16":
            self.eeg_queue.append(data)                
        elif str(data[1]) == "32": 
            self.gyro_queue.append(data)
        
    #self.cyIO.sendData(1, str(data[0]) + packet_data)
    #continue
    
class EEG():
    def __init__(self, eeg_queue, gyro_queue):
        super().__init__()
        self.eeg_queue = eeg_queue
        self.gyro_queue = gyro_queue
        self.delimiter = ', '
        
    def convertEPOC_PLUS(self, value_1, value_2):
        edk_value = "%.8f" % (((int(value_1) * .128205128205129) + 4201.02564096001) + ((int(value_2) -128) * 32.82051289))
        return edk_value
       
    def convertEPOC_PLUS_Gyro(self, value_1, value_2): 
        edk_value = "%.8f" % ((8192.0 + (int(value_1) * 1)) + ((int(value_2) - 128) * 64))   
        return edk_value
    
    def get_gyro(self, n):
        buffer = []
        c = 0 
        while c < n:
            try:
                data = self.gyro_queue.popleft()
            except IndexError:
                sleep(0.001)
                continue
            try:
                packet_data = ""
                for i in range(2,16,2):
                    packet_data = packet_data + str(self.convertEPOC_PLUS_Gyro(str(data[i]), str(data[i+1]))) + self.delimiter

                for i in range(18,len(data),2):
                    packet_data = packet_data + str(self.convertEPOC_PLUS_Gyro(str(data[i]), str(data[i+1]))) + self.delimiter

                packet_data = packet_data[:-len(self.delimiter)]
                buffer.append([float(x) for x in packet_data.split(', ')])                
            except Exception as exception2:
                print(str(exception2))
            c += 1
        return np.array(buffer)

    def get_data(self, n):
        buffer = []
        c = 0
        while c < n:
            try:
                data = self.eeg_queue.popleft()
            except IndexError:
                sleep(0.001)
                continue
            try:
                packet_data = ""
                for i in range(2,16,2):
                    packet_data = packet_data + str(self.convertEPOC_PLUS(str(data[i]), str(data[i+1]))) + self.delimiter
                for i in range(18,len(data),2):
                    packet_data = packet_data + str(self.convertEPOC_PLUS(str(data[i]), str(data[i+1]))) + self.delimiter

                packet_data = packet_data[:-len(self.delimiter)]
                buffer.append([float(x) for x in packet_data.split(', ')])                
            except Exception as exception2:
                print(str(exception2))
            c += 1
        return np.array(buffer)
    
def butter_bandpass(lowcut, highcut, fs, order=5):
    #https://github.com/faturita/python-nerv/blob/master/OfflineFeatureAnalysis3.py
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def plot_welch(t, canal = 0):
    f, pxx = scipy.signal.welch(t[:,canal], fs = 44100)
    plt.plot(f, pxx)
    plt.xlim(0, 10000)
    plt.xlabel('Frecuencias')

def plot_multiple(w, fs = 256):
    '''Plotea primero la onda en dominio de tiempo, luego el espectro (FFT) y luego el espectrograma'''
    
    #Creo el espacio para plotear, una "figura" vacia
    plt.figure(figsize=(15,15))

    #Dividido en 3 filas y 1 columna, ploteo la onda en el 1er espacio
    plt.subplot(3, 1, 1)
    plt.plot(w)
    
    #Pongo titulo al eje y
    plt.ylabel('Amplitude')
    #Grilla de fondo
    plt.grid()
    
    #FFT
    #n = len(w) = duración * framerate
    n = len(w)
    Y = np.fft.rfft(w) / n 
    freqs = np.fft.fftfreq(n, d= 1/44100)
    
    #Plot FFT
    plt.subplot(3, 1, 2)
    #Ploteo las frecuencias positivas y sus valores, con un color RGBA
    plt.plot(freqs[:10000], abs(Y[:10000]), c = [0.9, 0.2, 0.1, 0.8])
    plt.xlabel('Freq (Hz)')
    
    #Marco en el eje X ciertos valores incluyendo la frecuencia de máximo valor
    plt.xticks(np.sort(np.append(freqs[:10000:1000], freqs[np.argmax(abs(Y[:10000]))])), rotation = 60)
    plt.ylabel('|Y(freq)|')

    #Espectrograma
    plt.subplot(3, 1, 3)
    Pxx, freqs, bins, im = plt.specgram(w, Fs=fs, noverlap=10, cmap= 'coolwarm')
    plt.ylim(0, 8000)
    plt.ylabel('Freq')
    plt.xlabel('Time')
    plt.show()
    
def do_fft(data, fs = 256):
    # Get real amplitudes of FFT (only in postive frequencies)
    fft_vals = np.absolute(np.fft.rfft(data))

    # Get frequencies for amplitudes in Hz
    fft_freq = np.fft.rfftfreq(len(data), 1.0/fs)

    eeg_bands = {'Delta': (0, 4),
                 'Theta': (4, 8),
                 'Alpha': (8, 12),
                 'Beta': (12, 30),
                 'Gamma': (30, 45)}

    # Take the mean of the fft amplitude for each EEG band
    eeg_band_fft = dict()
    for band in eeg_bands:  
        freq_ix = np.where((fft_freq >= eeg_bands[band][0]) & 
                           (fft_freq <= eeg_bands[band][1]))[0]
        eeg_band_fft[band] = np.mean(fft_vals[freq_ix])
    return eeg_band_fft
        
def plot_bands(eeg_bands):
    df = pd.DataFrame(columns=['band', 'val'])
    df['band'] = eeg_bands.keys()
    df['val'] = [eeg_bands[band] for band in eeg_bands]
    ax = df.plot.bar(x='band', y='val', legend=False)
    ax.set_xlabel("EEG band")
    ax.set_ylabel("Mean band Amplitude") 