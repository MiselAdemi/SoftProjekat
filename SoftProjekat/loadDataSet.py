from numpy import sin, linspace, pi
from pylab import  plot, show, title, xlabel, ylabel, subplot
from scipy import fft, arange, signal
from scipy.io import wavfile
import heapq
import numpy as np


class LoadDataSet:
    DataSet = [] # Niz koji cini data set za obucavanje
    toneName = ''

    def __init__(self, file):
        Fs = 44100  # sampling rate
        rate, data = wavfile.read(file)
        y = data[:,1]
        lungime = len(y)
        timp = len(y)/rate
        t = linspace(0,timp,len(y))

        subplot(2,1,1)
        plot(t,y)
        xlabel('Time')
        ylabel('Amplitude')
        subplot(2,1,2)
        self.plotSpectrum(y, rate)
        #show()

    def toneFromFreq(self, freq):
        #print freq
        if (freq >= 127 and freq <= 133):
            return 'C3'
        elif (freq >= 144 and freq <= 153):
            return 'D3'
        elif (freq >= 160 and freq <= 170):
            return 'E3'
        elif (freq >= 171 and freq <= 180):
            return 'F3'
        elif (freq >= 190 and freq <= 203):
            return 'G3'
        elif (freq >= 215 and freq <= 227):
            return 'A3'
        elif (freq >= 240 and freq <= 255):
            return 'H3'
        else:
            return 'Nepoznati ton'

    def plotSpectrum(self, y, Fs):
        n = len(y) # duzina signala
        k = arange(n)
        T = n/Fs
        if T == 0:
            T = 1

        frq = k/T
        frq = frq[arange(n/32)]

        Y = fft(y)/n # racunanje fft-a
        Y = Y[range(n/32)]

        arr = abs(Y)

        self.DataSet = arr

        for i in heapq.nlargest(1, xrange(len(arr)), arr.take):
            self.toneName = self.toneFromFreq(frq[i])

        plot(frq, abs(Y), 'r') # plotovanje spectruma
        xlabel('Freq (Hz)')
        ylabel('|Y(freq)|')