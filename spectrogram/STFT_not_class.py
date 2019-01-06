'''
 Copyright (C) 2018  Busacca Davide
 
 This file is part of PV.
 
 PV is free software: you can redistribute it and/or modify it under
 the terms of the GNU Affero General Public License as published by the Free
 Software Foundation (FSF), either version 3 of the License, or (at your
 option) any later version.
 
 PV is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 details.
 
 You should have received a copy of the Affero GNU General Public License
 version 3 along with PV.  If not, see http://www.gnu.org/licenses/
'''

import sys, os
import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
import utils as U

class Param_STFT(U.NewParam):
    #Class that contains the parameters to create a STFT module
    def __init__(self, frameSize=4096, hopSize=2048, fftshift=True, windowType='hann', zeroPadding=0):
        self.frameSize = frameSize
        self.hopSize = hopSize
        self.fftshift = fftshift
        self.windowType = windowType
        self.zeroPadding = zeroPadding

def STFT(x, param_STFT=None ): 
    '''
    Short Time Fourier Transform
    normalized: parameters deleted because useless
    '''      
    
    if param_STFT == None:
        param_STFT = Param_STFT()
        
    frameSize = param_STFT.frameSize
    hopSize = param_STFT.hopSize
    fftshift = param_STFT.fftshift
    windowType = param_STFT.windowType
    zeroPadding = param_STFT.zeroPadding
    
    #Creating window
    w = U.windowing(np.ones(frameSize), frameSize, typeWindow=windowType)
    
    #Padding half frame at beginning and at the end (TSM Matlab Toolbox)
    #x = np.concatenate((np.zeros(int(frameSize/2)), x, np.zeros(int(frameSize/2))))
    #Apparently introduces artefacts in the magnitude spectrogram

    #Computing Spectrogram
    nFrames = int(np.floor((x.size-frameSize)/hopSize))
    for f in range(nFrames):
        frame = x[hopSize*f:hopSize*f+frameSize]
        frame_w = U.zeroPhasing(frame*w, frameSize, zeroPadding, fftshift=fftshift)
        frame_W = fft(frame_w, frameSize+zeroPadding)      
        if f == 0:
            X = frame_W
        else:   
            X = np.vstack([X, frame_W])  
    
    return X
    
def callback(nameInput='../sounds/sine.wav', frameSize=3071, zeroPadding=1025, hopSize=2048, fftshift=True, windowType='hann'):
    
    #Loading audio
    x, fs = U.wavread(nameInput)

    #Computing STFT
    X = stft(x, Param_STFT(frameSize, hopSize, fftshift, windowType, zeroPadding))
    
    #Plotting 
    fig = plt.figure(figsize=(15, 15), dpi= 80, facecolor='w', edgecolor='k')
    fig.canvas.set_window_title('Signals and Window')
    
    tmp = fig.add_subplot(3,1,1)
    tmp.plot(U.getTimeAxis(x, fs), x)
    tmp.set_title('Original Signal')
    
    [tx, fx] = U.getSpectrogramAxis(X, fs, hopSize)
    endX = int(X.shape[1]/2+1)
    fig.canvas.set_window_title('Spectrograms')
    tmp = fig.add_subplot(3,1,2)
    tmp.pcolormesh(tx, fx, np.transpose(U.amp2db(U.getMagnitude(X[:, :endX]))))
    tmp.set_title('Original Magnitude Spectrogram')
    tmp = fig.add_subplot(3,1,3)
    plt.pcolormesh(tx, fx, np.transpose(np.diff(U.getPhase(X[:, :endX]))))
    tmp.set_title('Differential of the Original Phase Spectrogram')

    plt.show()   
    
if __name__ == "__main__":
    callback()
