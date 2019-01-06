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

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import ifft

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
import STFT_not_class as STFT
import utils as U

class Param_ISTFT(STFT.Param_STFT):
    def __init__(self, frameSize=4096, hopSize=2048, fftshift=True, windowType='hann', zeroPadding=0):
        STFT.Param_STFT.__init__(self, frameSize=frameSize, hopSize=hopSize, fftshift=fftshift, windowType=windowType, zeroPadding=zeroPadding)

    
def ISTFT(X, param_ISTFT=None):
    '''
    Inverse Short Time Fourier Transform
    ISTFT using Least Squared Error Estimation for Modified Spectrograms
    Add description
    '''       

    if param_ISTFT == None:
        param_ISTFT = Param_ISTFT()
    
    frameSize = param_ISTFT.frameSize
    hopSize = param_ISTFT.hopSize
    fftshift = param_ISTFT.fftshift
    windowType = param_ISTFT.windowType
    zeroPadding = param_ISTFT.zeroPadding
    
    #Creating window
    w = U.windowing(np.ones(frameSize), frameSize, typeWindow=windowType)    
    #Computing squared windows for LSEE
    window = np.power(w, 2)
    
    #Computing IFFT, appending frames and windows
    i=0
    for frame_X in X:
        x_t = np.real(ifft(frame_X))      
        x_t = U.zeroPhasing(x_t, frameSize, zeroPadding=zeroPadding, fftshift=fftshift, inverse=True)
        x_w = x_t*w
        if i==0:
            i = 1
            x_f = x_w
            ow_f = window
        else:
            x_f = np.append(x_f, x_w)
            ow_f = np.append(ow_f, window)
        
        #The TSM Toolbox here has a "restore energy" part.  
    
    #Overlapping and adding frames and windows  
    x = U.myOverlapAdd(x_f, frameSize, hopSize)
    ow = U.myOverlapAdd(ow_f, frameSize, hopSize)
    ow[np.where(ow<0.001)] = 1 #avoid division by 0

    #Least Squares Error Estimation
    x = x/(ow + np.finfo(float).eps) #LSEE (avoiding division by zero)
    
    return x, ow
    
def callback(nameInput='../sounds/sine.wav', nameOutput='processed/sine_STFT.wav', frameSize=3071, zeroPadding=1025, hopSize=256, windowType='hann', fftshift=True):

    #Loading audio
    x, fs = U.wavread(nameInput)

    #Computing STFT
    X = STFT.STFT(x, STFT.Param_STFT(frameSize=frameSize, hopSize=hopSize, fftshift=fftshift, windowType=windowType, zeroPadding=zeroPadding))

    #Computing ISTFT
    [y, ow] = ISTFT(X, Param_ISTFT(frameSize=frameSize, hopSize=hopSize, fftshift=fftshift, windowType=windowType, zeroPadding=zeroPadding))
    
    #Writing audio output
    U.wavwrite(y, fs, nameOutput)

    #Plotting 
    fig = plt.figure(figsize=(15, 15), dpi= 80, facecolor='w', edgecolor='k')
    fig.canvas.set_window_title('Signals and Window')
    
    tmp = fig.add_subplot(3,1,1)
    tmp.plot(U.getTimeAxis(x, fs), x)
    tmp.set_title('Original Signal')
    tmp = fig.add_subplot(3,1,2)
    tmp.plot(U.getTimeAxis(y, fs), y)
    tmp.set_title('Re-Synthesized Signal')
    tmp = fig.add_subplot(3,1,3)
    tmp.plot(U.getTimeAxis(ow, fs), ow)  
    tmp.set_title('Sum of (Squared) Windows')
    
    [tx, fx] = U.getSpectrogramAxis(X, fs, hopSize)
    endX = int(X.shape[1]/2+1)
    fig2 = plt.figure(figsize=(15, 15), dpi= 80, facecolor='w', edgecolor='k')
    fig2.canvas.set_window_title('Spectrograms')
    tmp = fig2.add_subplot(2,1,1)
    tmp.pcolormesh(tx, fx, np.transpose(U.amp2db(U.getMagnitude(X[:, :endX]))))
    tmp.set_title('Original Magnitude Spectrogram')
    tmp = fig2.add_subplot(2,1,2)
    plt.pcolormesh(tx, fx, np.transpose(np.diff(U.getPhase(X[:, :endX]))))
    tmp.set_title('Differential of the Original Phase Spectrogram')

    plt.show()

    # Evaluating the difference between input and re-synthesized signals
    print("The sum of the differences between the original signal and the resynthsized using the STFT is: " + str(U.distance2signals(x, y, frameSize)) )
    
if __name__ == "__main__":
    callback()
