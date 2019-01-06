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

import STFT
import OverlapAdd

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
import utils as U

class Param_ISTFT(STFT.Param_STFT):
    def __init__(self, frameSize=4096, hopSize=2048, fftshift=True, windowType='hann', zeroPadding=0):
        STFT.Param_STFT.__init__(self, frameSize=frameSize, hopSize=hopSize, fftshift=fftshift, windowType=windowType, zeroPadding=zeroPadding)

            
class ISTFT(U.NewModule):        
    #STFT module
    def __init__(self, param_ISTFT=None):
        self.OLA = OverlapAdd.OverlapAdd()
        self.OLA_ow = OverlapAdd.OverlapAdd()

        self.setParam(param_ISTFT)                
        self.update()

        
        self.clear()
        
    def getParam(self):
        return Param_ISTFT(frameSize = self._frameSize_, hopSize = self._hopSize_, fftshift = self._fftshift_, windowType = self._windowType_, zeroPadding = self._zeroPadding_)
        
    def getFrameSize(self):
        return self._frameSize_
        
    def getHopSize(self):
        return self._hopSize_
       
    def getFftshift(self):
        return self._fftshift_
        
    def getWindowType(self):
        return self._windowType_
        
    def getZeroPadding(self):
        return self._zeroPadding_        

    def setParam(self, param_ISTFT=None): 
        if param_ISTFT is None:
            param_ISTFT = Param_ISTFT()
        self.param_ISTFT = param_ISTFT
        self._needsUpdate_ = True
        self.OLA.setParam(OverlapAdd.Param_OverlapAdd(hopSize=self.param_ISTFT.hopSize, frameSize=self.param_ISTFT.frameSize))
        self.OLA_ow.setParam(OverlapAdd.Param_OverlapAdd(hopSize=self.param_ISTFT.hopSize, frameSize=self.param_ISTFT.frameSize))

    def getParam(self):
        return Param_ISTFT(frameSize = self._frameSize_, hopSize = self._hopSize_, fftshift = self._fftshift_, windowType = self._windowType_, zeroPadding = self._zeroPadding_)
        
    def setFrameSize(self,frameSize):
        self.param_ISTFT.frameSize = frameSize
        self._needsUpdate_ = True
        self.OLA.setFrameSize(frameSize)
        self.OLA_ow.setFrameSize(frameSize)
        
    def setHopSize(self, hopSize):
        self.param_ISTFT.hopSize = hopSize
        self._needsUpdate_ = True
        self.OLA.setHopSize(hopSize)
        self.OLA_ow.setHopSize(hopSize)
       
    def setFftshift(self, fftshift):
        self.param_ISTFT.fftshift = fftshift
        self._needsUpdate_ = True
        
    def setWindowType(self, windowType):
        self.param_ISTFT.windowType = windowType
        self._needsUpdate_ = True
        
    def setZeroPadding(self, zeroPadding):
        self.param_ISTFT.zeroPadding = zeroPadding
        self._needsUpdate_ = True
        
    def update(self):
        self._frameSize_ = self.param_ISTFT.frameSize
        self._hopSize_ = self.param_ISTFT.hopSize
        self._fftshift_ = self.param_ISTFT.fftshift
        self._windowType_ = self.param_ISTFT.windowType
        self._zeroPadding_ = self.param_ISTFT.zeroPadding
        
        #Creating window
        self._w_ = U.windowing(np.ones(self._frameSize_), self._frameSize_, typeWindow=self._windowType_)
        #Computing squared windows for LSEE
        self._window_ = np.power(self._w_, 2)
        
        self._needsUpdate_ = False

    def process(self, X):
        if self._needsUpdate_ == True:
            self.update()
        
        y = np.array(())
        for X_frame in X:
            y = np.append(y, self.clockProcess(X_frame))

        return y
       
    def clockProcess(self, X_frame):    
        x = U.zeroPhasing(np.real(ifft(X_frame)), self._frameSize_, zeroPadding=self._zeroPadding_, fftshift=self._fftshift_, inverse=True) * self._w_
        y = self.OLA.clockProcess(x) 
        ow = self.OLA_ow.clockProcess(self._window_)
        
        return y/(ow + np.finfo(float).eps) #LSEE (avoiding division by zero)
    
    def clear(self):
        self.OLA.clear()
        self.OLA_ow.clear()

def callback(nameInput='../sounds/sine.wav', nameOutput='processed/sine_STFT.wav', frameSize=3071, zeroPadding=1025, hopSize=256, windowType='hann', fftshift=True):

    #Loading audio
    x, fs = U.wavread(nameInput)
    
    #Computing STFT
    X = STFT.STFT(STFT.Param_STFT(frameSize, hopSize, fftshift, windowType, zeroPadding)).process(x)
    
    #Computing ISTFT
    y = ISTFT(Param_ISTFT(frameSize=frameSize, hopSize=hopSize, fftshift=fftshift, windowType=windowType, zeroPadding=zeroPadding)).process(X)
    
    #Writing audio output
    U.wavwrite(y, fs, nameOutput)

    #Plotting 
    fig = plt.figure(figsize=(15, 15), dpi= 80, facecolor='w', edgecolor='k')
    fig.canvas.set_window_title('Signals and Window')
    
    tmp = fig.add_subplot(2,1,1)
    tmp.plot(U.getTimeAxis(x, fs), x)
    tmp.set_title('Original Signal')
    tmp = fig.add_subplot(2,1,2)
    tmp.plot(U.getTimeAxis(y, fs), y)
    tmp.set_title('Re-Synthesized Signal')
    
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
