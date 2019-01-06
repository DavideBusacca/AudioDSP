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
        
class STFT(U.NewModule):        
    #STFT module
    def __init__(self, param_STFT=None):   
        self.setParam(param_STFT)        
        self.clear()
        self.update()
        
    def getParam(self):
        return Param_STFT(frameSize = self._frameSize_, hopSize = self._hopSize_, fftshift = self._fftshift_, windowType = self._windowType_, zeroPadding = self._zeroPadding_)
        
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

    def setParam(self, param_STFT=None): 
        if param_STFT == None:
            param_STFT = Param_STFT()
        self.param_STFT = param_STFT
        
    def getParam(self):
        return Param_STFT(frameSize = self._frameSize_, hopSize = self._hopSize_, fftshift = self._fftshift_, windowType = self._windowType_, zeroPadding = self._zeroPadding_)
        
    def setFrameSize(self,frameSize):
        self.param_STFT.frameSize = frameSize
        self._needsUpdate_ = True
        
    def setHopSize(self):
        self.param_STFT.hopSize
        self._needsUpdate_ = True
       
    def setFftshift(self):
        self.param_STFT.fftshift
        self._needsUpdate_ = True
        
    def setWindowType(self):
        self.param_STFT.windowType
        self._needsUpdate_ = True
        
    def setZeroPadding(self):
        self.param_STFT.zeroPadding
        self._needsUpdate_ = True
        
    def update(self):
        self._frameSize_ = self.param_STFT.frameSize
        self._hopSize_ = self.param_STFT.hopSize
        self._fftshift_ = self.param_STFT.fftshift
        self._windowType_ = self.param_STFT.windowType
        self._zeroPadding_ = self.param_STFT.zeroPadding
        
        #Creating window
        self._w_ = U.windowing(np.ones(self._frameSize_), self._frameSize_, typeWindow=self._windowType_)
        
        self._needsUpdate_ = False
    
    def process(self, x):
        if self._needsUpdate_ == True:
            update()
            
        nFrames = int(np.floor((x.size-self._frameSize_)/self._hopSize_))
        for f in range(nFrames):
            frame = x[self._hopSize_*f:self._hopSize_*f+self._frameSize_] 
            frame = self.clockProcess(frame)
            if f == 0:
                X = frame
            else:   
                X = np.vstack([X, frame])  
            
        return X
        
    def clockProcess(self, x):    
        x = U.zeroPhasing(x*self._w_, self._frameSize_, self._zeroPadding_, fftshift=self._fftshift_)
        return fft(x, self._frameSize_+self._zeroPadding_)      
    
    def clear(self):
        pass

def callbackClass(nameInput='../sounds/sine.wav', frameSize=3071, zeroPadding=1025, hopSize=2048, fftshift=True, windowType='hann'):   

    #Loading audio
    x, fs = U.wavread(nameInput)

    #Computing STFT  
    X = STFT(Param_STFT(frameSize, hopSize, fftshift, windowType, zeroPadding)).process(x)
    
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
    callbackClass()
