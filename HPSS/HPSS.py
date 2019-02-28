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
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../spectrogram/'))
import STFT
import ISTFT

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
import utils as U

class Param_HPSS(U.NewParam):
    #Class that contains the parameters to create a STFT module
    def __init__(self,  param_STFT=None, win_harm=31, win_perc=31, masking='hard'):
        self.param_STFT = param_STFT
        self.win_harm = win_harm
        self.win_perc = win_perc
        self.masking = masking

class HPSS(U.NewModule):
    '''
    Median-filtering harmonic percussive source separation (HPSS)
    '''

    def __init__(self, param_HPSS=None):
        self._STFT_ = STFT.STFT()
        self._ISTFT_ = ISTFT.ISTFT()

        self.setParam(param_HPSS)
        self.update()
        self.clear()

    def getParam(self):
        return Param_HPSS(self._param_STFT_, win_harm=self._win_harm_, win_perc=self._win_perc_, masking=self._masking_)

    def getParamSTFT(self):
        return self._param_STFT_

    def getWinHarm(self):
        return self._win_harm_

    def getWinPerc(self):
        return self._win_perc_

    def getMasking(self):
        return self._masking_

    def setParam(self, param_HPSS=None):
        if param_HPSS == None:
            param_HPSS = Param_HPSS()
        self.param_HPSS = param_HPSS

    def setFrameSize(self, frameSize):
        self.param_HPSS.param_STFT.frameSize = frameSize
        self._needsUpdate_ = True

    def setHopSize(self, hopSize):
        self.param_HPSS.param_STFT.hopSize = hopSize
        self._needsUpdate_ = True

    def setFftshift(self, fftshift):
        self.param_HPSS.param_STFT.fftshift = fftshift
        self._needsUpdate_ = True

    def setWindowType(self, windowType):
        self.param_HPSS.param_STFT.windowType = windowType
        self._needsUpdate_ = True

    def setZeroPadding(self, zeroPadding):
        self.param_HPSS.param_STFT.zeroPadding = zeroPadding
        self._needsUpdate_ = True

    def setWin_harm(self, win_harm):
        self.param_HPSS.win_harm = win_harm
        self._needsUpdate_ = True

    def setWin_perc(self, win_perc):
        self.param_HPSS.win_perc = win_perc
        self._needsUpdate_ = True

    def setMasking(self, masking):
        self.param_HPSS.masking = masking
        self._needsUpdate_ = True

    def update(self):
        self._param_STFT_ = self.param_HPSS.param_STFT
        self._win_harm_ = self.param_HPSS.win_harm
        self._win_perc_ = self.param_HPSS.win_perc
        self._masking_ = self.param_HPSS.masking

        self._STFT_.setParam(self._param_STFT_)
        self._ISTFT_.setParam(self._param_STFT_)
        self._STFT_.update()
        self._ISTFT_.update()

        self._needsUpdate_ = False

    def hardMask(self, harm, perc):
        '''
        Hard masking for Harmonic/Percussive Source Separation
        '''
        mask_harm = harm > perc
        mask_perc = perc > harm

        return mask_harm, mask_perc

    # softMask

    def computing_hpss_masks(self, mX):
        """
        Masks used by median-filtering harmonic percussive source separation (HPSS).

        Parameters
        ----------
        mX: input magnitude spectrogram
        win_harm: number of bins used for median filtering in frequency direction
        win_perc: number of bins used for median filtering in time direction

        Returns
        -------
        harmonic component
        percussive component

        """

        #Computing harmonic enhanced spectrogram
        harm = np.empty_like(mX)
        harm[:] = median_filter(mX, size=(self._win_perc_, 1), mode='reflect')
        #Computing percussive enhanced spectrogram
        perc = np.empty_like(mX)
        perc[:] = median_filter(mX, size=(1, self._win_harm_), mode='reflect')

        #Computing harmonic and percussive masks
        if self._masking_ == 'hard':
            mask_harm, mask_perc = self.hardMask(harm, perc)
        # Soft masking can be implemented here

        return mask_harm, mask_perc

    '''

    def clockProcess(self, x):
        #need to create a STFT object inside
        #need to create a memory where to store the value that will be used to compute the median
        #maybe it's just better to create an object for the median before
        X = self.STFT.clockProcess(x)

        #Computing magnitude and phase spectrograms
        mX = U.getMagnitude(X)
        pX = U.getPhase(X)

        #Computing HPSS masks
        comp_harm, comp_perc = self.computing_hpss_masks_clock(mX) # how to do the masking in real-time?

        #Computing Harmonic and Percussive components
        Y_harm = mX*comp_harm*np.exp(1j*pX)
        Y_perc = mX*comp_perc*np.exp(1j*pX)

        y_harm = self._ISTFT_.clockProcess(Y_harm)
        y_perc = self._ISTFT_.clockProcess(Y_perc)

        return y_harm, y_perc
    '''

    def clear(self):
        pass

    def process(self, x):
        #Computing STFT
        X = self._STFT_.process(x)

        #Computing magnitude and phase spectrograms
        mX = U.getMagnitude(X)
        pX = U.getPhase(X)

        #Computing HPSS masks
        comp_harm, comp_perc = self.computing_hpss_masks(mX)

        #Computing Harmonic and Percussive components
        Y_harm = mX*comp_harm*np.exp(1j*pX)
        Y_perc = mX*comp_perc*np.exp(1j*pX)

        y_harm = self._ISTFT_.process(Y_harm)
        y_perc = self._ISTFT_.process(Y_perc)

        return y_harm, y_perc


def callback(nameInput='../sounds/piano.wav', prefixOutput='processed/sine_stretched_HPSS', format='.wav',
             win_harm=31, win_perc=31, masking='hard', hopSize=512, frameSize=2048, zeroPadding=0,
             windowType='hann', fftshift=True):
    
    #Loading audio
    x, fs = U.wavread(nameInput)

    param_STFT = STFT.Param_STFT(hopSize=hopSize, frameSize=frameSize, zeroPadding=zeroPadding, windowType=windowType,
                            fftshift=fftshift)
    param_HPSS = Param_HPSS(param_STFT=param_STFT, win_harm=win_harm, win_perc=win_perc, masking=masking)

    y_harm, y_perc = HPSS(param_HPSS=param_HPSS).process(x)

    #Writing audio
    U.wavwrite(y_harm, fs, prefixOutput+'harm'+format)
    U.wavwrite(y_perc, fs, prefixOutput+'perc'+format)
    
    #Plotting 
    fig = plt.figure(figsize=(15, 15), dpi= 80, facecolor='w', edgecolor='k')
    fig.canvas.set_window_title('Original Signal, Harmonic and Percussive Components')
    
    tmp = fig.add_subplot(3,1,1)
    tmp.plot(U.getTimeAxis(x, fs), x)
    tmp.set_title('Original Signal')
    tmp = fig.add_subplot(3,1,2)
    tmp.plot(U.getTimeAxis(y_harm, fs), y_harm)
    tmp.set_title('Harmonic Component')
    tmp = fig.add_subplot(3,1,3)
    tmp.plot(U.getTimeAxis(y_perc, fs), y_perc)  
    tmp.set_title('Percussive Component')
    
    
    frameSize_analysis = 3071
    zeroPadding_analysis = 1025
    hopSize_analysis = 1024

    param_visualization = STFT.Param_STFT(frameSize=frameSize_analysis, hopSize=hopSize_analysis, fftshift=fftshift,
                                          windowType=windowType, zeroPadding=zeroPadding_analysis)
    fftshift = True
    windowType ='hann'
    X = STFT.STFT(param_STFT=param_visualization).process(x)
    [tx, fx] = U.getSpectrogramAxis(X, fs, hopSize_analysis)
    endX = int(X.shape[1]/2+1)
    Y_harm = STFT.STFT(param_STFT=param_visualization).process(y_harm)
    endY_harm = int(Y_harm.shape[1]/2+1)
    [ty_harm, fy_harm] = U.getSpectrogramAxis(Y_harm, fs, hopSize_analysis)
    Y_perc = STFT.STFT(param_STFT=param_visualization).process(y_perc)
    endY_perc = int(Y_perc.shape[1]/2+1)
    [ty_perc, fy_perc] = U.getSpectrogramAxis(Y_perc, fs, hopSize_analysis)
    
    fig2 = plt.figure(figsize=(15, 15), dpi= 80, facecolor='w', edgecolor='k')
    fig2.canvas.set_window_title('Spectrograms')
    tmp = fig2.add_subplot(3,2,1)
    tmp.pcolormesh(tx, fx, np.transpose(U.amp2db(U.getMagnitude(X[:, :endX]))))
    tmp.set_title('Original Magnitude Spectrogram')
    tmp = fig2.add_subplot(3,2,2)
    plt.pcolormesh(tx, fx, np.transpose(np.diff(U.getPhase(X[:, :endX]))))
    tmp.set_title('Differential of the Original Phase Spectrogram')
    tmp = fig2.add_subplot(3,2,3)
    plt.pcolormesh(ty_harm, fy_harm, np.transpose(U.amp2db(U.getMagnitude(Y_harm[:, :endY_harm]))))
    tmp.set_title('Harmonic Component Magnitude Spectrogram')    
    tmp = fig2.add_subplot(3,2,4)
    plt.pcolormesh(ty_harm, fy_harm, np.transpose(np.diff(U.getPhase(Y_harm[:, :endY_harm]))))
    tmp.set_title('Differential of the Harmonic Component Phase Spectrogram')
    tmp = fig2.add_subplot(3,2,5)
    plt.pcolormesh(ty_perc, fy_perc, np.transpose(U.amp2db(U.getMagnitude(Y_perc[:, :endY_perc]))))
    tmp.set_title('Percussive Component Magnitude Spectrogram')    
    tmp = fig2.add_subplot(3,2,6)
    plt.pcolormesh(ty_perc, fy_perc, np.transpose(np.diff(U.getPhase(Y_perc[:, :endY_perc]))))
    tmp.set_title('Differential of the Percussive Component Phase Spectrogram')
    
    plt.show()

if __name__ == "__main__":
    callback()
