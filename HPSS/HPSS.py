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
from scipy.ndimage import median_filter
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
import utils as U
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../spectrogram/'))
import STFT
import ISTFT
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../visualization'))
import visualization as V
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utils/'))
import median_filtering as MF

import matplotlib.pyplot as plt


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

    def softMask(self, harm, perc):
        '''
        Soft masking for Harmonic/Percussive Source Separation
        '''
        total = harm + perc
        mask_harm = harm / total
        mask_perc = perc / total

        return mask_harm, mask_perc

    def clear(self):
        pass

    ''' #block-by-block implementation
    def process_using_clockProcess(self, x):
        if self._needsUpdate_ == True:
            self.update()

        nFrames = int(np.floor((x.size-self._STFT_.getFrameSize())/self._STFT_.getHopSize()))
        for f in range(nFrames):
            frame = x[self._STFT_.getHopSize()*f:self._STFT_.getHopSize()*f+self._STFT_.getFrameSize()]
            frame = self._STFT_.clockProcess(frame)

            # block-by-block process
    '''

    def process(self, x):
        if self._needsUpdate_ == True:
            self.update()

        #Computing STFT
        X = self._STFT_.process(x)

        #Computing magnitude and phase spectrograms
        mX = U.getMagnitude(X)
        pX = U.getPhase(X)

        #Computing Enhanced Spectrograms
        harm, perc = MF.computing_enhanced_spectrograms(mX, win_harm=self._win_harm_, win_perc=self._win_perc_)

        #Computing harmonic and percussive masks
        if self._masking_ == 'hard':
            mask_harm, mask_perc = self.hardMask(harm, perc)
        elif self._masking_ == 'soft':
            mask_harm, mask_perc = self.softMask(harm, perc)

        #Computing Harmonic and Percussive components
        Y_harm = (mX*mask_harm) * np.exp(1j*pX)
        Y_perc = (mX*mask_perc) * np.exp(1j*pX)

        y_harm = self._ISTFT_.process(Y_harm)
        y_perc = self._ISTFT_.process(Y_perc)

        return y_harm, y_perc


def callback(nameInput='../sounds/piano.wav', prefixOutput='processed/sine_stretched_HPSS', format='.wav',
             win_harm=17, win_perc=17, masking='soft', hopSize=512, frameSize=2048, zeroPadding=0,
             windowType='hann', fftshift=True):
    
    # Loading audio
    x, fs = U.wavread(nameInput)

    param_STFT = STFT.Param_STFT(hopSize=hopSize, frameSize=frameSize, zeroPadding=zeroPadding, windowType=windowType,
                            fftshift=fftshift)
    param_HPSS = Param_HPSS(param_STFT=param_STFT, win_harm=win_harm, win_perc=win_perc, masking=masking)

    y_harm, y_perc = HPSS(param_HPSS=param_HPSS).process(x)

    # Writing audio
    U.wavwrite(y_harm, fs, prefixOutput+'harm'+format)
    U.wavwrite(y_perc, fs, prefixOutput+'perc'+format)

    # Visualization
    fig = V.createFigure(title="Original and Harmonic/Percussive Components Signals")
    V.visualization_TD(x, fs, name="Original Signal", subplot=fig.add_subplot(3, 1, 1), show=False)
    V.visualization_TD(y_harm, fs, name="Harmonic Component", subplot=fig.add_subplot(3, 1, 2), show=False)
    V.visualization_TD(y_perc, fs, name="Percussive Component", subplot=fig.add_subplot(3, 1, 3), show=False)

    fig = V.createFigure(title="Original and Harmonic/Percussive Components Spectrograms")
    frameSize_analysis = 3071
    zeroPadding_analysis = 1025
    hopSize_analysis = 1024
    param_visualization = STFT.Param_STFT(frameSize=frameSize_analysis, hopSize=hopSize_analysis, fftshift=fftshift,
                                          windowType=windowType, zeroPadding=zeroPadding_analysis)
    V.visualization_FD(x, fs, name="Original", param_analysis_STFT=param_visualization,
                     mX_subplot=fig.add_subplot(3, 2, 1), pX_subplot=fig.add_subplot(3, 2, 2), show=False)
    V.visualization_FD(y_harm, fs, name="Harmonic Component", param_analysis_STFT=param_visualization,
                     mX_subplot=fig.add_subplot(3, 2, 3), pX_subplot=fig.add_subplot(3, 2, 4), show=False)
    V.visualization_FD(y_perc, fs, name="Percussive Component", param_analysis_STFT=param_visualization,
                     mX_subplot=fig.add_subplot(3, 2, 5), pX_subplot=fig.add_subplot(3, 2, 6), show=True)
                     



if __name__ == "__main__":
    callback()
