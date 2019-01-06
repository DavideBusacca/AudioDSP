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
import STFT as STFT
import ISTFT as ISTFT

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
import utils as U

def hardMask(harm, perc):
    '''
    Hard masking for Harmonic/Percussive Source Separation
    '''
    mask_harm = harm > perc
    mask_perc = perc > harm

    return mask_harm, mask_perc


def computing_hpss_masks(mX, win_harm=31, win_perc=31, masking='hard'):
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
    harm[:] = median_filter(mX, size=(win_perc, 1), mode='reflect')
	#Computing percussive enhanced spectrogram
    perc = np.empty_like(mX)
    perc[:] = median_filter(mX, size=(1, win_harm), mode='reflect')

    #Computing harmonic and percussive masks
    if masking == 'hard':
        mask_harm, mask_perc = hardMask(harm, perc)

    return mask_harm, mask_perc
    
def hpss(x, win_harm=31, win_perc=31, masking='hard', hopSize=512, frameSize=2048, zeroPadding=0, windowType='hann', fftshift=True, normalized=False):
    '''
    Median-filtering harmonic percussive source separation (HPSS)
    
    '''
    #Computing STFT
    X = STFT.STFT(x, frameSize=frameSize, hopSize=hopSize, fftshift=fftshift, windowType=windowType, zeroPadding=zeroPadding)
    
    #Computing magnitude and phase spectrograms
    mX = U.getMagnitude(X)
    pX = U.getPhase(X)
    
    #Computing HPSS masks
    comp_harm, comp_perc = computing_hpss_masks(mX, win_harm=31, win_perc=31, masking='hard')
    
    #Computing Harmonic and Percussive components
    Y_harm = mX*comp_harm*np.exp(1j*pX)
    Y_perc = mX*comp_perc*np.exp(1j*pX)
    y_harm, ow = ISTFT.ISTFT(Y_harm, frameSize=frameSize, hopSize=hopSize, fftshift=fftshift, windowType=windowType, zeroPadding=zeroPadding)
    y_perc, ow = ISTFT.ISTFT(Y_perc, frameSize=frameSize, hopSize=hopSize, fftshift=fftshift, windowType=windowType, zeroPadding=zeroPadding)
    
    return y_harm, y_perc
    

def callback(nameInput='../sounds/piano.wav', prefixOutput='processed/sine_stretched_HPSS', format='.wav', win_harm=31, win_perc=31, masking='hard', hopSize=512, frameSize=2048, zeroPadding=0, windowType='hann', fftshift=True, normalized=False):
    
    #Loading audio
    x, fs = U.wavread(nameInput)
    
    #Harmonic/Percussive Separation   
    y_harm, y_perc = hpss(x, win_harm=win_harm, win_perc=win_perc, masking=masking, hopSize=hopSize, frameSize=frameSize, zeroPadding=zeroPadding, windowType=windowType, fftshift=fftshift, normalized=normalized)
    
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
    fftshift = True
    windowType ='hann'
    X = STFT.STFT(x, frameSize=3071, hopSize=hopSize_analysis, fftshift=fftshift, windowType=windowType, zeroPadding=zeroPadding_analysis)
    [tx, fx] = U.getSpectrogramAxis(X, fs, hopSize_analysis)
    endX = int(X.shape[1]/2+1)
    Y_harm = STFT.STFT(y_harm, frameSize=frameSize_analysis, hopSize=hopSize_analysis, fftshift=fftshift, windowType=windowType, zeroPadding=zeroPadding_analysis)
    endY_harm = int(Y_harm.shape[1]/2+1)
    [ty_harm, fy_harm] = U.getSpectrogramAxis(Y_harm, fs, hopSize_analysis)
    Y_perc = STFT.STFT(y_perc, frameSize=frameSize_analysis, hopSize=hopSize_analysis, fftshift=fftshift, windowType=windowType, zeroPadding=zeroPadding_analysis)
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
    y_harm, y_perc = callback()
    
