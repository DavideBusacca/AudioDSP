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
from math import fmod
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../spectrogram/'))
import STFT as STFT
import ISTFT as ISTFT

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
import utils as U

def phaseVocoder(x, alpha=1.5, Hs=512, frameSize=2048, zeroPadding=0, phaseLocking='identity', windowType='hann', fftshift=True, normalized=False): 
    '''
    Phase Vocoder
    With (or without) identity phase locking scheme
    '''
    Ha = int(np.round(Hs/alpha))        
       
    #Computing STFT
    X = STFT.STFT(x, frameSize=frameSize, hopSize=Ha, fftshift=fftshift, windowType=windowType, zeroPadding=zeroPadding)

    #Phase Propagation
    if phaseLocking == 'identity':
        print("Identity phase locking")
        Y = identityPhaseLocking(X, Hs, alpha, frameSize)
    else:
        print("No phase locking")
        Y = phase_propagation(X, Hs, alpha, frameSize)
    
    #Computing ISTFT
    [y, ow] = ISTFT.ISTFT(Y, frameSize=frameSize, hopSize=Hs, fftshift=fftshift, windowType=windowType, normalized=normalized, zeroPadding=zeroPadding)
    
    return y, ow
    
def phase_propagation(X, Hs = 512, alpha=1.5, frameSize=2048): 
    Ha = int(np.round(Hs/alpha)) 
    pX = np.angle(X)
    
    omega = np.arange(X.shape[1])
    omega = 2*np.pi*omega/frameSize #bin's frequency
    expected_increment = Ha*omega
    
    pY = np.zeros_like(pX)
    pY[0,:]=pX[0,:]
    
    for m in np.arange(X.shape[0]-1)+1:
        for k in np.arange(X.shape[1]):
            #heterodyned phase increment
            kappa = (pX[m, k] - pX[m-1, k]) - expected_increment[k]
            kappa = fmod(kappa, 2*np.pi)
            #instantaneous frequency
            omega_instantaneous = omega[k] + kappa/Ha
            #phase updating
            pY[m,k] = pY[m-1, k] + Hs*omega_instantaneous
            
    #Application of the new phase spectrogram to the old spectrogram        
    mX = np.abs(X)  
    Y = mX*np.exp(1j*pY)
          
    return Y
    
def identityPhaseLocking(X, Hs, alpha, frameSize):
    mX = np.abs(X)
    pX = np.angle(X)
    #find peaks and regions of influence
    binPeaksInd, beginRegionInfluence, endRegionInfluence = computeRegionsOfInfluence(mX)
    #Update the phase of the peaks using the standard phase propagation equation
    pY = np.angle(phase_propagation(X, Hs, alpha, frameSize))
    #Update the phase of the bins in the region of influence
    for m in np.arange(X.shape[0]-1)+1:
        for kp, b, e in zip(binPeaksInd[m], beginRegionInfluence[m], endRegionInfluence[m]) :
            for k in range(b,e):
                kp = int(kp)
                k = int(k) #bin index to be processed
                pY[m,k] = pX[m,k]+ fmod(pY[m,kp]-pX[m,kp], 2*np.pi)
                  
    Y = mX*np.exp(1j*pY)
    
    return Y


def computeRegionsOfInfluence(mX):
    ''''
    Find peak bins
    A peak bin is a bin whose magnitude is higher than the magnitude of the 2 previous and the 2 successive bins.
    '''
    binPeaksInd = []
    for m in mX:
        binPeaksInd_tmp=()
        for k in np.arange(m.size-2)+2:
            if np.all(m[k] >= m[k-2:k+3]):
                binPeaksInd_tmp = np.append(binPeaksInd_tmp, k)
        binPeaksInd.append(binPeaksInd_tmp)
    '''
    Computing regions of influence
    Given two neighbor peaks, the borderline is the bin in the middle
    '''
    beginRegionInfluence=[]
    endRegionInfluence=[]
    for p in binPeaksInd:
        beginRegionInfluence_tmp=[0]
        endRegionInfluence_tmp=[]  
        for peak_ind in np.arange(p.size-1)+1:
            tmp = int(np.round((p[peak_ind]+p[peak_ind-1])/2))
            endRegionInfluence_tmp.append(tmp-1)
            beginRegionInfluence_tmp.append(tmp)
        endRegionInfluence_tmp.append(m.size)
        beginRegionInfluence.append(beginRegionInfluence_tmp)
        endRegionInfluence.append(endRegionInfluence_tmp)           

    return binPeaksInd, beginRegionInfluence, endRegionInfluence

   

def callback(nameInput='../sounds/sine.wav', nameOutput='processed/sine_stretched_PV_IPL.wav', alpha=1.5, Hs=512, frameSize=4096, phaseLocking='identity'):
    
    #Loading audio
    x, fs = U.wavread(nameInput)
    
    #Phase vocoder
    y, ow = phaseVocoder(x, alpha=alpha, Hs=Hs, frameSize=frameSize, phaseLocking=phaseLocking)

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
    tmp.set_title('Time-Stretched Signal')
    tmp = fig.add_subplot(3,1,3)
    tmp.plot(U.getTimeAxis(ow, fs), ow)  
    tmp.set_title('Sum of (Squared) Windows')
    
    
    frameSize_analysis = 3071
    zeroPadding_analysis = 1025
    hopSize_analysis = 1024
    fftshift = True
    windowType ='hann'
    X = STFT.STFT(x, frameSize=frameSize_analysis, hopSize=hopSize_analysis, fftshift=fftshift, windowType=windowType, zeroPadding=zeroPadding_analysis)
    [tx, fx] = U.getSpectrogramAxis(X, fs, hopSize_analysis)
    endX = int(X.shape[1]/2+1)
    Y = STFT.STFT(y, frameSize=frameSize_analysis, hopSize=hopSize_analysis, fftshift=fftshift, windowType=windowType, zeroPadding=zeroPadding_analysis)
    endY = int(Y.shape[1]/2+1)
    [ty, fy] = U.getSpectrogramAxis(Y, fs, hopSize_analysis)
    
    fig2 = plt.figure(figsize=(15, 15), dpi= 80, facecolor='w', edgecolor='k')
    fig2.canvas.set_window_title('Spectrograms')
    tmp = fig2.add_subplot(2,2,1)
    tmp.pcolormesh(tx, fx, np.transpose(U.amp2db(U.getMagnitude(X[:, :endX]))))
    tmp.set_title('Original Magnitude Spectrogram')
    tmp = fig2.add_subplot(2,2,3)
    plt.pcolormesh(tx, fx, np.transpose(np.diff(U.getPhase(X[:, :endX]))))
    tmp.set_title('Differential of the Original Phase Spectrogram')
    tmp = fig2.add_subplot(2,2,2)
    plt.pcolormesh(ty, fy, np.transpose(U.amp2db(U.getMagnitude(Y[:, :endY]))))
    tmp.set_title('Time-Stretched Magnitude Spectrogram')    
    tmp = fig2.add_subplot(2,2,4)
    plt.pcolormesh(ty, fy, np.transpose(np.diff(U.getPhase(Y[:, :endY]))))
    tmp.set_title('Differential of the Time-Stretched Phase Spectrogram')
    plt.show()

if __name__ == "__main__":
    callback()
