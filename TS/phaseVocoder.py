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
import numpy as np
from math import fmod
from AudioDSP.spectrogram import STFT, ISTFT
from AudioDSP.visualization import visualization as V
from AudioDSP import utils as U

def phaseVocoder(x, alpha=1.5, phaseLocking='identity', param_STFT=None):
    '''
    Phase Vocoder
    With (or without) identity phase locking scheme
    '''

    if param_STFT == None:
        param_STFT = STFT.Param_STFT(hopSize=512, frameSize=4096, zeroPadding=0, windowType='hann', fftshift=True)

    Hs = param_STFT.hopSize
    Ha = int(np.round(Hs/alpha))        
       
    #Computing STFT
    param_STFT.hopSize = Ha
    X = STFT.STFT(param_STFT).process(x)

    #Phase Propagation
    if phaseLocking == 'identity':
        print("Identity phase locking")
        Y = identityPhaseLocking(X, Hs, alpha, param_STFT.frameSize)
    else:
        print("No phase locking")
        Y = phase_propagation(X, Hs, alpha, param_STFT.frameSize)
    
    #Computing ISTFT
    param_STFT.hopSize = Hs
    y = ISTFT.ISTFT(param_STFT).process(Y)
    
    return y
    
def phase_propagation(X, Hs = 512, alpha=1.5, frameSize=2048): 
    Ha = int(np.round(Hs/alpha)) 
    pX = np.angle(X)
    
    omega = np.arange(X.shape[1])
    omega = 2 * np.pi * omega / frameSize  # bin's frequency
    expected_increment = Ha * omega
    
    pY = np.zeros_like(pX)
    pY[0, :] = pX[0, :]
    
    for m in np.arange(X.shape[0]-1)+1:
        for k in np.arange(X.shape[1]):
            # Heterodyned Phase Increment
            kappa = (pX[m, k] - pX[m-1, k]) - expected_increment[k]
            kappa = fmod(kappa, 2*np.pi)
            # Instantaneous Frequency
            omega_instantaneous = omega[k] + kappa/Ha
            # Phase Updating
            pY[m, k] = pY[m-1, k] + Hs*omega_instantaneous
            
    # Application of the new phase spectrogram to the old spectrogram
    mX = np.abs(X)  
    Y = mX * np.exp(1j*pY)
          
    return Y
    
def identityPhaseLocking(X, Hs, alpha, frameSize):
    mX = np.abs(X)
    pX = np.angle(X)

    # Find peaks and regions of influence
    binPeaksInd, beginRegionInfluence, endRegionInfluence = computeRegionsOfInfluence(mX)

    # Update the phase of the peaks using the standard phase propagation equation
    pY = np.angle(phase_propagation(X, Hs, alpha, frameSize))

    # Update the phase of the bins in the region of influence
    for m in np.arange(X.shape[0]-1)+1:
        for kp, b, e in zip(binPeaksInd[m], beginRegionInfluence[m], endRegionInfluence[m]) :
            for k in range(b, e):
                kp = int(kp)
                k = int(k)  # bin index to be processed
                pY[m, k] = pX[m, k] + fmod(pY[m, kp]-pX[m, kp], 2*np.pi)
                  
    Y = mX * np.exp(1j*pY)
    
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

   

def callback(nameInput='AudioDSP/sounds/sine.wav', nameOutput='AudioDSP/TS/processed/sine_stretched_PV_IPL.wav',
             alpha=1.5, phaseLocking='identity', param_STFT=None):
    
    #Loading audio
    x, fs = U.wavread(nameInput)
    
    #Phase vocoder
    y = phaseVocoder(x, alpha=alpha, phaseLocking=phaseLocking, param_STFT=None)

    #Writing audio output
    U.wavwrite(y, fs, nameOutput)

    #Plotting
    fig = V.createFigure(title='Signals and Window')
    V.visualization_TD(x, fs, 'Original Signal', subplot=fig.add_subplot(2, 1, 1), show=False)
    V.visualization_TD(y, fs, 'Time-Stretched Signal', subplot=fig.add_subplot(2, 1, 2), show=False)

    fig2 = V.createFigure(title='Spectrograms')
    param_analysis_STFT = STFT.Param_STFT(frameSize=3071, zeroPadding=1025, hopSize=1024, fftshift=True,
                                          windowType='hann')
    V.visualization_FD(x, fs, 'Original', param_analysis_STFT=param_analysis_STFT,
                     mX_subplot=fig2.add_subplot(2, 2, 1), pX_subplot=fig2.add_subplot(2, 2, 2), show=False)
    V.visualization_FD(y, fs, 'Time-Stretched', param_analysis_STFT=param_analysis_STFT,
                     mX_subplot=fig2.add_subplot(2, 2, 3), pX_subplot=fig2.add_subplot(2, 2, 4))

if __name__ == "__main__":
    callback()
