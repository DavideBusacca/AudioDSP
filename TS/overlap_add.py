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
from AudioDSP import utils as U
from AudioDSP.visualization import visualization as V

def overlap_add(x, alpha=1.5, Hs=512, frameSize=2048, normalization=True, windowType='hann'):
    '''
    Standard Overlap-Add algorithm for Time-Stretching.
        alpha is the Time-Stretching Factor
        Hs is the Synthesis Hop Size desired
        if normalization is True the output is divided by the windowing applied
    '''
    #Creating window
    w = U.windowing(np.ones(frameSize), frameSize, typeWindow=windowType)
    
    #Computing analysis hop size
    Ha = int(np.round(Hs/alpha))   
    
    y = np.zeros(int((x.size+2*frameSize)*alpha)) # output
    win = np.zeros_like(y) # sum of windows
    pointer = 0  

    #Framing
    nFrames = int(np.floor((x.size-(frameSize))/Ha))
    for pointer in range(nFrames):
        y[pointer*Hs:pointer*Hs+frameSize] =        \
            y[pointer*Hs:pointer*Hs+frameSize] +    \
            x[pointer*Ha:pointer*Ha+frameSize] * w
        win[pointer*Hs:pointer*Hs+frameSize] =      \
            win[pointer*Hs:pointer*Hs+frameSize] + w

    #Normalization
    if normalization == True:
        y = y / (win + np.finfo(float).eps)      
        
    return y
        

def callback(nameInput='AudioDSP/sounds/piano.wav', nameOutput='AudioDSP/TS/processed/piano_OLA.wav', alpha=1.5, Hs=2048, frameSize=8192, normalization=True, windowType='hann'):
    #Loading audio
    x, fs = U.wavread(nameInput)
    
    #Overlap-Add
    y = overlap_add(x, alpha=alpha, Hs=Hs, frameSize=frameSize, normalization=normalization, windowType=windowType)
    
    #Writing audio output
    U.wavwrite(y, fs, nameOutput)   

    #Plotting
    fig = V.createFigure(title='Original and Time-Stretched Signals')
    V.visualization_TD(x, fs, 'Original Signal', subplot=fig.add_subplot(2, 1, 1), show=False)
    V.visualization_TD(y, fs, 'Time-Stretched Signal (OLA)', subplot=fig.add_subplot(2, 1, 2))
    
if __name__ == "__main__":
    callback()
    
