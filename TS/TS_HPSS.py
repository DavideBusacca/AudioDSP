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
from AudioDSP.TS import phaseVocoder as PV
from AudioDSP.TS import overlap_add as OLA
from AudioDSP.HPSS import HPSS
from AudioDSP import utils as U
from AudioDSP.visualization import visualization as V

# TODO fix TS_HPSS with the new HPSS

def TS_HPSS(x, alpha=1.5,\
    win_harm_HPSS=31, win_perc_HPSS=31, masking_HPSS='hard', hopSize_HPSS=512, frameSize_HPSS=2048, zeroPadding_HPSS=0, windowType_HPSS='hann', fftshift_HPSS=True, normalized_HPSS=False, \
    Hs_PV=512, frameSize_PV=2048, zeroPadding_PV=0, phaseLocking_PV='identity', windowType_PV='hann', fftshift_PV=True, normalized_PV=False, \
    Hs_OLA=512, frameSize_OLA=2048, normalization_OLA=True, windowType_OLA='hann'): 
    '''
    Time-Stretching Technique based on Harmonic/Percussive Separation
    With (or without) identity phase locking scheme
    '''
    
    #Harmonic/Percussive Separation
    x_harm, x_perc = HPSS.hpss(x, win_harm=win_harm_HPSS, win_perc=win_perc_HPSS, masking=masking_HPSS, hopSize=hopSize_HPSS, frameSize=frameSize_HPSS, zeroPadding=zeroPadding_HPSS, windowType=windowType_HPSS, fftshift=fftshift_HPSS, normalized=normalized_HPSS)
    #Phase vocoder on the Harmonic Component
    y_harm, ow = PV.phaseVocoder(x_harm, alpha=alpha, Hs=Hs_PV, frameSize=frameSize_PV, zeroPadding=zeroPadding_PV, phaseLocking=phaseLocking_PV, windowType=windowType_PV, fftshift=fftshift_PV, normalized=normalized_PV)
    #Overlap-add on the percussive component
    y_perc = OLA.overlap_add(x_perc, alpha=alpha, Hs=Hs_OLA, frameSize=frameSize_OLA, normalization=normalization_OLA, windowType=windowType_OLA)
    #Superposing
    end = min(y_harm.size, y_perc.size)
    y = y_harm[:end] + y_perc[:end]

    return y, x_harm, x_perc, y_harm, y_perc 

def callback(nameInput='../sounds/piano.wav', nameOutput='processed/piano_stretched_TS_HPSS', format='.wav', alpha=1.5,\
    win_harm_HPSS=31, win_perc_HPSS=17, masking_HPSS='hard', hopSize_HPSS=512, frameSize_HPSS=2048, zeroPadding_HPSS=0, windowType_HPSS='hann', fftshift_HPSS=True, normalized_HPSS=False, \
    Hs_PV=1024, frameSize_PV=8192, zeroPadding_PV=0, phaseLocking_PV='identity', windowType_PV='hann', fftshift_PV=True, normalized_PV=False, \
    Hs_OLA=128, frameSize_OLA=256, normalization_OLA=True, windowType_OLA='hann'):
    
    #Loading audio
    x, fs = U.wavread(nameInput)
    
    #Phase vocoder
    y, x_harm, x_perc, y_harm, y_perc = TS_HPSS(x, alpha=alpha,\
    win_harm_HPSS=win_harm_HPSS, win_perc_HPSS=win_perc_HPSS, masking_HPSS=masking_HPSS, hopSize_HPSS=hopSize_HPSS, frameSize_HPSS=frameSize_HPSS, zeroPadding_HPSS=zeroPadding_HPSS, windowType_HPSS=windowType_HPSS, fftshift_HPSS=fftshift_HPSS, normalized_HPSS=normalized_HPSS, \
    Hs_PV=Hs_PV, frameSize_PV=frameSize_PV, zeroPadding_PV=zeroPadding_PV, phaseLocking_PV=phaseLocking_PV, windowType_PV=windowType_PV, fftshift_PV=fftshift_PV, normalized_PV=normalized_PV, \
    Hs_OLA=Hs_OLA, frameSize_OLA=frameSize_OLA, normalization_OLA=normalization_OLA, windowType_OLA=windowType_OLA)
    
    #Writing audio output
    U.wavwrite(x_harm, fs, nameOutput+'_x_harm'+format)
    U.wavwrite(x_perc, fs, nameOutput+'_x_perc'+format)
    U.wavwrite(y, fs, nameOutput+format)
    U.wavwrite(y_harm, fs, nameOutput+'_y_harm'+format)
    U.wavwrite(y_perc, fs, nameOutput+'_y_perc'+format)
    
    #Plotting
    title = 'Original Signal' 
    names = ['Original Signal', 'Original Signal (Harmonic Component)', 'Original Signal (Percussive Component)']
    signals = [x, x_harm, x_perc] 
    V.visualization_TD(signals, [fs], names, title, show=False)
    
    title = 'Time-Stretched Signal'
    names = ['Time-Stretched Signal', 'Time-Stretched Signal (Harmonic Component)', 'Time-Stretched Signal (Percussive Component)']
    signals = [y, y_harm, y_perc]
    V.visualization_TD(signals, [fs], names, title, show=False)
    
    title = 'Original Spectrograms' 
    names = ['Original ', 'Original (Harmonic Component)', 'Original (Percussive Component)']
    signals = [x, x_harm, x_perc] 
    V.visualization_FD(signals, [fs], names, title, show=False)
    
    title = 'Time-Stretched Signal'
    names = ['Time-Stretched ', 'Time-Stretched (Harmonic Component)', 'Time-Stretched (Percussive Component)']
    signals = [y, y_harm, y_perc]
    V.visualization_FD(signals, [fs], names, title, show=True)
    
if __name__ == "__main__":
    callback()
