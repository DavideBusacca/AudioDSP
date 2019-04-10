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
import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
import matplotlib.pyplot as plt
from AudioDSP import utils as U
from AudioDSP.spectrogram import STFT

def createFigure(title=""):
    figure = plt.figure(figsize=(15, 15), dpi= 80, facecolor='w', edgecolor='k')
    figure.canvas.set_window_title(title)

    return figure

def show():
    plt.show()

def visualization_TD(x, fs, name, show=True, subplot=None, yNormalization=True):
    '''
    Time-domain visualization
        fss: a list containing a sample rate or a list containing all the sample rates (one for each file)
    '''

    if subplot is None:
        fig = createFigure(title=name)
        subplot = fig.add_subplot(1,1,1)

    subplot.plot(U.getTimeAxis(x, fs), x)
    subplot.set_title(name)
    if yNormalization:
        subplot.set_ylim((-1,1))
     
    if show==True:     
        plt.show()
        
def visualization_FD(x, fs, name, show=True, param_analysis_STFT=None, mX_subplot=None, pX_subplot=None,
                     powerMagnitude=True, dbMagnitude=True, differentialPhase=True):
    '''
    Frequency-domain visualization of a signal
    '''

    if mX_subplot is None and pX_subplot is None:
        fig = createFigure(title=name)
        mX_subplot = fig.add_subplot(2, 1, 1)
        pX_subplot = fig.add_subplot(2, 1, 2)

    analysis_STFT = STFT.STFT(param_analysis_STFT)
    X = analysis_STFT.process(x)
    [tx, fx] = U.getSpectrogramAxis(X, fs, analysis_STFT.getHopSize())
    endX = int(X.shape[1]/2+1)

    mX = U.getMagnitude(X[:, :endX])
    pX = U.getPhase(X  [:, :endX])
    if powerMagnitude == True:
        mX = mX**2
    if dbMagnitude == True:
        mX = U.amp2db(mX)
    if differentialPhase == True:
        np.diff(pX)

    if mX_subplot:
        mX_subplot.pcolormesh(tx, fx, np.transpose(mX), vmin=-60, vmax=120, cmap='gray_r') #use a param to pass these values!
        mX_subplot.set_title(name + " Magnitude Spectrogram")
    if pX_subplot:
        pX_subplot.pcolormesh(tx, fx, np.transpose(pX))
        pX_subplot.set_title('Differential of the ' + name + ' Phase Spectrogram')

    if show==True:     
        plt.show()

def callback(nameInput='../sounds/sine.wav', frameSize=3071, zeroPadding=1025, hopSize=2048, fftshift=True,
                  windowType='hann'):
    # Loading audio
    x, fs = U.wavread(nameInput)
    name = "Sinewave"

    # Visualization in time domain
    fig = createFigure(title="Visualization Test")
    visualization_TD(x, fs, name, subplot=fig.add_subplot(3, 1, 1), show=False)

    # Visualization frequency domain
    param_analysis_STFT = STFT.Param_STFT(frameSize=3071, zeroPadding=1025, hopSize=1024, fftshift=True,
                                          windowType='hann')
    visualization_FD(x, fs, name, param_analysis_STFT=param_analysis_STFT,
                     mX_subplot=fig.add_subplot(3, 1, 2), pX_subplot=fig.add_subplot(3, 1, 3), show=False)

    visualization_TD(x, fs, name, show=False)
    visualization_FD(x, fs, name, param_analysis_STFT=param_analysis_STFT)

if __name__ == '__main__':
    callback()
