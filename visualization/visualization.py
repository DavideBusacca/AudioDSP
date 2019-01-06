import sys, os
import numpy as np
import matplotlib.pyplot as plt
import utils as U

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../spectrogram/'))
import STFT as STFT

def control(signals, fs, names):
    if len(signals) == len(names):
        nPlot = len(signals)
    else:
        print("The number of elements given in the signals' list is not the same as the number of elements in the names' list")
        nPlot = min(len(signals), len(names))
        
    if fs != nPlot:
        print("Time-domain visualization function: The number of sample rates given is different respect to the number of signals given. The first sample rate will be used to display all the signals.")
        tmp = fs[0]
        fs = []
        for i in range(nPlot):
            fs.append(tmp)
            
    return nPlot, fs


def visualization_TD(signals, fs, names, title='A nice title (apparently)', show=True):
    '''
    Time-domain visualization
        fss: a list containing a sample rate or a list containing all the sample rates (one for each file)
    '''
    nPlot, fs = control(signals, fs, names)
    
    fig = plt.figure(figsize=(15, 15), dpi= 80, facecolor='w', edgecolor='k')
    fig.canvas.set_window_title(title)   
    for i in range(nPlot):
        tmp = fig.add_subplot(nPlot,1,i+1)
        tmp.plot(U.getTimeAxis(signals[i], fs[i]), signals[i])
        tmp.set_ylim((-1,1))
        tmp.set_title(names[i])
     
    if show==True:     
        plt.show()
        
def visualization_FD(signals, fss, names, title='A nice title (apparently)', show=True, frameSize_analysis=3071, zeroPadding_analysis=1025, hopSize_analysis=1024, fftshift=True, windowType='hann'):
    '''
    Frequency-domain visualization
    '''
    powerMagnitude = True
    
    nPlot, fss = control(signals, fss, names)
    
    fig2 = plt.figure(figsize=(15, 15), dpi= 80, facecolor='w', edgecolor='k')   
    fig2.canvas.set_window_title(title)
    for i in range(nPlot):
        fs = fss[i]
        X = STFT.STFT(signals[i], frameSize=frameSize_analysis, hopSize=hopSize_analysis, fftshift=fftshift, windowType=windowType, zeroPadding=zeroPadding_analysis)
        [tx, fx] = U.getSpectrogramAxis(X, fs, hopSize_analysis)
        endX = int(X.shape[1]/2+1)
        tmp = fig2.add_subplot(nPlot,2,i*2+1)
        if powerMagnitude == True:
            X = X*X
        tmp.pcolormesh(tx, fx, np.transpose(U.amp2db(U.getMagnitude(X[:, :endX]))), vmin=-60, vmax=120, cmap='gray_r')
        tmp.set_title(names[i]+" Magnitude Spectrogram")
        tmp = fig2.add_subplot(nPlot,2,i*2+2)
        plt.pcolormesh(tx, fx, np.transpose(np.diff(U.getPhase(X  [:, :endX]))))
        tmp.set_title('Differential of the '+ names[i] +' Phase Spectrogram')
    if show==True:     
        plt.show()
