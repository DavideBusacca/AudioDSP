import sys, os
import essentia.standard as ess
from essentia import array as essarray
import median_filtering as MF
import numpy as np
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
import utils as U
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../spectrogram/'))
import STFT
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../visualization'))
import visualization as V


def compute_STFT(x, frameSize=2048, hopSize=1024, zeroPadding=0, windowType='hann'):
    window = ess.Windowing(size=frameSize, type=windowType)
    fft = ess.FFT(size=frameSize+zeroPadding)

    # Computing spectrogram (STFT)
    flag = True
    for frame in ess.FrameGenerator(x, frameSize=frameSize, hopSize=hopSize, startFromZero=True):
        if flag:
            X = fft(window(frame))
            flag = False
        else:
            X = np.vstack((X, fft(window(frame))))

    return X


def compute_ISTFT(X, frameSize=2048, hopSize=1024, zeroPadding=0, windowType='hann', len=None):
    window = ess.Windowing(size=frameSize, type=windowType)
    ifft = ess.IFFT(size=frameSize+zeroPadding)
    # ola = ess.OverlapAdd(frameSize=frameSize, hopSize=hopSize) # I wasn't able to use that for the ISTFT
    # f2r = ess.FrameToReal(frameSize=frameSize, hopSize=hopSize) # not tried yet

    if len == None:
        len = (X.shape[0]-1) * hopSize+frameSize
    x = np.zeros(len)

    w = window(essarray(np.ones(frameSize)))
    ow = np.zeros_like(x)
    for i in range(X.shape[0]-1):
        start = hopSize * i
        end = start + frameSize
        x[start:end] = x[start:end] + window(ifft(X[i]))
        ow[start:end] = ow[start:end] + w

    x = x/ow

    return x


def HPSS(X, kernel=(17, 17), margin=(1, 1), power=2.0):
    # Implementation of harmonic percussive separation using median filtering with the Essentia library

    # kernel(length harmonic median filter, length percussive median filter)
    # margin(margin harmonic, margin percussive)

    if np.isscalar(kernel):
        win_harm = kernel
        win_perc = kernel
    else:
        win_harm = kernel[0]
        win_perc = kernel[1]

    if np.isscalar(margin):
        margin_harm = max(1, margin) # margin should be at least 1
        margin_perc = max(1, margin)
    else:
        margin_harm = max(1, margin[0])
        margin_perc = max(1, margin[1])

    # Computing magnitude and phase spectrograms
    mX = U.getMagnitude(X)
    pX = U.getPhase(X)

    # Computing harmonic and percussive enhanced magnitude spectrograms
    mH, mP = MF.compute_enhanced_spectrograms(mX, win_harm=win_harm, win_perc=win_perc)

    # Computing harmonic and percussive masks
    mask_harm, mask_perc = MF.compute_masks(mH, mP, power=power, margin_harm=margin_harm, margin_perc=margin_perc)

    # Computing harmonic and percussive components spectrograms
    Y_harm = (mX*mask_harm) * np.exp(1j*pX)
    Y_perc = (mX*mask_perc) * np.exp(1j*pX)

    return Y_harm, Y_perc


def HPSS_routine_essentia(x, hopSize=512, frameSize=2048, zeroPadding=0, windowType='hann',
                          kernel=(17, 17), margin=(1, 1), power=2.0):

    X = compute_STFT(x, frameSize=frameSize, hopSize=hopSize, zeroPadding=zeroPadding, windowType=windowType)
    Y_harm, Y_perc = HPSS(X, kernel=kernel, margin=margin, power=power)
    y_harm = compute_ISTFT(Y_harm, frameSize=frameSize, hopSize=hopSize, zeroPadding=zeroPadding,
                           windowType=windowType, len=x.shape[0])
    y_perc = compute_ISTFT(Y_perc, frameSize=frameSize, hopSize=hopSize, zeroPadding=zeroPadding,
                           windowType=windowType, len=x.shape[0])

    return y_harm, y_perc


def visualize_HPSS(x, fs, y_harm, y_perc):
    # Visualization
    fig = V.createFigure(title="Original and Harmonic/Percussive Components Signals")
    V.visualization_TD(x, fs, name="Original Signal", subplot=fig.add_subplot(3, 1, 1), show=False)
    V.visualization_TD(y_harm, fs, name="Harmonic Component", subplot=fig.add_subplot(3, 1, 2), show=False)
    V.visualization_TD(y_perc, fs, name="Percussive Component", subplot=fig.add_subplot(3, 1, 3), show=False)

    fig = V.createFigure(title="Original and Harmonic/Percussive Components Spectrograms")
    param_visualization = STFT.Param_STFT(frameSize=3071, zeroPadding=1025, hopSize=1024, fftshift=True,
                                          windowType='hann')
    V.visualization_FD(x, fs, name="Original", param_analysis_STFT=param_visualization,
                       mX_subplot=fig.add_subplot(3, 2, 1), pX_subplot=fig.add_subplot(3, 2, 2), show=False)
    V.visualization_FD(y_harm, fs, name="Harmonic Component", param_analysis_STFT=param_visualization,
                       mX_subplot=fig.add_subplot(3, 2, 3), pX_subplot=fig.add_subplot(3, 2, 4), show=False)
    V.visualization_FD(y_perc, fs, name="Percussive Component", param_analysis_STFT=param_visualization,
                       mX_subplot=fig.add_subplot(3, 2, 5), pX_subplot=fig.add_subplot(3, 2, 6), show=True)



def main(nameInput='../sounds/piano.wav', fs = 44100, kernel = (17, 17), margin = (1, 1), power=np.inf,
         hopSize=512, frameSize=2048, zeroPadding=0, windowType='hann'):

    x = ess.MonoLoader(filename=nameInput, sampleRate=fs)()

    y_harm, y_perc = HPSS_routine_essentia(x, hopSize=hopSize, frameSize=frameSize, zeroPadding=zeroPadding,
                                           windowType=windowType, kernel=kernel, margin=margin, power=power)

    visualize_HPSS(x, fs, y_harm, y_perc)

if __name__ == '__main__':
    main()
