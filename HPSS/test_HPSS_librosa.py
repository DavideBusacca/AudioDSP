import sys, os
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import essentia.standard as ess
import librosa
import HPSS_essentia
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
import utils as U


def compare(X, X2):
    # plt.figure()
    # plt.pcolormesh(U.getMagnitude(X - X2))
    print(np.sum(np.sum(U.getMagnitude(X - X2))))

def main(nameInput='../sounds/piano.wav', fs=44100, hopSize=512, frameSize=2048, zeroPadding=0, windowType='hann',
         kernel=(17, 17), margin=(1, 1), power=2.0):
    x = ess.MonoLoader(filename=nameInput, sampleRate=fs)()
    X = HPSS_essentia.compute_STFT(x, frameSize=frameSize, hopSize=hopSize, zeroPadding=zeroPadding,
                                   windowType=windowType)
    H_essentia, P_essentia = HPSS_essentia.HPSS(X, kernel=kernel, margin=margin, power=power)
    H_librosa, P_librosa = librosa.decompose.hpss(np.transpose(X), kernel_size=kernel, margin=margin, power=power)

    compare(H_essentia, np.transpose(H_librosa))
    compare(P_essentia, np.transpose(P_librosa))
    # plt.show()


if __name__ == '__main__':
    power = [0.5, 1, 2, 10, np.inf]
    kernels = [(17), (11, 11), (13, 7), (41, 51)]
    margins = [(1), (1, 1), (1.5, 1.3)]
    for kernel in kernels:
        for margin in margins:
            main(kernel=kernel, margin=margin)
