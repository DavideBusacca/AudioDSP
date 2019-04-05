'''
 Copyright (C) 2018  Busacca Davide

 This file is part of AudioDSP-Python.

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
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import essentia.standard as ess
import librosa
import HPSS_essentia
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
import utils as U

'''
Raw testing for the proposed implementation of Harmonic Percussive Separation based on Median Filtering (HPSS).
'''

def compare(X, X2):
    '''
    Print the sum of the differences between each bin of two magnitude spectrograms.

    :param X:
    :param X2:
    :return:
    '''
    print(np.sum(np.sum(U.getMagnitude(X - X2))))

def main(nameInput='../sounds/piano.wav', fs=44100, hopSize=512, frameSize=2048, zeroPadding=0, windowType='hann',
         kernel=(17, 17), margin=(1, 1), power=2.0):
    '''
    Compares the result of the HPSS proposed and its Librosa implementation.
    The spectrogram is computed using Essentia (computing the spectrogram using Librosa leads to different results).

    :param nameInput:
    :param fs:
    :param hopSize:
    :param frameSize:
    :param zeroPadding:
    :param windowType:
    :param kernel:
    :param margin:
    :param power:
    :return:
    '''
    x = ess.MonoLoader(filename=nameInput, sampleRate=fs)()
    X = HPSS_essentia.compute_STFT(x, frameSize=frameSize, hopSize=hopSize, zeroPadding=zeroPadding,
                                   windowType=windowType)

    H_essentia, P_essentia = HPSS_essentia.HPSS(X, kernel=kernel, margin=margin, power=power)
    H_librosa, P_librosa = librosa.decompose.hpss(np.transpose(X), kernel_size=kernel, margin=margin, power=power)

    compare(H_essentia, np.transpose(H_librosa))
    compare(P_essentia, np.transpose(P_librosa))


if __name__ == '__main__':

    # Generate 'random' settings to debug
    power = [0.5, 1, 2, 10, np.inf]
    kernels = [(17), (11, 11), (13, 7), (41, 51)]
    margins = [(1), (1, 1), (1.5, 1.3)]

    for kernel in kernels:
        for margin in margins:
            main(kernel=kernel, margin=margin)
