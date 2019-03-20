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
from scipy.ndimage import median_filter
from scipy.signal import medfilt
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import time

'''
Two versions of the median filters for computing the percussive/harmonic enhanced spectrograms for HPSS.
The second version is supposed to be a prototype for a block-to-block- real-time implementation.
Consider that the median filter in time (harmonic) should be centered in the 'real-time' frame.
'''

def computing_enhanced_spectrograms(mX, win_harm=17, win_perc=17, test_mode=False):
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
    # Computing harmonic and percussive enhanced spectrograms
    harm = np.empty_like(mX)
    perc = np.empty_like(mX)

    if not(test_mode): # test mode is used to match the results of the block-based version
        harm[:] = median_filter(mX, size=(win_harm, 1), mode='reflect') # use mode='wrap' to get the same results of the other version
        perc[:] = median_filter(mX[:], size=(1, win_perc), mode='reflect') # use mode='constant' to get the same results of the other version
    else:
        harm[:] = median_filter(mX, size=(win_harm, 1), mode='wrap')
        perc[:] = median_filter(mX[:], size=(1, win_perc), mode='constant')

    return harm, perc

def computing_enhanced_spectrograms_block_based(mX, win_harm=17, win_perc=17):
    row = mX.shape[0]
    col = mX.shape[1]

    shift = (int(np.floor(win_harm / 2))) # middle of the buffer

    buffer = np.transpose(mX)[:col, 0:win_harm]
    per = np.empty_like(mX)
    har = np.empty_like(mX)

    for r in range(row):
        per[r] = medfilt(buffer[:, shift], win_perc) # time position centered in the middle of the buffer
        har[r] = np.median(buffer, axis=1)

        # updating the buffer: shifting to the left and insert new element at the end
        buffer = np.roll(buffer, -1, axis=1)
        buffer[:, win_harm-1] = mX[(r+win_harm) % row]

    # time shifting: per[r] could be substituted with per[(r+shift) % row] to get the same result!
    har = np.roll(har, shift, 0)
    per = np.roll(per, shift, 0)

    return har, per

def main():
    # Function to test if the two functions return the same.
    # To get the same the mode of the median_filter of 'computing_enhanced_spectrograms()' need to be changed.
    test_matrix = np.random.rand(36, 34)

    win_harm = 17
    win_perc = 17

    time_0 = time.time()
    harm, perc = computing_enhanced_spectrograms(test_matrix, win_harm=win_harm, win_perc=win_perc, \
                                                 test_mode=True)
    time_1 = time.time()
    har, per = computing_enhanced_spectrograms_block_based(test_matrix, win_harm=win_harm, win_perc=win_perc)
    time_2 = time.time()

    print("Time profiling 1st method: " + str(time_1 - time_0))
    print("Time profiling 2nd method: " + str(time_2 - time_1))

    # Original matrix
    plt.pcolormesh(test_matrix)
    # Difference matrix between the two methods for each enhanced spectrogram
    plt.figure()
    plt.pcolormesh(harm-har)
    plt.figure()
    plt.pcolormesh(abs(perc-per))

    plt.show()

if __name__ == '__main__':
    main()