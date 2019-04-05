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

def compute_enhanced_spectrograms(mX, win_harm=17, win_perc=17, test_mode=False):
    '''
    Median Filtering of a magnitude spectrogram contained in a 2D matrix.

    :param mX: input magnitude spectrogram
    :param win_harm: number of bins used for median filtering in frequency direction
    :param win_perc: number of bins used for median filtering in time direction
    :param test_mode: set TRUE if you want to match the results of the block-based approach
    :return:
    mH: harmonic magnitude spectrogram
    mP: percussive magnitude spectrogram
    '''

    # Computing harmonic and percussive enhanced spectrograms
    mH = np.empty_like(mX)
    mP = np.empty_like(mX)

    if not(test_mode):
        mH[:] = median_filter(mX, size=(win_harm, 1), mode='reflect')
        mP[:] = median_filter(mX, size=(1, win_perc), mode='reflect')
    else:
        # Set a different behaviour on the edge of the spectrogram using the parameter mode.
        # This configuration matches the output of compute_enhanced_spectrograms_block_based.
        mH[:] = median_filter(mX, size=(win_harm, 1), mode='wrap')
        mP[:] = median_filter(mX, size=(1, win_perc), mode='constant')

    return mH, mP

def compute_enhanced_spectrograms_block_based(mX, win_harm=17, win_perc=17):
    '''
    Median Filtering of a magnitude spectrogram using a block-to-block approach.

    :param mX: input magnitude spectrogram
    :param win_harm: number of bins used for median filtering in frequency direction
    :param win_perc: number of bins used for median filtering in time direction
    :return mH: harmonic magnitude spectrogram
    :return mP: percussive magnitude spectrogram
    '''

    blocks = mX.shape[0]
    bins = mX.shape[1]

    shift = (int(np.floor(win_harm / 2)))  # middle of the buffer

    buffer = np.transpose(mX)[:bins, 0:win_harm]
    mP = np.empty_like(mX)
    mH = np.empty_like(mX)

    for b in range(blocks):
        mP[b] = medfilt(buffer[:, shift], win_perc)  # time position centered in the middle of the buffer
        mH[b] = np.median(buffer, axis=1)

        # updating the buffer: shifting to the left and insert new element at the end
        buffer = np.roll(buffer, -1, axis=1)
        buffer[:, win_harm-1] = mX[(b+win_harm) % blocks]

    # time shifting: center the output of the median filter with the middle of the buffer
    # NB: mP[r] could be substituted with per[(b+shift) % blocks] to get the same result avoiding the final shifting!
    mH = np.roll(mH, shift, 0)
    mP = np.roll(mP, shift, 0)

    return mH, mP

def main():
    '''
    Comparing the results of the median filtering obtained using the 2D vs the block-by-block approaches.

    :return:
    '''

    test_matrix = np.random.rand(36, 34)

    win_harm = 17
    win_perc = 17

    time_0 = time.time()
    # test_mode need to be set True to match the results on the edges of the spectrograms. See the function.
    mH, mP = compute_enhanced_spectrograms(test_matrix, win_harm=win_harm, win_perc=win_perc, \
                                           test_mode=True)
    time_1 = time.time()
    mH_t, mP_t = compute_enhanced_spectrograms_block_based(test_matrix, win_harm=win_harm, win_perc=win_perc)
    time_2 = time.time()

    print("Time profiling 2D approach: " + str(time_1 - time_0))
    print("Time profiling block-by-block approach: " + str(time_2 - time_1))

    # Original matrix
    plt.pcolormesh(test_matrix)
    plt.title('Original Matrix')
    # Differences matrix between the two approaches for each enhanced spectrogram
    plt.figure()
    plt.pcolormesh(np.abs(mH-mH_t))
    plt.title('Differences between harmonic enhanced spectrograms')
    plt.figure()
    plt.pcolormesh(np.abs(mP-mP_t))
    plt.title('Differences between percussive enhanced spectrograms')

    print('Sum of the differences between harmonic enhanced spectrograms: ' + str(np.sum((np.abs(mH-mH_t)))))
    print('Sum of the differences between percussive enhanced spectrograms: ' + str(np.sum((np.abs(mP - mP_t)))))

    plt.show()

if __name__ == '__main__':
    main()

 # Functions used to compute the soft/hard masking in the remainder of the script (could be moved to a different file).

def compute_hard_mask(X, X_ref):
    '''
    Hard masking for Harmonic/Percussive Source Separation

    :param X:
    :param X_ref:
    :return:
    '''
    return X > X_ref

def compute_soft_mask(X, X_ref, power=2, split_zeros=False):
    '''
    Soft masking for Harmonic/Percussive Source Separation

    :param X:
    :param X_ref:
    :param power:
    :param split_zeros:
    :return:
    '''

    # Preparing rescaling and looking for invalid indexes
    Z = np.maximum(X, X_ref).astype(np.float32)
    bad_idx = (Z < np.finfo(np.float32).tiny)
    good_idx = ~bad_idx
    Z[bad_idx] = 1

    # Rescaling and power application
    X = (X / Z) ** power
    X_ref = (X_ref / Z) ** power

    # Computing mask
    mask = np.empty_like(X)
    mask[good_idx] = X[good_idx] / (X[good_idx] + X_ref[good_idx])

    bad_value = 0
    if split_zeros:
        bad_value = 0.5
    mask[bad_idx] = bad_value

    return mask


def compute_masks(mH, mP, power=2.0, margin_harm=1, margin_perc=1, masking='hard'):
    '''
    Call the masking function for Harmonic/Percussive Source Separation

    :param mH:
    :param mP:
    :param power:
    :param margin_harm:
    :param margin_perc:
    :param masking:
    :return:
    '''
    if margin_harm < 1 or margin_perc < 1:
        print("Attention: both margins should be major than 1")
        margin_harm = margin_perc = 1

    if np.isfinite(power):
        split_zeros = (margin_harm == 1 and margin_perc == 1)
        mask_harm = compute_soft_mask(mH, mP * margin_harm, power=power, split_zeros=split_zeros)
        mask_perc = compute_soft_mask(mP, mH * margin_perc, power=power, split_zeros=split_zeros)
    else:
        mask_harm = compute_hard_mask(mH, mP * margin_harm)
        mask_perc = compute_hard_mask(mP, mH * margin_perc)

    return mask_harm, mask_perc

def compute_residual_spectrogram(mX, mH, mP):
    '''
    Get the residual component, as difference between the original spectrogram and the sum of the harmonic and
    percussive components.

    :param mX:
    :param mH:
    :param mP:
    :return:
    '''

    return mX - (mH + mP)
