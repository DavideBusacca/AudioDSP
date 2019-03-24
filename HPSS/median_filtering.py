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

def compute_hard_mask(X, X_ref):
    # Hard masking for Harmonic/Percussive Source Separation
    return X > X_ref

def compute_soft_mask(X, X_ref, power=2, split_zeros=False):

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
    # Soft masking for Harmonic/Percussive Source Separation
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
    return mX - (mH + mP)

def compute_enhanced_spectrograms(mX, win_harm=17, win_perc=17, test_mode=False):
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
    mH = np.empty_like(mX)
    mP = np.empty_like(mX)

    if not(test_mode): # test mode is used to match the results of the block-based version
        mH[:] = median_filter(mX, size=(win_harm, 1), mode='reflect') # use mode='wrap' to get the same results of the other version
        mP[:] = median_filter(mX, size=(1, win_perc), mode='reflect') # use mode='constant' to get the same results of the other version
    else:
        mH[:] = median_filter(mX, size=(win_harm, 1), mode='wrap')
        mP[:] = median_filter(mX, size=(1, win_perc), mode='constant')

    return mH, mP

def compute_enhanced_spectrograms_block_based(mX, win_harm=17, win_perc=17):
    row = mX.shape[0]
    col = mX.shape[1]

    shift = (int(np.floor(win_harm / 2))) # middle of the buffer

    buffer = np.transpose(mX)[:col, 0:win_harm]
    mP = np.empty_like(mX)
    mH = np.empty_like(mX)

    for r in range(row):
        mP[r] = medfilt(buffer[:, shift], win_perc) # time position centered in the middle of the buffer
        mH[r] = np.median(buffer, axis=1)

        # updating the buffer: shifting to the left and insert new element at the end
        buffer = np.roll(buffer, -1, axis=1)
        buffer[:, win_harm-1] = mX[(r+win_harm) % row]

    # time shifting: per[r] could be substituted with per[(r+shift) % row] to get the same result!
    mH = np.roll(mH, shift, 0)
    mP = np.roll(mP, shift, 0)

    return mH, mP

def main():
    # Function to test if the two functions return the same.
    # To get the same the mode of the median_filter of 'computing_enhanced_spectrograms()' need to be changed.
    test_matrix = np.random.rand(36, 34)

    win_harm = 17
    win_perc = 17

    time_0 = time.time()
    mH, mP = compute_enhanced_spectrograms(test_matrix, win_harm=win_harm, win_perc=win_perc, \
                                           test_mode=True)
    time_1 = time.time()
    mH_t, mP_t = compute_enhanced_spectrograms_block_based(test_matrix, win_harm=win_harm, win_perc=win_perc)
    time_2 = time.time()

    print("Time profiling 1st method: " + str(time_1 - time_0))
    print("Time profiling 2nd method: " + str(time_2 - time_1))

    # Original matrix
    plt.pcolormesh(test_matrix)
    # Difference matrix between the two methods for each enhanced spectrogram
    plt.figure()
    plt.pcolormesh(mH-mH_t)
    plt.figure()
    plt.pcolormesh(abs(mP-mP_t))

    plt.show()

if __name__ == '__main__':
    main()