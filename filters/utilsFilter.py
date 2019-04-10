import sys, os
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from scipy.signal import freqz
from scipy.signal import tf2zpk as scipy_tf2zpk
from AudioDSP import utils as U

def plotFilterFreqResponse(numerator, denominator=1, name=None, subplot=None, show=True):
    if name == None:
        name = 'Digital filter frequency response'
    else:
        name = name + ' frequency response'

    if subplot is None:
        fig = plt.figure()
        plt.title(name)
        subplot = fig.add_subplot(111)
    ax2 = subplot.twinx()

    w, h = freqz(b=numerator, a=denominator)
    mX = U.amp2db(U.getMagnitude(h))
    subplot.plot(w, mX, color='b')
    if np.abs(np.max(mX) - np.min(mX)) < 2: # if the difference in dB is less then 2 db we force the dB range axes
        subplot.set_ylim(np.min(mX) - 1, np.max(mX) + 1)
    subplot.set_ylabel('Amplitude [dB]', color='b')
    subplot.set_xlabel('Frequency [rad/sample]')
    angles = np.unwrap(np.angle(h))
    ax2.plot(w, angles, 'g')
    ax2.set_ylabel('Angle (radians)', color='g')
    ax2.grid()
    ax2.axis('tight')

    if show:
        plt.show()

    return w, h

def normalizeCoefficients(b,a):
    if np.max(b) > 1:
        kn = np.max(b)
        b = b / float(kn)
    else:
        kn = 1

    if np.max(a) > 1:
        kd = np.max(a)
        a = a / float(kd)
    else:
        kd = 1

    return b, a, kn, kd

def tf2zpk(b, a):
    # Get poles and zeros
    return scipy_tf2zpk(b, a)
