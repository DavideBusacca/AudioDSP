import sys, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
import utils as U

def plotFilterFreqResponse(numerator, denominator=1, name=None):
    if name == None:
        name = 'Digital filter frequency response'
    else:
        name = name + ' frequency response'

    fig = plt.figure()
    plt.title(name)
    ax1 = fig.add_subplot(111)
    w, h = freqz(b=numerator, a=denominator)
    mX = U.amp2db(U.getMagnitude(h))
    plt.plot(w, mX, color='b')
    # if the difference in dB is less then 2 db we force to print a range of 2 dB (for flat freq. responses)
    if np.abs(np.max(mX) - np.min(mX)) < 2:
        tmp = np.min(mX) - 1
        plt.ylim(np.min(mX) - 1, np.max(mX) + 1)
    plt.ylabel('Amplitude [dB]', color='b')
    plt.xlabel('Frequency [rad/sample]')
    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(h))
    plt.plot(w, angles, 'g')
    plt.ylabel('Angle (radians)', color='g')
    plt.grid()
    plt.axis('tight')
    plt.show()

    return w, h
