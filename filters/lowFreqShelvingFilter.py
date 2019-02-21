import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz

def lowShelving(x, Wc, G):
    '''
    Adaptation from DAFX book, 2nd edition

    x: input signal
    Wc: is the normalized cut - off frequency 0 < Wc < 1, i.e. (2 * fc / fS)
    G: gain in dB

    '''
    V0 = 10 ^ (G / 20)
    H0 = V0 - 1
    if G >= 0:
        c = (np.tan(Wc/2)-1) / (np.tan(Wc/2)+1) # boost
    else:
        c = (np.tan(Wc/2)-V0) / (np.tan(Wc/2)+V0) # cut

    xh = 0
    y = np.zeros_like(x)

    for n in np.arange(len(x)):
        xh_new = x[n] - c * xh
        ap_y = c * xh_new + xh
        xh = xh_new
        y[n] = 0.5 * H0 * (x[n] + ap_y) + x[n] # change to minus for HS

    return y



def lowFreqShelvingFilter_1stOrder(Gain, wc):
    '''
    First order low-frequency shelving filter

    INPUTS
    Gain = Gain at low frequencies (linear, not dB)
    wc = crossover frequency
    OUTPUTS
    numerator = numerator coefficients b0 and b1
    denominator = denominator coefficients a0 and a1
    '''
    
    '''
    TANGENT
    A tangent is a straight line which passes through a point in a curve with slope equal to the derivative of the curve in that point
    tan(alfa) = sin(alfa)/cos(alfa) 
    '''
    
    # Transfer function coefficients
    a0 = np.tan(wc/2) + np.sqrt(Gain)
    a1 = (np.tan(wc/2) - np.sqrt(Gain))
    b0 = (Gain * np.tan(wc/2) + np.sqrt(Gain))
    b1 = (Gain * np.tan(wc/2) - np.sqrt(Gain))

    # Transfer function polynomials
    a = [a0, a1]    # denominator
    b = [b0, b1]    # numerator
        
    return b, a
    
def plotFilterFreqResponse(numerator, denominator=1):
    fig = plt.figure()
    plt.title('Digital filter frequency response')
    ax1 = fig.add_subplot(111)
    w, h = freqz(b=numerator, a=denominator)
    plt.plot(w, 20 * np.log10(np.abs(h)), color='b')
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
    
# implement filter
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html


def main(Gain=0.25, freqCutoff=1000, fs=44100):
    wc = 2 * np.pi * freqCutoff / fs
    b, a = lowFreqShelvingFilter_1stOrder(Gain, wc)

    plotFilterFreqResponse(b, a)

if __name__ == '__main__':
    main()