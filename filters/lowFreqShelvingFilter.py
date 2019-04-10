import numpy as np
from AudioDSP.filters import utilsFilter as UF

def BandPass_AllPassFilter(x, Wc, Wb):
    '''
    BandPass filter based upon an All Pass filter
    Adaptation from DAFX book, 2nd edition

    x: input signal
    Wc: is the normalized cut - off frequency 0 < Wc < 1, i.e. (2 * fc / fS)
    Wb: is the normalized bandwidth
    '''

    c = (np.tan(np.pi*Wb/2) - 1) / (np.tan(np.pi*Wb/2) + 1)
    d = -np.cos(np.pi * Wc)

    xh = [0, 0]
    y = np.zeros_like(x)
    for n in np.arange(len(x)):
        xh_new = x[n] - d * (1 - c) * xh[1] + c * xh[2]
        ap_y = -c * xh_new + d * (1 - c) * xh[1] + xh[2]
        xh = [xh_new, xh[1]]
        y[n] = ap_y # all pass should be like this
        #y[n] = 0.5 * (x[n] - ap_y) # change to plus for bandreject

def lowShelving_allPassFilter(x, Wc, G):
    '''
    Low Shelving filter based on a All Pass filter and a summation with the old input processed
    Adaptation from DAFX book, 2nd edition

    x: input signal
    Wc: is the normalized cut - off frequency 0 < Wc < 1, i.e. (2 * fc / fS)
    G: gain in dB

    '''
    V0 = 10 ^ (G / 20)
    H0 = V0 - 1
    if G >= 0:
        c = (np.tan(np.pi*Wc/2) - 1) / (np.tan(np.pi*Wc/2) + 1) # boost
    else:
        c = (np.tan(np.pi*Wc/2) - V0) / (np.tan(np.pi*Wc/2) + V0) # cut

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
    
def main(Gain=0.25, freqCutoff=11025, fs=44100):
    wc = 2 * np.pi * freqCutoff / fs

    b, a = lowFreqShelvingFilter_1stOrder(Gain, wc)
    UF.plotFilterFreqResponse(b, a, name="Low frequency shelving filter: ")


if __name__ == '__main__':
    main()