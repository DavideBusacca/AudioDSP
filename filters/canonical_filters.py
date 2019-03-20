import sys, os
import numpy as np
import utilsFilter as UF
import zplane as Z
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../visualization'))
import visualization as V

def CanonicalFilter_FirstOrder(K, type):
    '''
    Computing the denominator/nominator coefficients used in canonical filters of the first order.
    '''

    # The denominator contains the same coefficients independently from the type
    a0 = 1
    a1 = (K - 1) / (K + 1)

    if type == 'lowpass':
        b0 = K / (K + 1)
        b1 = b0

    if type == 'highpass':
        b0 = 1 / (K + 1)
        b1 = - b0

    if type == 'allpass':
        # note that b0 and a1 are the same in this case! (in another script this was called c)
        b0 = a1
        b1 = 1

    # Transfer function polynomials
    a = [a0, a1]  # denominator
    b = [b0, b1]  # numerator

    return b, a

def CanonicalFilter_SecondOrder(K, Q, type):
    '''
    Computing the denominator/nominator coefficients used in canonical filters of the first order.
    '''

    K_2 = np.power(K, 2)
    den = K_2*Q + K + Q # most of the coefficients share the same denominator

    # The denominator contains the same coefficients independently from the type
    a0 = 1
    a1 = 2 * Q * (K_2 - 1) / den
    a2 = (K_2 - K + Q) / den

    if type == 'lowpass':
        b0 = K_2 * Q / den
        b1 = 2*K_2 * Q / den
        b2 = b0

    if type == 'highpass':
        b0 = Q / den
        b1 = (-2*Q) / den
        b2 = b0

    if type == 'bandpass':
        b0 = K / den
        b1 = 0
        b2 = - b0

    if type == 'bandreject':
        b0 = Q * (1 + K_2) / den
        b1 = 2*Q * (K_2 - 1) / den
        b2 = b0

    if type == 'allpass':
        b0 = (K_2*Q - K + Q) / den
        b1 = 2*Q * (K_2 - 1) / den
        b2 = 1

    # Transfer function polynomials
    a = [a0, a1, a2]  # denominator
    b = [b0, b1, b2]  # numerator

    return b, a

def CanonicalFilter(Wc, type='allpass', order=1, Q=None):
    '''
    Canonical filter

    Wc: is the normalized cut - off frequency 0 < Wc < 1, i.e. (2 * fc / fS)
    '''

    #check type in : {'lowpass', 'highpass', 'allpass', < {'bandpass', 'bandreject'} allowed if order is 2 > }
    #check order in: {1, 2}

    K = np.tan(np.pi * Wc / 2)

    if order == 1:
        b, a = CanonicalFilter_FirstOrder(K, type)
    elif order == 2:
        if Q == None:
            Q = 1/np.sqrt(2)
        b, a = CanonicalFilter_SecondOrder(K, Q, type)

    return b, a

def main(Gain=0.25, freqCutoff=11025, fs=44100):
    wc = 2 * np.pi * freqCutoff / fs

    # b, a = lowFreqShelvingFilter_1stOrder(Gain, wc)
    # plotFilterFreqResponse(b, a)

    types = ['lowpass', 'highpass', 'allpass', 'bandpass', 'bandreject']

    wc = wc / np.pi
    for i in range(len(types)-2):
        b, a = CanonicalFilter(wc, type=types[i], order=1)
        name = 'Canonical ' + types[i] + ' 1st order filter'
        fig = V.createFigure(title=name)
        UF.plotFilterFreqResponse(b, a, name, subplot=fig.add_subplot(1, 2, 1), show=False)
        z, p, k = UF.tf2zpk(b, a)
        Z.plot_zplane(z, p, k, subplot=fig.add_subplot(1, 2, 2))

    for i in range(len(types)):
        b, a = CanonicalFilter(wc/np.pi, type=types[i], order=2)
        name = 'Canonical ' + types[i] + ' 2nd order filter'
        fig = V.createFigure(title=name)
        UF.plotFilterFreqResponse(b, a, name, subplot=fig.add_subplot(1, 2, 1), show=False)
        z, p, k = UF.tf2zpk(b, a)
        Z.plot_zplane(z, p, k, subplot=fig.add_subplot(1, 2, 2))

    V.show()


if __name__ == '__main__':
    main()