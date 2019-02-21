import sys, os, copy
import subprocess
import numpy as np
from scipy.signal import hanning
from scipy.io.wavfile import write, read
from scipy.signal import resample as scipy_resample

INT16_FAC = (2**15)-1
INT32_FAC = (2**31)-1
INT64_FAC = (2**63)-1
norm_fact = {'int16':INT16_FAC, 'int32':INT32_FAC, 'int64':INT64_FAC,'float32':1.0,'float64':1.0}

class NewParam():
    #Class used only to encapsulate the parameters for the relative module
    pass

class NewModule():
    #Unit module for signal processing
    def __init__(self):
        self.clear()
        self.update()
        
    def getParam(self):
        pass
        
    def setParam(self):
        pass
        
    def update(self):
        pass

    def process(self):
        pass
        
    def clockProcess(self):
        pass
    
    def clear(self):
        pass    

def distance2signals(x, y, frameSize):
    clean = True
    length = min(x.size, y.size)
    if clean==True:
        start = frameSize
        end = length-frameSize
    else:
        start = 0
        end = length
    return sum(np.abs(x[start:end]-y[start:end]))

def myOverlapAdd(xw, windowSize, hopSize):
    x=np.zeros(np.int(((np.ceil((len(xw))/float(windowSize)))) * hopSize + windowSize))
    ch=0
    cw=0
    for i in np.arange(np.floor(len(xw))/float(windowSize)-1):
        x[ch:ch+windowSize]=x[ch:ch+windowSize]+xw[cw:cw+windowSize]
        ch=ch+hopSize #current hop
        cw=cw+windowSize #current window
    
    return x

def windowing(x, frameSize, typeWindow='hann'):
    if typeWindow == 'hann':
        return x*hanning(frameSize)
        
def zeroPhasing(x, frameSize, zeroPadding, fftshift=True, inverse=False):
    '''
    fftShift
    N is frameSize+zeroPadding
    '''      
    if inverse == False:
        if fftshift == True:
            N = frameSize+zeroPadding
            h1 = int(np.floor((x.size+1)/2))
            h2 = int(np.floor(x.size/2))
            fftbuffer = np.zeros(N)
            fftbuffer[:h1] = x[h2:]
            fftbuffer[-h2:] = x[:h2]
        else:
            fftbuffer = np.append(x, np.zeros(zeroPadding))
    if inverse == True:        
        N = frameSize+zeroPadding
        h1 = int(np.floor((x.size+1)/2))
        h2 = int(np.floor(x.size/2))
        fftbuffer = np.zeros(N)
        fftbuffer[:h1] = x[h2:]
        fftbuffer[-h2:] = x[:h2]
        if zeroPadding != 0:  
            #deleting zeros
            z1 = int(np.floor((zeroPadding+1)/2))
            z2 = int(np.floor(zeroPadding)/2)
            fftbuffer = fftbuffer[z1:-z2] #attention, when z2 is 0 the array is empty(solved using the if)
    
    return fftbuffer  
    
def amp2db(amp):
    return 20*np.log10(amp+sys.float_info.epsilon)

def db2amp(db):
    return 10**(db/20)
    
def getMagnitude(X):
    return np.abs(X)
    
def getPhase(X):
    #add phase wrapping!
    return np.angle(X)
    
def getSpectrogramAxis(X, fs, hopSize):
    t = np.arange(X.shape[0])*hopSize/float(fs)  
    f = np.arange(X.shape[1]/2+1)*float(fs/2)/float(X.shape[1]/2+1) 
    return t, f  
    
def getTimeAxis(x, fs):
    return np.arange(len(x))/float(fs)

def wavplay(filename):
    #Adapted from sms-tools

	"""
	Play a wav audio file from system using OS calls
	filename: name of file to read
	"""
	if (os.path.isfile(filename) == False):                  # raise error if wrong input file
		print("The file to be played does not exist.")
	else:
		if sys.platform == "linux" or sys.platform == "linux2":
		    # linux
		    subprocess.call(["aplay", filename])

		elif sys.platform == "darwin":
			# OS X
			subprocess.call(["afplay", filename])
		elif sys.platform == "win32":
			if winsound_imported:
				winsound.PlaySound(filename, winsound.SND_FILENAME)
			else:
				print("Cannot play sound, winsound could not be imported")
		else:
			print("Platform not recognized")    
			
def wavread(filename):
    #Adapted from sms-tools

	"""
	Read a sound file and convert it to a normalized floating point array
	filename: name of file to read
	returns fs: sampling rate of file, x: floating point array
	"""

	if (os.path.isfile(filename) == False):
		raise ValueError("Input file is wrong")

	fs, x = read(filename)
	x = np.float32(x)/norm_fact[x.dtype.name]
	
	return x, fs
	
def wavwrite(y, fs, filename):
    #Adapted from sms-tools

	"""
	Write a sound file from an array with the sound and the sampling rate
	y: floating point array of one dimension, fs: sampling rate
	filename: name of file to create
	"""

	x = copy.deepcopy(y)                         # copy array
	x *= INT16_FAC                               # scaling to int16 range
	x = np.int16(x)                              # converting to int16 type
	write(filename, fs, x)

def peakFollower(x, fs, releaseTime):
    release_factor = np.exp(-1/(releaseTime * fs))
    x = abs(x)
    envelope = np.zeros_like(x)
    for id in range(np.size(x)-1): 
        i=id+1
        envelope[i] = max(x[i], envelope[i-1]*release_factor )
        
    return envelope

def resample(x, sr_old, sr_new):
    x_duration = len(x) / sr_old                         # Number of seconds in signal X
    target_samples = int(x_duration * sr_new)            # Samples number after resampling
    return scipy_resample(x, target_samples)

def rms(x):
    return np.sqrt((np.mean(np.power(x, 2))))