#This scripts read all mat files into python, convert them into numpy array
#and then aplpy the transform.
#the result is then saved onto the disk for later use.

import scipy.io as sci
import numpy as np
import matplotlib.pyplot as plt
import pywt
import os
import sys
import re
from sklearn import preprocessing
from scipy import signal
from scipy.signal import resample, hann


def transformAll(externalHD = False):
  if externalHD:
    #External Hard drive
    directory = "ExternalHD"
    listOfScans = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
  else:
    #local only Dog5
    directory = "LocalHD"
    listOfScans = ['Dog_5']
  for scan in listOfScans:
    print scan
    readMatAndTransformInDirectory(directory + scan + "/") 
    

#Read matlab file and return numpy array
def readMatFile(filename):
  matfile = sci.loadmat(filename)
  listOfKeys = matfile.keys()
  listOfKeys.remove('__version__')
  listOfKeys.remove('__header__')
  listOfKeys.remove('__globals__')
  segment = listOfKeys[0]
  fileKind = re.match(r'[a-z]+', segment).group()
  if fileKind == "interictal": kind = 0
  elif fileKind == "preictal": kind = 1
  elif fileKind == "test": kind = 2 
  else: kind = -1
  matfileData = matfile[segment]
  packet = matfileData[0][0][0]
  duration = matfileData[0][0][1][0][0]
  frequency = matfileData[0][0][2][0][0]
  probeNames =[]
  for nameAsNumpy in matfileData[0][0][3][0]:
    probeNames.append(nameAsNumpy[0])
  #using generators
  yield packet 
  yield duration 
  yield frequency
  yield kind 


#Transforms
#Name: SquibDWTFFT
#There is wavelet + max peak fft frequency  for each squib
def getWavelet(data, w, level = 14):
  mode = pywt.MODES.sp1
  #w = 'coif5' #"DWT: Signal irregularity"
  #w = 'sym5'  #"DWT: Frequency and phase change - Symmlets5")
  w = pywt.Wavelet(w)
  a = data
  allc = []
  for i in xrange(level):
    (a, d) = pywt.dwt(a, w, mode)
    #print len(a)
    #print len(d)
  allc.extend(a.tolist())
  allc.extend(d.tolist())
  return allc

def getFFT(data, window_size):
  win=signal.hann(window_size)    
  yf = np.fft.fft(data)
  freqs = np.fft.fftfreq(len(data))
  #for coef,freq in zip(yf,freqs):
  #  if coef:
  #      print('{c:>6} * exp(2 pi i t * {f})'.format(c=coef,f=freq))
  #print(freqs.min(),freqs.max())
  # (-0.5, 0.499975)
  # Find the peak in the coefficients
  frate=11025.0
  yf=yf[1:]#be sure to remove 0hz
  idx=np.argmax(np.abs(yf)**2)
  freq=freqs[idx]
  freq_in_hertz=abs(freq*frate)
  #print(freq_in_hertz)
  return freq_in_hertz 

def getCrossCorrelation(packet):
  if type(packet) is eT.EEGpackage:
    return getCrossCorrelation(packet.packet)
  else:
    features = []
    #splitted packet
    spa = splitPacket(packet)
    for si in range(len(spa)):
      split = spa[si]
      n_channels = range(len(split))
      for ch1 in n_channels:
        for ch2 in n_channels:
          cor = np.correlate(split[ch1], split[ch2])
          #s = 'Corr: ' + repr(cor.item()) + ' of split ' + repr(si) + ' between channel ' + repr(ch1) + ' and channel ' + repr(ch2)
          #print s
          features.append(cor.item())
  return features

def SquibDWTFFT3(packet):
  features = []
  #number of sample in the n second window
  nchan, npts = packet.shape
  for squib in packet:
    #features.extend(getWavelet(squib, 'coif5', 10))
    features.extend(getWavelet(squib, 'sym5', 14))
    features.append(getFFT(squib, 399))
    features.append(squib.max())
    features.append(np.mean(squib))
    features.append(np.var(squib))
    features.append(np.std(squib))
    features.append(squib.min())
    for squib1 in packet:
      cross_corr = np.correlate(squib, squib1)
      features.append(cross_corr[0])
    #print len(features)
    #print "XXXXXXXX"
    #print len(features)
  return np.array(features)
#Kaggle score: 0.65569
      
def SquibDWTFFT(packet):
  features = []
  #number of sample in the n second window
  nchan, npts = packet.shape
  for squib in packet:
    #features.extend(getWavelet(squib, 'coif5', 18))
    features.extend(getWavelet(squib, 'sym5', 18))
    features.append(getFFT(squib, 399))
    features.append(squib.max())
    features.append(squib.mean())
    features.append(squib.std())
    features.append(squib.min())
    #print len(features)
    #print "XXXXXXXX"
    #print len(features)
  return np.array(features)
#Kaggle score: 0.65682 in BH framework


def readMatAndTransformInDirectory(directory): 
  files = os.listdir(directory)
  #Simple match
  #matfiles = filter(lambda x:'.mat' in x, files)
  #Simple reduced match - for test
  #matfiles = filter(lambda x:'_0004.mat' in x, files)
  #External taking care of macosx hidden files
  matfiles = filter((lambda x: re.search(r'^[PD].*\.mat', x)),files)
  test_countf = 0
  train_countf = 0
  for f in matfiles:
    fn = directory + f
    if os.path.exists(fn):
      print "process " + fn
      pac, dur, freq, kind = readMatFile(fn)
      #print "transform " + fn
      feat2 = np.append(SquibDWTFFT(pac), SquibDWTFFT3(pac))
      feat12 = np.append(kind, feat2)
      #dur, freq, kind
      if kind == 2:
        #Test
        if test_countf == 0:
          test_features = feat12
          test_countf = 1
        else:
          test_features = np.vstack((test_features, feat12))
          test_countf += 1
      else:
        #train
        if train_countf == 0:
          train_features = feat12
          train_countf = 1
        else:
          train_features = np.vstack((train_features, feat12))
          train_countf += 1
  if train_countf > 0:
    outf = directory + "transformed.train.SquibDWTFFT.npy"
    print "Writting : " + outf
    np.save(outf, train_features)
  if test_countf > 0:
    outf = directory + "transformed.test.SquibDWTFFT.npy"
    print "Writting : " + outf
    np.save(outf, test_features)
  return test_countf, train_countf
  
  
#to load npy use np.load()
