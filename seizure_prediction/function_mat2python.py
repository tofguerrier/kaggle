'''
Christophe Guerrier - 2014 September
Kaggle competition Seizure prediction
https://www.kaggle.com/c/seizure-prediction/data
'''

import scipy.io as sci
import numpy as np
import matplotlib.pyplot as plt
import os
import re


'''Function to load matlab file in python, return a numpy.array'''
def load_matfile(filename):
  print("Loading " + filename)
  mat = sci.loadmat(filename)
  segment = re.sub(r'.mat$', "", filename)
  segment = re.sub(r'Dog_\d_', "", segment)
  segment = re.sub(r'_0+', "_", segment)
  mnd = mat[segment]
  mnd_array = mnd.item()[0]
  mnd_arrayt = mnd_array.transpose()
  return mnd_arrayt

'''Function to plot current array'''
def plot_eeg(eeg_array):
  eeg_mean = eeg_array.mean()
  eeg_min = eeg_array.min()
  eeg_max = eeg_array.max()
  for electro in range(0,15,1):
    y = (electro * (eeg_max + 10)) + eeg_array[:,electro]
    num_points = len(y)
    x = range(0,num_points,1)
    plt.scatter(x,y)
  plt.ylabel('EEG')
  plt.show()

'''Function to export array to csv file'''
def export_mat_to_csv(nparray, filename):
  outf = re.sub("\.mat", ".csv", filename)
  print("Exporting : " + outf)
  np.savetxt(outf , nparray, delimiter=",")

'''Function to regexp filter in a list'''
def filterPick(list,filter):
  return [ ( l, m.group(1) ) for l in list for m in (filter(l),) if m]

'''Read all mat file in current directory and export to csv files'''
def export_all():
  #Get all files in current directory.
  searchRegex = re.compile('(.mat$)').searcyyh
  files = [f for f in os.listdir('.') if os.path.isfile(f)]
  matlabf = filterPick(files, searchRegex)
  for mf in matlabf:
    matarray = load_matfile(mf[0])
    #temporarily commented
    #export_mat_to_csv(matarray, mf[0])
    print(mf[0])



