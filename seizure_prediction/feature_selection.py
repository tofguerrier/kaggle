import scipy.io as sci
import numpy as np
import matplotlib.pyplot as plt
import pywt
import time
import os
import sys
import re
from sklearn import preprocessing
from scipy import signal
from scipy.signal import resample, hann
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles

#Settinf seed
np.random.seed(44)


def fitRF(X,Y):
  clf = RandomForestClassifier(n_estimators=10)
  clf = clf.fit(X, Y)
  return clf


def loadTransformed(filepath):
  return np.load(filepath)

def splitData(data):
  y, x, n = np.hsplit(data,np.array([1, data.shape[1]]))
  print "Number of observations :" + str(data.shape[0])
  print "Number of features :" + str(data.shape[1])
  return (x, y)

#Removing features with low variance
#see http://scikit-learn.org/stable/modules/feature_selection.html
#for more details.
def removeLowVar(X, threshold = 0.8):
  sel = VarianceThreshold(threshold=(threshold * (1 - threshold)))
  return sel.fit_transform(X)


#RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, min_density=None, compute_importances=None)

##PCA
def runPCA(X,Y):
  from sklearn.decomposition import PCA
  pca = PCA(n_components=2)
  pca.fit(X)
  PCA(copy=True, n_components=2, whiten=False)
  print(pca.explained_variance_ratio_) 


def compareRF(filepath):
  data = loadTransformed(filepath)
  X,Y = splitData(data)
  y = Y.flatten()
  reducedX = removeLowVar(X)
  print "Decision Tree Classifier Default"
  clf = DecisionTreeClassifier(max_depth=None, min_samples_split=1,random_state=0)
  scores = cross_val_score(clf, X, y, cv =10)
  print scores.mean()                             
  print "Random Forest Classifier Default"
  clf = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=1, random_state=0)
  scores = cross_val_score(clf, X, y, cv = 10)
  print scores.mean()                             
  print "Extra Tree Random Forest Classifier Default"
  clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=1, random_state=0)
  scores = cross_val_score(clf, X, y, cv = 10)
  print scores.mean() 
  print "Using Reducded features"
  print "Decision Tree Classifier Default"
  clf = DecisionTreeClassifier(max_depth=None, min_samples_split=1,random_state=0)
  scores = cross_val_score(clf, reducedX, y, cv =10)
  print scores.mean()                             
  print "Random Forest Classifier Default"
  clf = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=1, random_state=0)
  scores = cross_val_score(clf, reducedX, y, cv = 10)
  print scores.mean()                             
  print "Extra Tree Random Forest Classifier Default"
  clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=1, random_state=0)
  scores = cross_val_score(clf, reducedX, y, cv = 10)
  print scores.mean() 
  print "Gini vs Entropy"
  clf = RandomForestClassifier(n_estimators=10,  criterion='gini',max_depth=None,min_samples_split=1, random_state=0)
  scores = cross_val_score(clf, X, y, cv = 10)
  print scores.mean()                             
  clf = RandomForestClassifier(n_estimators=10,  criterion='entropy',max_depth=None,min_samples_split=1, random_state=0)
  scores = cross_val_score(clf, X, y, cv = 10)
  print scores.mean()                             



def runRF(directory, name):
  print(time.time(), time.clock())
  test = directory + "transformed.test.SquibDWTFFT.npy"
  train = directory + "transformed.train.SquibDWTFFT.npy"
  data = loadTransformed(train)
  X,Y = splitData(data)
  y = Y.flatten()
  ocp = np.sum(y == 1)
  oci = np.sum(y == 0)
  ratio = oci.astype(float) / ocp.astype(float) 
  nest = int(300 * ratio.round())
  print ratio.round()
  threshold = 0.8
  sel = VarianceThreshold(threshold=(threshold * (1 - threshold)))
  Xr =  sel.fit_transform(X)
  #clf = RandomForestClassifier(n_estimators=10,  criterion='entropy',max_depth=None,min_samples_split=1, random_state=0)
  print "Fit Random forest on " + name
  clf = RandomForestClassifier(n_estimators=nest , criterion = "entropy", max_features="auto", min_samples_split=1, bootstrap=False, n_jobs=8, random_state=0)
  clf.fit(Xr, y, sample_weight= np.array([ratio.round() if i == 0 else 1 for i in y]))
  print(time.time(), time.clock())
  print "Predict " + name
  dtest = loadTransformed(test)
  XT,YT = splitData(dtest)
  yt = YT.flatten()
  XTr =  sel.transform(XT)
  predictions = clf.predict_proba(XTr)
  filename = name + "_predictions.csv"
  print(time.time(), time.clock())
  print "Writting out results ... "
  finalFile = open(filename, "w")
  count = 1
  for pre in predictions:
    pr = '%.6f' %  pre[1]
    count4 = "%04d" % (count,)
    print name + " " + count4 + " " + pr
    finalFile.write(name + "_test_segment_" + count4 + ".mat," + str(pr) + "\n")
    count += 1
  finalFile.close()
  print(time.time(), time.clock())


def runAll():
  runRF("../transformed/Dog_1/", "Dog_1")
  runRF("../transformed/Dog_2/", "Dog_2")
  runRF("../transformed/Dog_3/", "Dog_3")
  runRF("../transformed/Dog_4/", "Dog_4")
  runRF("../transformed/Dog_5/", "Dog_5")
  runRF("../transformed/Patient_1/", "Patient_1")
  runRF("../transformed/Patient_2/", "Patient_2")


