#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import globals
import csv
import numpy as np
import math
from scipy import stats
from sklearn.svm import *
from sklearn.ensemble import *
from sklearn.metrics import *
from sklearn.cross_validation import *
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import *
from sklearn.hmm import *
from sklearn.lda import LDA
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import pylab as pl
#from knn import *
def get_mean_ci(data,confidence=0.95):
	mean = np.mean(data)
	std = np.std(data)
	R = stats.norm.interval(confidence,loc=mean,scale=std/math.sqrt(len(data)))
	ci = R[1]-mean 
	return mean, ci


def single_modality_classification(modality_type):
	"""Classify data based on data's modality

	:param number_of_folds: int. Number of folds for crossvalidation
	:param modality_type: string. Data's modality, namely 'a' for audio and 'v' for video
	"""
	#path = '/home/samuel/Dropbox/Dissertacao/repo/samples/smalldataset/'

	globals.path_init('geometry')

	X = []
	Y = []
	mean_acc = []
	std = []

	with open(globals.path+modality_type+'_features.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='|')
		for row in reader:
			X.append([float(x) for x in row[:-1]])
			Y.append(int(row[-1]))

	with open(globals.kitchen_path+'set1/'+modality_type+'_features.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='|')
		for row in reader:
			X.append([float(x) for x in row[:-1]])
			Y.append(17)

	# if modality_type is 'v':# or modality_type is 'av':
	# 	transformer = TfidfTransformer()
	# 	X = transformer.fit_transform(X).todense().tolist()
		#X = transformer.toarray()
	#E = np.random.uniform(0, 0.1, size=(len(X), 20))

	# Add the noisy data to the informative features

	# X = np.array(X)
	# X_indices = np.arange(X.shape[-1])
	# selector = SelectPercentile(f_classif, percentile=10)
	# selector.fit(X, Y)
	# scores = -np.log10(selector.pvalues_)
	# scores /= scores.max()
	# pl.clf()
	# pl.bar(X_indices - .45, scores, width=.2, color='g')
	# pl.ylabel(r'Escore univariado ($-Log(p_{value})$)')
	# pl.xlabel('Numero da feature')
	# pl.title('Discriminancia das features - ' + modality_type)
	# pl.show()
	# print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
	# print len(scores)
	Y_ground_truth = []
	Y_prob = []


	for classifier in ['RandomForest']:
		 #NaiveBayes', 'DecisionTree', 'LogisticRegression', 'LDA', 'Adaboost', 'GradientBoosting', 'RandomForest', 'ANN', 'SVM', 'KNN']:
		print "Training %s" % modality_type
		
		Y_ground_truth = []
		Y_prob = []

		x_train = X
		y_train = Y	
			
		clf = RandomForestClassifier(n_estimators=100)
		clf_roc = OneVsRestClassifier(clf)

		x_test = []
		y_test = []

		with open(globals.kitchen_path+modality_type+'_features.csv', 'rb') as csvfile:
			reader = csv.reader(csvfile, delimiter=',', quotechar='|')
			for row in reader:
				x_test.append([float(x) for x in row[:-1]])
				y_test.append(17)

		x_train = np.array(x_train)
		x_test = np.array(x_test)

		y_train = label_binarize(y_train, classes=range(17))
		y_test = label_binarize(y_test, classes=range(17))
		probas_ = clf_roc.fit(x_train, y_train).predict_proba(x_test)

		# roc_auc = auc(fpr, tpr)
		# plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
		Y_ground_truth.extend(y_test)
		Y_prob.extend(probas_)

		# fpr = dict()
		# tpr = dict()
		# roc_auc = dict()
		# for i in range(16):
		#     fpr[i], tpr[i], _ = roc_curve(Y_ground_truth[:, i], Y_prob[:, i])
		#     roc_auc[i] = auc(fpr[i], tpr[i])
		Y_ground_truth = np.array(Y_ground_truth)
		Y_ground_truth = np.ravel(Y_ground_truth)

		Y_prob = np.array(Y_prob)
		Y_prob = np.ravel(Y_prob)

	return roc_curve(Y_ground_truth, Y_prob)