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
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import pylab as pl
#from knn import *

def get_mean_ci(data,confidence=0.95):
	mean = np.mean(data)
	std = np.std(data)
	R = stats.norm.interval(confidence,loc=mean,scale=std/math.sqrt(len(data)))
	ci = R[1]-mean 
	return mean, ci

#MAV
def meta_classification(a_k, v_k):
	"""Classify data based on data's modality

	:param number_of_folds: int. Number of folds for crossvalidation
	:param modality_type: string. Data's modality, namely 'a' for audio and 'v' for video
	"""
	#path = '/home/samuel/Dropbox/Dissertacao/repo/samples/smalldataset/'

	X = []
	Y = []
	mean_acc = []
	std = []

	a_count = 0
	v_count = 0
	av_count = 0

	with open(globals.path+'av_features.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='|')
		for row in reader:
			X.append([float(x) for x in row[:-1]])
			Y.append(int(row[-1]))

	# if modality_type is 'v' or modality_type is 'av':
	# 	transformer = TfidfTransformer()
	# 	X = transformer.fit_transform(X).todense().tolist()
		#print X[0,:]
		# print X
		# X = transformer.toarray()
		#print X

	Y_ground_truth = []
	Y_predicted = []
	Y_prob = []

	for classifier in ['RandomForest']:
		 #NaiveBayes', 'DecisionTree', 'LogisticRegression', 'LDA', 'Adaboost', 'GradientBoosting', 'RandomForest', 'ANN', 'SVM', 'KNN']:

		a_count = 0
		v_count = 0
		av_count = 0
		
		acc = []
		precision = []
		f1 = []

		Y_ground_truth = []
		Y_predicted = []
		Y_prob = []
		
		kf = KFold(len(X), globals.number_of_folds)
		for train_index, test_index in kf:
				
			# KFold split		
			x_train = np.zeros( shape = ( len(train_index) , len(X[0]) ))
			y_train = np.zeros( shape = ( len(train_index) ))
			for i in range(len(train_index)):
				for j in range(len(X[0])):
					x_train[i][j] = X[train_index[i]][j]
				y_train[i] = Y[train_index[i]]
			x_test = np.zeros( shape = ( len(test_index) , len(X[0]) ))
			y_test = np.zeros( shape = ( len(test_index) ))
			for i in range(len(test_index)):
				for j in range(len(X[0])):
					x_test[i][j] = X[test_index[i]][j]
				y_test[i] = Y[test_index[i]]

			#subsetSize = 0.6

			#x_train, unusedX, y_train, unusedY = train_test_split(x_train, y_train, train_size=subsetSize, random_state=1)

			clf = None
			
			a_train = []
			v_train = []
			for s in x_train:
				a_train.append(s[:-v_k])
				v_train.append(s[-v_k:])

			a_clf = None
			v_clf = None
			if classifier == 'NaiveBayes':
				a_clf = GaussianNB().fit(a_train,y_train)
				v_clf = GaussianNB().fit(v_train,y_train)
			elif classifier == 'SVM':
				a_clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0, kernel='linear', max_iter=-1, probability=True, random_state=None,shrinking=True, tol=0.001, verbose=False).fit(a_train,y_train)
				v_clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0, kernel='linear', max_iter=-1, probability=True, random_state=None,shrinking=True, tol=0.001, verbose=False).fit(v_train,y_train)
			elif classifier == 'RandomForest':
				a_clf = RandomForestClassifier(n_estimators=100).fit(a_train,y_train)
				v_clf = RandomForestClassifier(n_estimators=100).fit(v_train,y_train)
			else:
				pass

			#Concatenate probabilities from modalities to form the feature vector
			x_train = a_clf.predict_proba(a_train)
			aux = v_clf.predict_proba(v_train)

			x_train = x_train.tolist()
			for i in range(len(x_train)):
				x_train[i].extend(aux[i])
		

			a_test = []
			v_test = []
			for s in x_test:
				a_test.append(s[:-v_k]) #first a_k elements from audio stream
				v_test.append(s[-v_k:])

			a_pred = a_clf.predict(a_test)
			v_pred = v_clf.predict(v_test)

			for i in range(len(y_test)):
				if a_pred[i] == y_test[i]:
					a_count +=1
				if v_pred[i] == y_test[i]:
					v_count +=1
				if a_pred[i] == y_test[i] and v_pred[i] == y_test[i]:
					av_count +=1

			x_test = a_clf.predict_proba(a_test)
			aux = v_clf.predict_proba(v_test)

			x_test = x_test.tolist()
			for i in range(len(x_test)):
				x_test[i].extend(aux[i])


			#Stack training
			clf = LogisticRegression()
			clf_roc = OneVsRestClassifier(clf)

			y_train = label_binarize(y_train, classes=range(17))
			y_test = label_binarize(y_test, classes=range(17))
			probas_ = clf_roc.fit(x_train, y_train).predict_proba(x_test)

			Y_ground_truth.extend(y_test)
			Y_prob.extend(probas_)

		Y_ground_truth = np.array(Y_ground_truth)
		Y_ground_truth = np.ravel(Y_ground_truth)

		Y_prob = np.array(Y_prob)
		Y_prob = np.ravel(Y_prob)
		
	return roc_curve(Y_ground_truth, Y_prob)