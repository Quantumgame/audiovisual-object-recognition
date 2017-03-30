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
	Y_predicted = []
	Y_prob = []


	for classifier in ['NaiveBayes','RandomForest']:
		 #NaiveBayes', 'DecisionTree', 'LogisticRegression', 'LDA', 'Adaboost', 'GradientBoosting', 'RandomForest', 'ANN', 'SVM', 'KNN']:
		#print "Training %s" % modality_type
		
		Y_ground_truth = []
		Y_predicted = []
		Y_prob = []

		acc = []
		precision = []
		f1 = []
		
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
			if classifier == 'NaiveBayes':
				clf = GaussianNB().fit(x_train,y_train)
			elif classifier == 'DecisionTree':
				clf = DecisionTreeClassifier().fit(x_train,y_train)
			elif classifier == 'LogisticRegression':
				clf = LogisticRegression().fit(x_train,y_train)
			elif classifier == 'LDA':
				clf = LDA().fit(x_train,y_train)
			elif classifier == 'Adaboost':
				clf = AdaBoostClassifier(n_estimators=100).fit(x_train,y_train)
			elif classifier == 'GradientBoosting':
				clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(x_train,y_train)
			elif classifier == 'RandomForest':
				clf = RandomForestClassifier(n_estimators=100).fit(x_train,y_train)
			elif classifier == 'ANN':
				clf = Perceptron(penalty=None, alpha=0.0001, fit_intercept=True, n_iter=20, shuffle=False, verbose=0, eta0=1.0, n_jobs=1, random_state=0, class_weight=None, warm_start=False, seed=None).fit(x_train,y_train)
			elif classifier == 'SVM':
				clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0, kernel='linear', max_iter=-1, probability=True, random_state=None,shrinking=True, tol=0.001, verbose=False).fit(x_train,y_train)
			elif classifier == 'KNN':
				clf = KNeighborsClassifier(n_neighbors=20).fit(x_train,y_train)
			else:
				pass

			y_pred = clf.predict(x_test)

			acc.append(accuracy_score(y_test, y_pred))

			Y_ground_truth.extend(y_test)
			Y_predicted.extend(y_pred)
			Y_prob.extend(clf.predict_proba(x_test))

		# print '#################################'
		# print acc
		# print np.mean(acc)
		# print np.std(acc)

		# # print get_mean_ci(acc)
		# print '----------------------------------'

		# fpr, tpr, thresholds = roc_curve(Y_ground_truth, Y_prob)
		# roc_auc = auc(fpr, tpr)
		# print auc(Y_ground_truth, Y_predicted), f1_score(Y_ground_truth, Y_predicted), accuracy_score(Y_ground_truth, Y_predicted)
			
		#Mean accuracy and std of each classifier
		mean_acc.append(100*np.mean(acc))
		std.append(100*np.std(acc))

	cm = confusion_matrix(Y_ground_truth, Y_predicted)

	return mean_acc, std, cm