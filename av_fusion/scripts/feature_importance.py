"""
Module for feature importance analysis
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import globals
import numpy as np
import pylab as pl
import sys
import csv
from sklearn import datasets, svm
from sklearn.feature_selection import SelectPercentile, f_classif

def plot_fscore_graph(filename):
	X = []
	Y = []

	#open audio features dump
	with open(globals.path+'a_features.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='|')
		for row in reader:
			X.append([float(x) for x in row[:-1]])
			Y.append(int(row[-1]))

	pl.figure(1)
	pl.clf()

	X = np.asarray(X)

	X_indices = np.arange(X.shape[-1])

	#univariate feature selection with F-test for feature scoring
	#the selection function uses the 10% most significant features
	selector = SelectPercentile(f_classif, percentile=10)
	selector.fit(X, Y)
	scores = -np.log10(selector.pvalues_)
	print scores
	print np.argmax(scores)

	#plot graph
	pl.bar(X_indices - .45, scores, width=.2,
	       label=r'Univariate score ($-Log(f_{score})$)', color='g')

	pl.title("Feature impact")
	pl.xlabel('Feature')
	pl.axis('tight')
	pl.legend(loc='lower right')
	pl.savefig(filename)


if __name__ == "__main__":
	#dataset to use
	dataset = sys.argv[1]
	#initialize global variables
	globals.path_init(dataset)

	plot_fscore_graph('/home/samuel/Desktop/selection.png')
