#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import os
import sys
import csv
import shutil
import numpy as np
import rospy
import matplotlib.pyplot as plt
from feature_saver import *
from single_modality_classification import *
from meta_classification import *

def plot_confusion_matrix(cm, objects_list, fusion_type, filename):
	
	pl.clf()

	pl.imshow(cm, interpolation='nearest')
	for i, cas in enumerate(cm):
		for j, c in enumerate(cas):
			if c > 0:
				pl.text(j-.2, i+.2, c, fontsize=10)

	title = u'Matriz de confusão - ' + fusion_type

	pl.title(title)
	#pl.colorbar()
	tick_marks = np.arange(len(objects_list))
	pl.xticks(tick_marks, objects_list, rotation=90)
	pl.yticks(tick_marks, objects_list)
	pl.tight_layout()
	pl.ylabel('Categoria prevista')
	pl.xlabel('Categoria real')
	pl.savefig(filename)

def autolabel(ax, rects, mu, var):
    # attach some text labels
    k = -1
    for rect in rects:
    	k += 1
    	height = rect.get_height()
    	ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, u'$\sigma=%.1f$'%stds[k],
            ha='center', va='bottom')
    	ax.text(rect.get_x()+rect.get_width()/2., 1.05*height + 3, u'$\mu=%.1f$'%means[k],
            ha='center', va='bottom')


if __name__ == "__main__":
	rospy.init_node('audiovisual_noiseplot')
	#clusters = pickle.load(open("bag_of_features.p", "rb"))
	dataset = sys.argv[1]
	a_k = int(sys.argv[2])
	v_k = int(sys.argv[3])
	# a_noise = int(sys.argv[3])# == 'True'
	# v_noise = int(sys.argv[5])# == 'True'
	instance_recog = sys.argv[4] == 'True'

	globals.path_init(dataset)

	# print np.min(m), np.max(m)
	# print np.min(w), np.max(w)

	objects_list = open(globals.path+'objects.txt').read().splitlines()

	SNR = [float('-inf')]
	SNR.extend([-20, -15, -10, -5, 0, 5, 10, 20, 30, 40, 50])
	SNR.append(float('inf'))
	#SNR = [-15, -5, 5]

	v_noise = 0

	instance_recog = 'True'

	audioMean = []
	audioStd = []

	videoMean = []
	videoStd = []

	favMean = []
	favStd = []

	mavMean = []
	mavStd = []

	for a_noise in SNR:

		save_features(a_noise, v_noise, instance_recog)

		SNRpath = globals.savepath + str(a_noise) + '/'

		if not os.path.exists(SNRpath):
		 	os.makedirs(SNRpath)

		shutil.copy(globals.path+'a_features.csv', SNRpath)
		shutil.copy(globals.path+'v_features.csv', SNRpath)
		shutil.copy(globals.path+'av_features.csv', SNRpath)

		#shutil.copy(SNRpath+'a_features.csv', globals.path)
		#shutil.copy(SNRpath+'v_features.csv', globals.path)
		#shutil.copy(SNRpath+'av_features.csv', globals.path)

		#([1,2,3,4],[1,2,3,4])#
		(mean_audio,std_audio, cm) = single_modality_classification('a')

		plot_confusion_matrix(cm, objects_list, u'Áudio, SNR = ' + str(a_noise) + ' dB', SNRpath + 'cm-a.png')

		(mean_video,std_video, cm) = single_modality_classification('v')#single_modality_classification(10,'v')

		plot_confusion_matrix(cm, objects_list, u'Vídeo', SNRpath + 'cm-v.png')

		(mean_feature_av, std_feature_av, cm) = single_modality_classification('av')

		plot_confusion_matrix(cm, objects_list, u'FAV, SNR = ' + str(a_noise) + ' dB', SNRpath + 'cm-fav.png')

		(mean_meta_av, std_meta_av, cm) = meta_classification(a_k, v_k)

		plot_confusion_matrix(cm, objects_list, u'MAV, SNR = ' + str(a_noise) + ' dB', SNRpath + 'cm-mav.png')

		colors = "bgrcmykwbg"
		
		n_groups = 4
		
		fig = plt.figure()
		ax = fig.add_subplot(111)

		index = np.arange(n_groups)#([0,2.5,5])#,7.5])

		width = 0.2
		opacity = 0.4
		error_config = {'ecolor': '0.3'}
	  
	  	rects = []
		i = 0
		for classifier in [u"NaiveBayes", u"RandomForest"]:#NaiveBayes', 'DecisionTree', 'LogisticRegression', 'LDA', 'Adaboost', 'GradientBoosting', 'RandomForest', 'ANN', 'SVM', 'KNN']:
			means = [mean_audio[i], mean_video[i], mean_feature_av[i], mean_meta_av[i]]#, mean_drav[i]]#, mean_decision_av[i]]
			stds = [std_audio[i], std_video[i], std_feature_av[i], std_meta_av[i]]#, std_drav[i]]#, std_decision_av[i]] 

			bars = ax.bar(index + i*width, means, width,
					alpha=opacity,
	                color=colors[i],
	                yerr=stds,
	                error_kw=error_config,
	                label=classifier)

			rects.append(bars)

			autolabel(ax, bars, means, stds)
			# plt.bar(index, means, bar_width,
			# 		alpha=opacity,
			# 		color=colors[i],
			# 		yerr=stds,
			# 		error_kw=error_config,
			# 		label=classifier)

			#for k in range(4):
			#	ax.annotate(r'$\mu=%.2f, \sigma=%.3f$'%(means[k],stds[k]), (index[k] + i*0.4, means[k] - stds[k] - 3))

			i += 1

		# axes and labels
		ax.set_xlim(-width,len(index)+width)
		ax.set_ylim(0,110)

		ax.set_xlabel(u"Abordagem")
		ax.set_ylabel(u"Taxa de acerto (%)")
		ax.set_title(u"Comparação de desempenho - fusão audiovisual")


		#approach = [u'Áudio', 'Vídeo', u'Fusão AV feature', u'Meta fusão AV']
		xTickMarks = [u"Áudio", u"Vídeo", u"FAV", u"MAV", u"DRAV"]
		ax.set_xticks(index+width)
		xtickNames = ax.set_xticklabels(xTickMarks)
		plt.setp(xtickNames, rotation=45, fontsize=10)

		#ax.legend( (rects[0,0], rects2[1,0]), ('SVM', 'Random Forest') )

		plt.legend(loc='lower center', prop={'size':'small'}, fancybox=True, shadow=True)
		plt.grid()
		#plt.tight_layout()
		#plt.savefig('resultados20.png')
		plt.savefig(SNRpath + 'result.png')

		k = 1
		audioMean.append(mean_audio[k])
		audioStd.append(std_audio[k])

		videoMean.append(mean_video[k])
		videoStd.append(std_video[k])

		favMean.append(mean_feature_av[k])
		favStd.append(std_feature_av[k])

		mavMean.append(mean_meta_av[k])
		mavStd.append(std_meta_av[k])

	plt.clf()
	fig = plt.figure()
	ax = fig.add_subplot(111)
	plt.title(u'Relação taxa de acerto x SNR')
	plt.xlabel(u'SNR (dB)')
	plt.ylabel(u'Taxa de acerto (%)')

	ticks = range(len(SNR))
	ax.set_xticks(ticks)
	plt.xlim(ticks[0] - 1, ticks[len(ticks) - 1] + 1)
	
	#xtickNames = [str(s) for s in SNR]
	ticksNames = ax.set_xticklabels(SNR)
	plt.setp(ticksNames)

	ax.errorbar(ticks, audioMean, color=colors[0], yerr=audioStd, label=u'Áudio')
	ax.errorbar(ticks, videoMean, color=colors[3], yerr=videoStd, label=u'Vídeo')
	ax.errorbar(ticks, favMean, color=colors[1], yerr=favStd, label=u'FAV')
	ax.errorbar(ticks, mavMean, color=colors[2], yerr=mavStd, label=u'MAV')

	# box = ax.get_position()
	# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

	# Put a legend to the right of the current axis
	ax.legend(loc='bottom right', bbox_to_anchor=(1, 0.5))
	#plt.legend(loc='center right', prop={'size':'small'}, fancybox=True, shadow=True, bbox_to_anchor=(1, 0.5))
	plt.savefig(globals.savepath + 'error.png')


  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  

  #path = '/home/samuel/Dropbox/Dissertacao/repo/samples/smalldataset/'

  #with open(path+'av_random_features.csv', 'rb') as csvfile:
    #reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    #for row in reader:
      ##x_temp = []
      ##for e in row[:-1]:
	##x_temp.append(bin(int(e)).count('1'))
      ##X.append(x_temp)
      #X.append([float(x) for x in row[:-1]])
      #Y.append(int(row[-1]))
      

  #with open(path+'av_decision_mean.txt', 'w') as text_file:

    ##for classifier in ['NaiveBayes', 'DecisionTree', 'LogisticRegression', 'LDA', 'Adaboost', 'GradientBoosting', 'RandomForest', 'ANN', 'SVM', 'KNN']:
      
    #acc = []
    #acc2 = []
    #a_acc1 = []
    #a_acc2 = []
    #v_acc1 = []
    #v_acc2 = []
    #precision = []
    #f1 = []
    
    #kf = KFold(len(X), 5)
    #for train_index, test_index in kf:
	    
      ## KFold split		
      #x_train = np.zeros( shape = ( len(train_index) , len(X[0]) ))
      #y_train = np.zeros( shape = ( len(train_index) ))
      #for i in range(len(train_index)):
	      #for j in range(len(X[0])):
		      #x_train[i][j] = X[train_index[i]][j]
	      #y_train[i] = Y[train_index[i]]
      #x_test = np.zeros( shape = ( len(test_index) , len(X[0]) ))
      #y_test = np.zeros( shape = ( len(test_index) ))
      #for i in range(len(test_index)):
	      #for j in range(len(X[0])):
		      #x_test[i][j] = X[test_index[i]][j]
	      #y_test[i] = Y[test_index[i]]

      #subsetSize = 0.6
      ##scores = []

	##x_train, unusedX, y_train, unusedY = train_test_split(x_train, y_train, train_size=subsetSize, random_state=1)

	##clf = None
	##if classifier == 'NaiveBayes':
	  ##clf = GaussianNB().fit(x_train,y_train)
	##elif classifier == 'DecisionTree':
	  ##clf = DecisionTreeClassifier().fit(x_train,y_train)
	##elif classifier == 'LogisticRegression':
	  ##clf = LogisticRegression().fit(x_train,y_train)
	##elif classifier == 'LDA':
	  ##clf = LDA().fit(x_train,y_train)
	##elif classifier == 'Adaboost':
	  ##clf = AdaBoostClassifier(n_estimators=100).fit(x_train,y_train)
	##elif classifier == 'GradientBoosting':
	  ##clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(x_train,y_train)
	##elif classifier == 'RandomForest':
	  ##clf = RandomForestClassifier(n_estimators=100).fit(x_train,y_train)
	##elif classifier == 'ANN':
	  ##clf = Perceptron(penalty=None, alpha=0.0001, fit_intercept=True, n_iter=20, shuffle=False, verbose=0, eta0=1.0, n_jobs=1, random_state=0, class_weight=None, warm_start=False, seed=None).fit(x_train,y_train)
	##elif classifier == 'SVM':
	  ##clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0, kernel='linear', max_iter=-1, probability=False, random_state=None,shrinking=True, tol=0.001, verbose=False).fit(x_train,y_train)
	##else:
	  ##pass
	
	##y_pred = None
	##if classifier == 'KNN':
	  ##y_pred = knn(x_train, y_train, x_test, 5)
	##else:
	  ##y_pred = clf.predict(x_test)
      #x_train_v = [x[:64] for x in x_train]
      #x_train_a = [x[-26:] for x in x_train]
      
      #x_test_v = [x[:64] for x in x_test]
      #x_test_a = [x[-26:] for x in x_test]
      
      #v_clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0, kernel='linear', max_iter=-1, probability=True, random_state=None,shrinking=False, tol=0.001, verbose=False).fit(x_train_v,y_train)
      ##y_pred_v = v_clf.predict(x_test_v)
      
      #a_clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0, kernel='linear', max_iter=-1, probability=True, random_state=None,shrinking=False, tol=0.001, verbose=False).fit(x_train_a,y_train)
      ##y_pred_a = a_clf.predict(x_test_a)
      
      #v_clf2 = LogisticRegression().fit(x_train_v,y_train)
      ##y_pred_v = v_clf.predict(x_test_v)
      
      #a_clf2 = LogisticRegression().fit(x_train_a,y_train)
      ##y_pred_a = a_clf.predict(x_test_a)
      
      ##for i in range(len(y_test)):
	##print y_pred_a[i], y_pred_v[i], y_test[i]
	
      ##probs = np.add(a_clf.predict_proba(x_test_a)*a_clf.score(x_test_a,y_test), v_clf.predict_proba(x_test_v)*v_clf.score(x_test_v,y_test))
      #probs = a_clf.predict_proba(x_test_a)**a_clf.score(x_test_a,y_test)*v_clf.predict_proba(x_test_v)**v_clf.score(x_test_v,y_test)
      #probs2 = a_clf2.predict_proba(x_test_a)**a_clf2.score(x_test_a,y_test)*v_clf2.predict_proba(x_test_v)**v_clf2.score(x_test_v,y_test)

      ##print probs
      #y_pred = probs.argmax(axis=1) + 1.0
      #y_pred2 = probs2.argmax(axis=1) + 1.0
      ##print y_pred
      ##print y_test
      
      ##y_pred = []
      ##for i in probs:
	##y_pred.append(probs.index(max(probs)) + 1)
      #acc.append(accuracy_score(y_test, y_pred))
      #acc2.append(accuracy_score(y_test, y_pred2))
    #print acc
    #print np.mean(acc)
    #print acc2
    #print np.mean(acc2)

      ##clf = GMM(n_components=5, covariance_type='diag', random_state=None, thresh=0.01, min_covar=0.001, n_iter=100, n_init=1, params='mc', init_params='mc')
      ##clf.fit(x_train)
      ##y_pred = clf.predict(x_test)
      ###print clf.predict_proba(x_test)
      ##print accuracy_score(y_test, y_pred)
      ##acc.append(accuracy_score(y_test, y_pred))
	##auc += auc(y_test, y_pred)  
	##precision.append(precision_score(y_test, y_pred, average='macro'))
	##f1.append(f1_score(y_test, y_pred))
	
      ##text_file.write('%s\t%.3f\t%.3f\t%.3f\n' % (classifier, np.mean(acc), min(acc), max(acc)))
    ##text_file.write('%s\t%.3f\n' % ('GMM', np.mean(acc)))