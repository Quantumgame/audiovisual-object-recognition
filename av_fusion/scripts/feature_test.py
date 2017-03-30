#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import sys
import globals
import audio_utils as au
from mfcc import *
from wavelet import *
import numpy as np
import csv
import scipy.io.wavfile as wav
from visual_descriptor.srv import *
import rospy
import cv
import cv2
import PIL
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from sklearn.cluster import KMeans
from sklearn.preprocessing import *
import pickle
from subprocess import call
import warnings
warnings.filterwarnings("ignore")

w = []
m = []
ciza = []
cart = []

def popcnt(value):
	counter = 0
	while value:
		value &= value - 1
		counter += 1
	return counter


def hamming_distance(x,y):

	dist = 0
	for i in range(len(x)):
		dist += popcnt(x[i]^y[i])

	return dist


def jaccard_distance(x,y):

	intersection_count = 0
	union_count = 0

	for i in range(len(x)):
		intersection_count += popcnt(x[i]&y[i])
		union_count += popcnt(x[i]|y[i])

	return 1.0 - float(intersection_count)/union_count


def bof_medoid(samples, medoids):
		
	#cluster_pred = clusters.predict(samples)

	bag = [0]*len(medoids)

	if samples:

		#quantization
		for s in samples:
			min_cost = float('inf')
			chosen_medoid = None
			for m in range(len(medoids)):
				cost = hamming_distance(s,medoids[m]);
				if(cost < min_cost):
					chosen_medoid = m
					min_cost = cost
			bag[chosen_medoid] += 1

		max_value = max(bag)

		#normalization
		bag = [(x*1.0)/max_value for x in bag]

	return bag

def bof(samples, clusters):

	k = clusters.get_params()['n_clusters']

	bag = [0]*k

	if samples:
		
		prediction = clusters.predict(samples)	

		#quantization
		for p in prediction:
			bag[p] += 1

		max_value = max(bag)

		#normalization
		bag = [(x*1.0)/max_value for x in bag]

	return bag

def bohb(samples, k):
	bag = [0]*k

	for s in samples:
		for word in s:
			bag[word] += 1

	max_value = max(bag)

	#normalization
	#bag = [(x*1.0)/max_value for x in bag]

	return bag

# #For dominant recombination
# def audio_recombination(audio):
# 	compact_rep = []

# 	intensities = []
# 	for i in range(len(audio)):
# 		for j in range(i+1,len(audio)):
# 			intensities.append(1 if audio[i] > audio[j] else 0)

# 	#pad final to 0 for 8bit encoding
# 	for i in len(intensities)%8:
# 		intensities.append(0)

# 	for i in range((len(audio)*(len(audio)-1))/(2*8)):

# 		bin_rep = ""
# 		for j in range(8):
# 			bin_rep += str(intensities[i*8 +j])

# 		compact_rep.append(int(bin_rep, 2))

# 	return compact_rep

def get_descriptors(obj, i, a_dictionary, a_k, a_noise, v_noise, combination): 

	#Audio filename
	filename = globals.path+'wav/'+obj+'-'+str(i)+'.wav'

	#Reads signal
	(fs,signal) = au.read(filename)

	#Clips signal
	decay = range(10,110,10)
	(RTN, signal, I) = au.RTN(signal, fs, 0.01, decay)

	signal = signal[:I[decay.index(60)]]

	signal = au.amplify(signal)

	audio_descriptor = []

	combination = '{:06b}'.format(combination)
	#TN
	if(int(combination[0])):
		audio_descriptor.extend(RTN)

	#Stacked MFCC
	if(int(combination[1])):
		mfcc = mfcc_descriptor(signal, fs)
		mfcc = np.reshape(mfcc, -1, order='C')
		mfcc = mfcc.tolist()
		m.append(len(mfcc))
		#mfcc.extend((2938 - len(mfcc))*[0])		
		audio_descriptor.extend(mfcc[:4056])

	#BOF
	if(int(combination[2])):
		mfcc = mfcc_descriptor(signal, fs)
		#mfcc = np.reshape(mfcc, -1, order='C')
		#mfcc = mfcc.tolist()
		b = bof(mfcc, a_dictionary)
		audio_descriptor.extend(b)

	#wavelet
	if(int(combination[3])):
		wavelet = dwt_coefs(signal, a_noise, 'haar',6)
		w.append(len(wavelet))
		#wavelet.extend((5050 - len(wavelet))*[0])
		audio_descriptor.extend(wavelet[:5050])
	
	#fftIza
	if(int(combination[4])):
		coefs = au.fft_iza(signal, fs, 0.01, 0.001)

		# ciza.append(len(coefs))
		# coefs.extend((18225 - len(coefs))*[0])
		audio_descriptor.extend(coefs[:25000])

	#fftArt
	if(int(combination[5])):
		coefs = au.cft(signal, fs, 0.01, 0.001)

		cart.append(len(coefs))
		 
		audio_descriptor.extend(coefs)


	#Video
	
	depth = cv.LoadImage(globals.path+'img/'+obj+'-'+str(i)+'-depth.png', cv.CV_LOAD_IMAGE_GRAYSCALE)
	mask = cv.LoadImage(globals.path+'mask.png', cv.CV_LOAD_IMAGE_GRAYSCALE)

	if v_noise > 0:
		rgb = cv2.imread(globals.path+'img/'+obj+'-'+str(i)+'-rgb.png', cv.CV_LOAD_IMAGE_GRAYSCALE)
		noise = np.random.randn(*rgb.shape)*v_noise

		# Add this noise to image
		noisy = rgb + noise
		cv2.imwrite(globals.path + 'noisy.png',noisy)
		rgb = cv.LoadImage(globals.path+'noisy.png', cv.CV_LOAD_IMAGE_GRAYSCALE)
		#rgb = noisy

	else:
		rgb = cv.LoadImage(globals.path+'img/'+obj+'-'+str(i)+'-rgb.png', cv.CV_LOAD_IMAGE_GRAYSCALE)
		
	#cv.fromarray(rgb)
	
	bridge = CvBridge()
	rgb_image = bridge.cv_to_imgmsg(rgb)
	depth_image = bridge.cv_to_imgmsg(depth)  
	mask_image = bridge.cv_to_imgmsg(mask)

	rospy.wait_for_service('masked_base_descriptor_service')
	try:
		descriptor_request = rospy.ServiceProxy('masked_base_descriptor_service', masked_service)
		response = descriptor_request(rgb=rgb_image, depth=depth_image, mask=mask_image)
		base_descriptor = bridge.imgmsg_to_cv(response.descriptor)

		visual_descriptor = np.array(base_descriptor)

	except rospy.ServiceException, e:
		print "Service call failed: %s" % e

	return(audio_descriptor, visual_descriptor.tolist())

def save_features(a_k, v_k, a_noise, v_noise, instance_recog, combination):
	#k: int. Number of clusters for visual bag of features
	#clusters: cluster representation matrix for bag of features clustering
	
	#save train features
	with open(globals.path+'a_features.csv', 'w') as a_file, open(globals.path+'v_features.csv', 'w') as v_file, open(globals.path+'av_features.csv', 'w') as av_file:#, open(globals.path+'dr_features.csv', 'w') as dr_file:

		a_writer = csv.writer(a_file)
		v_writer = csv.writer(v_file)
		av_writer = csv.writer(av_file)
		#dr_writer = csv.writer(dr_file)

		s = 0


		a_dictionary = pickle.load(open("/home/samuel/a_dictionary.p", "rb"))

		v_dictionary = []
		with open(globals.path+'v_dictionary.csv', 'rb') as csvfile:
			reader = csv.reader(csvfile, delimiter=',', quotechar='|')
			for row in reader:
				v_dictionary.append([int(float(x)) for x in row])

		objects_list = open(globals.path+'objects.txt').read().splitlines()
		category = -1
		for obj in objects_list:
			#print obj
			
			#Classification anthology:
			#Instance recognition vs class recognition (cube, prism, etc.)
			if instance_recog:
				category += 1
			else:				
				if obj.startswith('cubo'):
					category = 0
				elif obj.startswith('tetraedro'):
					category = 1
				elif obj.startswith('prisma'):
					category = 2
				elif obj.startswith('octaedro'):
					category = 3
				else:
					category += 1
				
			#10% da base
			for i in range(10):

				(audio,visual) = get_descriptors(obj, i, a_dictionary, a_k, a_noise, v_noise, combination)

				a_features = audio
				if(len(audio) > s):
					s = len(audio)

				#print len(a_features)
				#a_features = bof(audio, a_k, a_dictionary)


				# for a in range(len(audio)):
				# 	audio[a] = [audio[a], 0]				
				#v5
				#a_features = audio#np.mean(audio, axis = 0).tolist() + np.var(audio, axis = 0).tolist()
				#print a_features

				#v4
				






				
				#v0
				#a_features = np.mean(audio, axis = 0).tolist()# + np.var(audio, axis = 0).tolist()

				#v1
				# a_features = np.sum(audio, axis = 0)

				# a_min = np.min(a_features)
				# a_max = np.max(a_features)

				# a_features = a_features.tolist()
				# a_features = [int(0 + (x - a_min)*(255)/(a_max - a_min)) for x in a_features]

				#v2  -- melhor

				# audio = np.delete(audio, 0, 1)
				# print audio.shape

				# audio_l = audio.tolist()
				# #a_features = [sum(x) for x in zip(*audio_l[:15])] + [sum(x) for x in zip(*audio_l[15:31])] + [sum(x) for x in zip(*audio_l[-15:])]
				# a_features = [sum(x) for x in zip(*audio_l[:2])] + [sum(x) for x in zip(*audio_l[2:4])] + [sum(x) for x in zip(*audio_l[-2:])]
				
				# a_min = np.min(a_features)
				# a_max = np.max(a_features)
				# a_features = [int(0 + (x - a_min)*(255)/(a_max - a_min)) for x in a_features]

				# #v3
				# momc = [0]*13
				# audio_l = np.absolute(audio).tolist()

				# for a in audio_l:
				# 	print a.index(max(a))
				# 	momc[a.index(max(a))] += 1

				# a_features = momc
				# a_min.append(np.min(audio, axis = 0).tolist())
				# a_max.append(np.max(audio, axis = 0).tolist())

				# compact_mfcc = compact_audio(np.mean(audio, axis = 0).tolist())
		
				# av = []
				# for keypoint in visual.tolist():
				# 	#av_descriptor = list(keypoint)
				# 	#av_descriptor.extend(compact_mfcc)
				# 	#av.append(av_descriptor)
				# 	av += [keypoint+a_features]#compact_mfcc]

				

				v_features = bof_medoid(visual, v_dictionary)


				#dr_features = bof_medoid(dr, dr_dictionary)
				#v_features = bohb(visual.tolist(), 256)#bof(v_samples, k, v_clusters)

				#av_features = bohb(av, 256)

				#a_features = a_intensities


				#v_features = vq(visual,vocabulaire)

				#v_features = np.mean(visual, axis = 0).tolist() + np.var(visual, axis = 0).tolist() + np.min(visual, axis = 0).tolist() + np.max(visual, axis = 0).tolist()
				
				
				#bag of features, k is the maximum number of clusters possible defined
					
				
				#print v_features
				#av_features = a_features + v_features
				
				#write audio features to file
				a_writer.writerow(a_features + [category])
				
				#write video features to file
				v_writer.writerow(v_features + [category])
				
				#write audiovisual features to file
	 			av_writer.writerow(a_features + v_features + [category])

	 			#dr_writer.writerow(v_features + dr_features + [category])


	#print np.min(m), np.max(m)

	# np_min = np.array(a_min)
	# np_max = np.array(a_max)

	# print np.min(np_min, axis = 0)
	# print np.var(np_min, axis = 0)
	# print '\n\n'
	# print np.max(np_max, axis = 0)
	# print np.var(np_max, axis = 0)
	# #save generalization features
	# with open(path+'a_features_generalization.csv', 'w') as a_file, open(path+'v_features_generalization.csv', 'w') as v_file, open(path+'av_features_generalization.csv', 'w') as av_file:

	# 	a_writer = csv.writer(a_file)
	# 	v_writer = csv.writer(v_file)
	# 	av_writer = csv.writer(av_file)

	# 	v_clusters = pickle.load(open("v_clusters.p", "rb"))
	# 	av_clusters = pickle.load(open("av_clusters.p", "rb"))

	# 	#BoF vocabulaire
	# 	#vocabulaire = cv2.imread(path+'centers.png', cv.CV_LOAD_IMAGE_GRAYSCALE)

	# 	objects_list = open(path+'objetos-teste.txt').read().splitlines()
	# 	for obj in objects_list:
	# 		category = None
	# 		if obj in {'caixasuco_dafruta','caixaleite_porto','caixaleite_itambe'}:
	# 			category = 1
	# 		if obj in {'guaranalata', 'tntlata', 'redbulllata'}:
	# 			category = 2
	# 		if obj in {'caneca_abstrata','caneca_plastico'}:
	# 			category = 3

	# 		for i in range(50):

	# 			(audio,visual) = get_descriptors(obj, i)
				
	# 			#a_features = np.mean(audio, axis = 0).tolist() + np.var(audio, axis = 0).tolist()

	# 			a_intensities = []

	# 			mfcc = np.mean(audio, axis = 0).tolist()
			
	# 			for i in range(len(mfcc)):
	# 				for j in range(i+1,len(mfcc)):
	# 					a_intensities.append(1 if mfcc[i] > mfcc[j] else 0)
		
	# 			#bof
	# 			v_samples = []
	# 			av_samples = []
	# 			for keypoint in visual.tolist():
	# 				binary_representation = []
	# 				for elem in keypoint:
	# 					binary_representation.extend([int(x) for x in '{:08b}'.format(elem)])

	# 				v_samples.append(list(binary_representation))

	# 				binary_representation.extend(a_intensities)
	# 				av_samples.append(binary_representation)

	# 			v_features = bof(v_samples, k, v_clusters)

	# 			av_features = bof(av_samples, k, av_clusters)

	# 			a_features = a_intensities


	# 			#v_features = vq(visual,vocabulaire)

	# 			#v_features = np.mean(visual, axis = 0).tolist() + np.var(visual, axis = 0).tolist() + np.min(visual, axis = 0).tolist() + np.max(visual, axis = 0).tolist()
				
				
	# 			#bag of features, k is the maximum number of clusters possible defined
					
				
	# 			#print v_features
	# 			#av_features = a_features + v_features
				
	# 			#write audio features to file
	# 			a_writer.writerow(a_features + [category])
				
	# 			#write video features to file
	# 			v_writer.writerow(v_features + [category])
				
	# 			#write audiovisual features to file
	# 			av_writer.writerow(a_features + v_features + [category])

	#print 'Feature extraction done! Gentleman, start your engines!'	

	call(["shuf", globals.path+"a_features.csv", "-o", globals.path+"a_features.csv"])
	call(["shuf", globals.path+"v_features.csv", "-o", globals.path+"v_features.csv"])
	call(["shuf", globals.path+"av_features.csv", "-o", globals.path+"av_features.csv"])

	#print s
	#print 'm', np.min(m), np.max(m)

if __name__ == "__main__":
	rospy.init_node('audiovisual_feature_extractor')
	#clusters = pickle.load(open("bag_of_features.p", "rb"))
	dataset = sys.argv[1]
	a_k = int(sys.argv[2])
	v_k = int(sys.argv[3])
	a_noise = int(sys.argv[4])# == 'True'
	v_noise = int(sys.argv[5])# == 'True'
	instance_recog = sys.argv[6] == 'True'
	

	globals.path_init(dataset)

	save_features(a_k, v_k, a_noise, v_noise, instance_recog)

	# print np.min(m), np.max(m)
	# print np.min(w), np.max(w)
	# print np.min(ciza), np.max(ciza)
	# print np.min(cart), np.max(cart)