"""
Module for feature computing and saving
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import sys
import globals
import audio_utils as au
import mfcc
import audio_recombination
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

# a_min = []
# a_max = []
w = []
m = []

def popcnt(value):
	"""
	Calculates number of bits 1 in number
	:param value: int. Value from which to compute popcnt.
	"""
	counter = 0
	while value:
		value &= value - 1
		counter += 1
	return counter

def hamming_distance(x,y):
	"""
	Calculates hamming distance of two values (number of bits 1 at the exact same place in both values)
	:param x: int. Value from which to compute hamming distance.
	:param y: int. Value from which to compute hamming distance.
	"""

	dist = 0
	for i in range(len(x)):
		dist += popcnt(x[i]^y[i])

	return dist

def jaccard_distance(x,y):
	"""
	Calculates jaccard distance of two values (proportion between intersection and union of shared bits between the two values)
	:param x: int. Value from which to compute hamming distance.
	:param y: int. Value from which to compute hamming distance.
	"""
	intersection_count = 0
	union_count = 0

	for i in range(len(x)):
		intersection_count += popcnt(x[i]&y[i])
		union_count += popcnt(x[i]|y[i])

	return 1.0 - float(intersection_count)/union_count

def bow(samples, medoids):
	"""
	Bag of words descriptor, uses BASE medoids as dictionary
	:param samples: array of arrays. Contains one entry per object sample.
	:param medoids: array of arrays. Contains one entry per vocabulary word.
	"""

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
	"""
	Bag Of Frames descriptor, uses MFCC centroids as dictionary
	:param samples: array of arrays. Contains one entry per object sample.
	:param clusters: array of arrays. Contains one entry per vocabulary word.
	"""
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

def audio_descriptor(obj, i, a_dictionary, a_noise):
	"""
	Computes audio descriptor of ith sample of object obj.
	:param obj: string. Object from which descriptor is computed.
	:param i: int. ith sample of object from which descriptor is computed
	"""

	filename = globals.path+'wav/'+obj+'-'+str(i)+'.wav'

	#Reads signal
	(fs,signal) = au.read(filename)

	#Clips signal
	decay = range(10,110,10)
	(RTN, signal, I) = au.RTN(signal, fs, 0.01, decay)

	signal = signal[:I[decay.index(60)]]

	#Adds additive noise to signal, if a_noise != float('inf'), where a_noise is SNR between signal and noise
	signal = au.get_noisy_signal(signal, a_noise, noise_type='white')

	signal = au.amplify(signal)

	#Gets RTN descriptor from updated signal
	(RTN, signal, I) = au.rtn(signal, fs, 0.01, decay)

	#audio descriptor
	a_descriptor = []

	#RTN
	a_descriptor.extend(RTN)

	#Calculates MFCC descriptor from signal
	mfcc = mfcc_descriptor(signal, fs)
	b = bof(mfcc, a_dictionary)
	a_descriptor.extend(b)

	#Calculates FFT descriptor from signal
	fftcoefs = au.cft(signal, fs, 0.01, 0.001)

	a_descriptor.extend(fftcoefs)

	return a_descriptor

def video_descriptor(obj , i, v_dictionary, v_noise): 
	
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

	#BASE Keypoint descriptors
	keypoint_descriptors = []
	rospy.wait_for_service('masked_base_descriptor_service')
	try:
		descriptor_request = rospy.ServiceProxy('masked_base_descriptor_service', masked_service)
		response = descriptor_request(rgb=rgb_image, depth=depth_image, mask=mask_image)
		base_descriptor = bridge.imgmsg_to_cv(response.descriptor)


		keypoint_descriptors = np.array(base_descriptor)
		keypoint_descriptors = keypoint_descriptors.tolist()

	except rospy.ServiceException, e:
		print "Service call failed: %s" % e


	#Bag of words descriptor
	v_descriptor = bow(keypoint_descriptors, v_dictionary)

	return v_descriptor


def save_features(a_noise, v_noise, instance_recog):
	#k: int. Number of clusters for visual bag of features
	#clusters: cluster representation matrix for bag of features clustering
	
	#save train features
	with open(globals.path+'a_features.csv', 'w') as a_file, open(globals.path+'v_features.csv', 'w') as v_file, open(globals.path+'av_features.csv', 'w') as av_file:#, open(globals.path+'dr_features.csv', 'w') as dr_file:

		a_writer = csv.writer(a_file)
		v_writer = csv.writer(v_file)
		av_writer = csv.writer(av_file)

		#cached dictionary
		a_dictionary = pickle.load(open("/home/samuel/a_dictionary.p", "rb"))

		#pickle.dump(a_dictionary, open("/home/samuel/a_dictionary.p", "wb"))

		v_dictionary = []
		with open(globals.path+'v_dictionary.csv', 'rb') as csvfile:
			reader = csv.reader(csvfile, delimiter=',', quotechar='|')
			for row in reader:
				v_dictionary.append([int(float(x)) for x in row])

		# dr_dictionary = []
		# with open(globals.fusionpath+'dr_dictionary.csv', 'rb') as csvfile:
		# 	reader = csv.reader(csvfile, delimiter=',', quotechar='|')
		# 	for row in reader:
		# 		dr_dictionary.append([int(float(x)) for x in row])

		#v_clusters = pickle.load(open("v_clusters.p", "rb"))
		#av_clusters = pickle.load(open("av_clusters.p", "rb"))

		#BoF vocabulaire
		#vocabulaire = cv2.imread(path+'centers.png', cv.CV_LOAD_IMAGE_GRAYSCALE)

		objects_list = open(globals.path+'objects.txt').read().splitlines()
		category = -1
		for obj in objects_list:
			
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
				
			for i in range(globals.number_of_samples):
				print obj, i

				#category = np.random.random_integers(0, len(objects_list) - 1)

				#(a_features,v_features) = get_descriptors(obj, i, a_dictionary, v_dictionary, a_noise, v_noise)

				a_features = audio_descriptor(obj, i, a_dictionary, a_noise)

				v_features = video_descriptor(obj, i, v_dictionary, v_noise)

				#write audio features to file
				a_writer.writerow(a_features + [category])
				
				#write video features to file
				v_writer.writerow(v_features + [category])
				
				#write audiovisual features to file
	 			av_writer.writerow(a_features + v_features + [category])


	print 'Feature extraction done! Gentleman, start your engines!'	

	call(["shuf", globals.path+"a_features.csv", "-o", globals.path+"a_features.csv"])
	call(["shuf", globals.path+"v_features.csv", "-o", globals.path+"v_features.csv"])
	call(["shuf", globals.path+"av_features.csv", "-o", globals.path+"av_features.csv"])

if __name__ == "__main__":
	rospy.init_node('audiovisual_feature_extractor')
	#clusters = pickle.load(open("bag_of_features.p", "rb"))
	dataset = sys.argv[1]
	#a_k = int(sys.argv[2])
	#v_k = int(sys.argv[3])
	a_noise = float('inf')#int(sys.argv[2])# == 'True'
	v_noise = int(sys.argv[3])# == 'True'
	instance_recog = sys.argv[4] == 'True'
	

	globals.path_init(dataset)

	save_features(a_noise, v_noise, instance_recog)
	