"""
Module for BOF vocabulary generation for both audio and video
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import sys
import globals
from wavelet import *
from mfcc import *
import numpy as np
import csv
import scipy.io.wavfile as wav
from visual_descriptor.srv import *
import rospy
import cv
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from sklearn.cluster import *
from subprocess import call
import pickle
import warnings
warnings.filterwarnings("ignore")

def get_all_descriptors():
	"""
	Computes descriptors from both audio and video from all samples for random pick and creation of vocabulary used in BOF for each modality.
	Returns: tuple formatted as (audio_local_descriptors_list, video_local_descriptors_list)
	"""

	#Bag of all local descriptors from audio and video
	a_descriptors = []
	v_descriptors = []
	
	objects_list = open(globals.path+'objects.txt').read().splitlines()

	#Add all keypoints of sample i of object obj to the bag of features for both modalities
	for obj in objects_list:
		for i in range(globals.number_of_samples):
			#Get descriptor for object obj, sample i
			(a_sample, v_sample) = sample_descriptors(obj, i)
					
			a_descriptors.extend(a_sample)
			v_descriptors.extend(v_sample)

	#pickle.dump(keypoints_bag, open("v_keypoints.p", "wb"))
	
	return (a_descriptors,v_descriptors)


def sample_descriptors(obj, i): 
	"""
	Compute the audio and video descriptors for ith sample of object obj.
	:param obj: string. Object to get descriptors from.
	:param i: string. ith sample of object to get descriptors from.
	"""
	print obj, i

	#Audio
	filename = globals.path+'wav/'+obj+'-'+str(i)+'.wav'

	audio_descriptor = mfcc_descriptor(filename)
	#audio_descriptor = wavelet_coefs(filename,'haar',1)

	#Video
	#rgb image
	rgb = cv.LoadImage(globals.path+'img/'+obj+'-'+str(i)+'-rgb.png', cv.CV_LOAD_IMAGE_GRAYSCALE)
	#depth image
	depth = cv.LoadImage(globals.path+'img/'+obj+'-'+str(i)+'-depth.png', cv.CV_LOAD_IMAGE_GRAYSCALE)
	#mask of interest zone to retrieve descriptors
	mask = cv.LoadImage(globals.path+'mask.png', cv.CV_LOAD_IMAGE_GRAYSCALE)

	bridge = CvBridge()
	rgb_image = bridge.cv_to_imgmsg(rgb)
	depth_image = bridge.cv_to_imgmsg(depth)
	mask_image = bridge.cv_to_imgmsg(mask)

	#call external visual descriptor module
	rospy.wait_for_service('masked_base_descriptor_service')
	try:
		descriptor_request = rospy.ServiceProxy('masked_base_descriptor_service', masked_service)
		response = descriptor_request(rgb=rgb_image, depth=depth_image, mask=mask_image)
		base_descriptor = bridge.imgmsg_to_cv(response.descriptor)

		visual_descriptor = np.array(base_descriptor)

	except rospy.ServiceException, e:
		print "Service call failed: %s" % e
    
	return(audio_descriptor.tolist(), visual_descriptor.tolist())


if __name__ == "__main__":

	rospy.init_node('bag_of_features_computation')

	#dataset to be used
	dataset = sys.argv[1]
	#audio dictionary size to create, value determined from silhouette method
	a_k = int(sys.argv[2])
	#video dictionary size to create, value determined from silhouette method
	v_k = int(sys.argv[3])

	#initialize global variables
	globals.path_init(dataset)

	#local descriptors from all dataset
	(a_descriptors, v_descriptors) = get_all_descriptors()

	#create dictionary from centroids of clustered descriptors
	a_dictionary = MiniBatchKMeans(n_clusters=a_k, max_iter=1000, random_state=1).fit(a_descriptors)

	#cache results for later reuse
	pickle.dump(a_dictionary, open("a_dictionary.p", "wb"))

	print 'Audio clusters created!'

	#export keypoint descriptors to external clustering based on kmedoids on hamming distance
	with open(globals.elkipath+'v_descriptors.csv', 'wb') as f:
	 	writer = csv.writer(f)
	 	writer.writerows(v_descriptors)

	call(["sh",globals.elkipath+'VClusteringELKI.sh'])

	print 'Visual clusters created!'