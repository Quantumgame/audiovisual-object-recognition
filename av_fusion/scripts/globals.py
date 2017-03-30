#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os

path = None
savepath = None
elkipath = None
fusionpath = None
number_of_samples = None
number_of_folds = None

sys.path.insert(0, '/home/samuel/Dropbox/Mestrado/repo/workspace/src/audio_descriptor/scripts')

def path_init(dataset):
    
    global path
    global savepath
    global elkipath
    global fusionpath
    global number_of_samples
    global number_of_folds
    global kitchen_path
    global kitchen_size

    savepath = '/home/samuel/Desktop/'+ dataset + '/'
    if not os.path.exists(savepath):
    	os.makedirs(savepath)

    elkipath = '/home/samuel/workspace/'

    fusionpath = '/home/samuel/Dropbox/Mestrado/workspace/src/av_fusion/scripts/'
    

    if dataset == 'geometry':
    	path = '/media/samuel/treasure_chest/geometry_dataset/'
    	number_of_samples = 10
    else:
    	path = '/media/treasure_chest/kitchen_dataset/'
    	number_of_samples = 50

    number_of_folds = 10

    kitchen_path = '/media/samuel/treasure_chest/kitchen_dataset/'
    kitchen_size = 50