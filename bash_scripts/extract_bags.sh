#!/bin/bash
path='/media/narnia/geometry_dataset'

while read object
do
	rosrun av_utils image_bag_extractor.py $path/img $path/bag/$object.bag
done < $1