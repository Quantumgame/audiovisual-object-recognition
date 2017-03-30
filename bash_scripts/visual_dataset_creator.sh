#!/bin/bash

#roslaunch openni_launch openni.launch depth_registration:=true &

while read object
do
  echo 'Startubg capture of object ' $object
  sleep 20
    
  rosbag record camera/depth/image_raw camera/depth/camera_info camera/rgb/image_raw camera/rgb/camera_info camera/rgb/image_rect_color camera/depth/image_rect_raw camera/depth/image camera/depth/image_rect --limit=100 -O bag/$object
    
   echo 'New iteration in 30s'
   sleep 30
   clear
done < $1