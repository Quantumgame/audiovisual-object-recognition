#!/usr/bin/python
PKG = 'audiovisual_fusion' # to load manifest and dependencies
import re
import roslib; roslib.load_manifest(PKG)
import rosbag
import rospy
import cv
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import os
import sys

class ImageBagExtractor():
	def __init__(self):
		# Get parameters when starting node from a launch file.
		if len(sys.argv) < 1:
			save_dir = rospy.get_param('save_dir')
			filename = rospy.get_param('filename')
			rospy.loginfo("Bag filename = %s", filename)
		# Get parameters as arguments to 'rosrun my_package bag_to_images.py <save_dir> <filename>', where save_dir and filename exist relative to this executable file.
		else:
			#save_dir = os.path.join(sys.path[0], sys.argv[1])
			#filename = os.path.join(sys.path[0], sys.argv[2])
			save_dir = sys.argv[1]
			filename = sys.argv[2]
			rospy.loginfo("Bag filename = %s", filename)

		# Use a CvBridge to convert ROS images to OpenCV images so they can be saved.
		self.bridge = CvBridge()

		i = -1
		j = -1
		# Open bag file.
		with rosbag.Bag(filename, 'r') as bag:
			for topic, msg, t in bag.read_messages():
				dataset_obj = filename.split("/")[-1].split(".")[0]
				img_type = topic.split("/")[1]
				
				#choose appropriate topic to extract images from
				if topic == "camera/rgb/image_rect_color":
					i += 1
					image_name = str(save_dir)+"/"+dataset_obj+"-"+str(i)+"-"+img_type+".png"
					try:
						cv_image = self.bridge.imgmsg_to_cv(msg, "bgr8")
					except CvBridgeError, e:
						print e

					cv.SaveImage(image_name, cv_image)
				if topic == "camera/depth/image_rect_raw":
					j += 1
					image_name = str(save_dir)+"/"+dataset_obj+"-"+str(j)+"-"+img_type+".png"
					try:
						cv_image = self.bridge.imgmsg_to_cv(msg, "passthrough")
					except CvBridgeError, e:
						print e
					
					cv.SaveImage(image_name, cv_image)
				#if topic == "camera/depth/image":
					#try:
						#cv_image = self.bridge.imgmsg_to_cv(msg, "passthrough")
					#except CvBridgeError, e:
						#print e
					
					#cv.SaveImage(image_name, cv_image)
				#if topic == "camera/depth/image_raw":
					#try:
						#cv_image = self.bridge.imgmsg_to_cv(msg, "passthrough")
					#except CvBridgeError, e:
						#print e
					
					#cv.SaveImage(image_name, cv_image)
				#if topic == "camera/depth/image_rect":
					#try:
						#cv_image = self.bridge.imgmsg_to_cv(msg, "passthrough")
					#except CvBridgeError, e:
						#print e
					
					#cv.SaveImage(image_name, cv_image)
  
if __name__ == '__main__':
	# Initialize the node and name it.
	rospy.init_node(PKG)
	try:
		image_bag_extractor = ImageBagExtractor()
	except rospy.ROSInterruptException: pass