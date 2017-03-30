#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import rospy
import math
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import JointState
from dynamixel_arm_controller.srv import *

#publisher
def publish(joints, angles):
  state = JointState()

  state.header.stamp = rospy.Time.now()
  state.name = joints
  state.position = angles

  pub = rospy.Publisher('joint_states', JointState)
  pub.publish(state)


#callback
def move_to_point(request):
  L1 = 77.724 #mm
  BASE = 85.344 #mm (sum with roomba)
  L2 = 171.45 #mm
  L3 = 209.55 #mm

  point = request.knock_point.point

  angles = []

  try:
    if(point.x == 0):
      angles.append(math.pi/2)
    else:
      angles.append(math.atan2(point.y, point.x))

    z1 = math.sqrt((point.x*point.x)+(point.y*point.y)) - L1
    x1 = -1*(point.z - BASE)

    w = math.sqrt((x1*x1)+(z1*z1))

    alpha = math.acos(((L3*L3)-(L2*L2)-(w*w))/(-2*L2*w))
    beta = math.acos(((w*w)-(L2*L2)-(L3*L3))/(-2*L2*L3))

    angles.append((alpha - (math.pi/2) + math.atan2(z1,x1)))
    angles.append((beta - math.pi))

    joints = ["base", "base_to_bone", "bone_to_hand"]

    publish(joints, angles)
    
    return True

  except ValueError:
    rospy.loginfo("Goal outside configuration space!")
    
    return False


if __name__ == '__main__':
  rospy.init_node('dynamixel_arm_ik_service')
  s = rospy.Service('dynamixel_arm_ik_service', service, move_to_point)
  print "Dynamixel arm inverse kinematics service ready!"
  rospy.spin()
