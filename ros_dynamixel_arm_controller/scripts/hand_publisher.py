#!/usr/bin/env python

import rospy
import yaml

from time import time
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

hand_positions = None
hand_mapping = None
hand_joints = {}

joint_publisher = None

current_position = None

def remap_cfg(data):

	cfg = {}

	for i, name in enumerate( data.name ):
		cfg[ name ] = float(data.position[i])

	return cfg

def callback(data):
	for i, name in enumerate( data.name ):
		set_angle( name, data.position[i] )

def set_angle(to, angle):
	global hand_joints
	global hand_mapping

	if to in hand_mapping:
		for motor in hand_mapping[to]:

			value = -angle if "-" in motor else angle
			motor = motor.replace("-", "")

			if motor in hand_joints:
				hand_joints[motor].publish( Float64( value ) )

def update_state( current_position ):
	global hand_positions

	if current_position in hand_positions:
		for joint, angle in hand_positions[ current_position ].items():

			rospy.loginfo("%s:%f", joint, angle)

			set_angle( joint, angle )

def update_joints( current_position ):
	global hand_positions
	global joint_publisher

	if current_position in hand_positions:

		state = JointState()

		state.name = []
		state.position = []

		for joint, angle in hand_positions[ current_position ].items():

			state.name.append( joint )
			state.position.append( angle )

		joint_publisher.publish( state )

def listener():

	global hand_positions
	global hand_joints
	global hand_mapping
	global joint_publisher
	global current_position

	rospy.init_node("hand_publisher")

	rospy.Subscriber("/joint_states", JointState, callback)
	joint_publisher = rospy.Publisher("/joint_states", JointState)

	#joint_publisher = rospy.Publisher("/hand_joint_states", JointState)

	hand_positions = rospy.get_param("~hand_positions")
	hand_mapping = rospy.get_param("~hand_mapping")

	for joint, motor_list in hand_mapping.items():
		for motor in motor_list:
			motor = motor.replace("-", "")
			hand_joints[motor] = rospy.Publisher("/%s/command" % motor, Float64)

	
	rate = rospy.Rate(10)

	while not rospy.is_shutdown():

		new_current = rospy.get_param("~hand_current_position")

		if new_current != current_position:
			current_position = new_current

			update_state(current_position)
			update_joints(current_position)

		rate.sleep()

if __name__ == '__main__':
	listener()
