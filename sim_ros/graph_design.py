#!/usr/bin/env python
import os
import sys
import rospy

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker

import tf
import math
import json
import numpy as np

def generate_obstacles_geom(map_width, map_height, map_array):
	""" Given a 2D occupancy grid where each (i,j) element of the array
		'map_array' specifies whether an obstacle is present at that
		position of the grid or not, compute the list of objects that are
		present in the grid. This is simply done by regrouping neighboring
		cells that are considered to be obstacles as a single objet
		:param map_width : Number of cells in the x axis
		:param map_height : Number of cells in the y axis
		:param map_array : 2D occupancy grid
	"""
	obstacle_list = list()
	for width in range(map_width):
		for height in range(map_height):
			explored = set()
			to_explore = set()
			if map_array[height*map_width + width] > 0:
				to_explore.add((width,height))
				map_array[height*map_width + width] = 0
			while len(to_explore)>0:
				new_to_explore = set()
				for elem in to_explore:
					neighbors = getNeighbors(elem, map_width,map_height)
					for neigh in neighbors:
						if neigh in explored or neigh in to_explore:
							continue
						if map_array[neigh[1]*map_width + neigh[0]] > 0:
							new_to_explore.add((neigh[0],neigh[1]))
							map_array[neigh[1]*map_width + neigh[0]] = 0
					explored.add(elem)
				to_explore = new_to_explore
			if len(explored) <= 1:
				for (x,y) in explored:
					obstacle_list.append((x,y,x,y))
				continue
			min_x = map_width
			min_y = map_height
			max_x = 0
			max_y = 0
			for (x,y) in explored:
				if x < min_x:
					min_x = x
				if x > max_x:
					max_x = x
				if y < min_y:
					min_y = y
				if y > max_y:
					max_y = y
			obstacle_list.append((min_x,min_y,max_x,max_y))
	return obstacle_list


def grid2d_callback(my_map):
	""" Callback function obtaining the 2D occupancy grid via ROS topic,
		then it computes all the obstacles present in the world and finally
		respawn this obstacles to RVIZ while saving the list of obstacles
		found as a numpy array
	"""
	global firstTime
	global grid_map
	global obstacle_geom
	if firstTime:
		grid_map = my_map
		map_array = np.array([x for x in grid_map.data])
		firstTime = False
	else:
		map_array = np.array([x for x in grid_map.data])
		for width in range(my_map.info.width):
			for height in range(my_map.info.height):
				if my_map.data[height*my_map.info.width + width] > 0:
					map_array[height*my_map.info.width + width] = \
							my_map.data[height*my_map.info.width + width]

	obstacle_geom = generate_obstacles_geom(grid_map.info.width,
							grid_map.info.height, map_array)
	send_obstacle_geom(obstacle_geom,grid_map.info)

def saveBinsInfo(saveFileName,curr_res, originPosition):
	""" Save a representation of the underlying graph as a json file.
		The global variables binsDict, transVect and transPoints contains
		the bins partitioning, the transition between bins, and the 
		continuous coordinate of the points defining each bins, respectively
	"""
	global binsDict
	global transVect
	global transPoints
	dictResult = dict()
	dictResult['resolution'] = curr_res
	dictResult['bins'] = dict()
	dictResult['trans'] = transVect
	dictResult['transTrue'] = transPoints
	with open(saveFileName, 'w') as fp:
		for binsId , listSpot in binsDict.items():
			dictResult["bins"][binsId] = list()
			for pos in set(listSpot):
				dictResult["bins"][binsId].append(
					grid2pos((pos[0],pos[1]),originPosition,curr_res))
		json.dump(dictResult, fp)

def send_bins_geom(resolution, origin):
	""" Draw the bins in the RVIZ visualization
	"""
	global grid_map
	global seq_bins
	# resolution = grid_map.info.resolution
	# z = grid_map.info.origin.position.z
	z = origin.z
	rateSend = rospy.Rate(100)
	marker = Marker()
	marker.header.frame_id = "/odom"
	marker.header.stamp = rospy.Time.now()
	marker.ns = 'bins'
	marker.type = 1
	marker.action = 3
	# print "Before publish"
	binsPub.publish(marker)
	# print "After publish"
	for i in range(50):
		rateSend.sleep()
	seq_bins =  seq_bins + 1
	# print "After sleep"
	marker.action = 0
	marker.pose.orientation.w  = 1.0
	marker.color.r = 0.0
	marker.color.g = 1.0
	marker.color.b = 0.0
	marker.color.a = 0.5
	nbTerm = 0
	# print "in sending bins"
	for (key, binsVal) in  binsDict.items():
		for elem in set(binsVal):
			# pos_xy = grid2pos(elem, grid_map.info.origin.position, resolution)
			pos_xy = grid2pos(elem, origin, resolution)
			marker.pose.position = Point(pos_xy[0],pos_xy[1],z)
			marker.scale.x = resolution
			marker.scale.y = resolution
			marker.scale.z = 0.01
			marker.header.seq = seq_bins
			marker.id = nbTerm
			marker.header.stamp = rospy.Time.now()
			binsPub.publish(marker)
			rateSend.sleep()
			seq_bins =  seq_bins + 1
			nbTerm = nbTerm + 1
		# Send the text message

def send_transitions(z):
	""" Draw the transitions between the bins in the RVIZ visualization
	"""
	global transPoints
	global seq_trans
	rateSend = rospy.Rate(100)
	marker = Marker()
	marker.header.frame_id = "/odom"
	marker.header.stamp = rospy.Time.now()
	marker.ns = 'transitions'
	marker.type = 4
	marker.action = 3
	transPub.publish(marker)
	for i in range(10):
		rateSend.sleep()
	seq_trans = seq_trans + 1
	marker.action = 0
	marker.pose.orientation.w  = 1.0
	marker.color.r = 0.0
	marker.color.g = 0.0
	marker.color.b = 1.0
	marker.color.a = 0.5
	marker.scale.x = 0.5
	marker.scale.y = 0.5
	marker.scale.z = 0.5
	nbTerm = 0
	for (pt1,pt2) in transPoints:
		marker.points = [Point(pt1[0],pt1[1],z), Point(pt2[0],pt2[1],z)]
		marker.id = nbTerm
		marker.header.stamp = rospy.Time.now()
		marker.header.seq = seq_trans
		transPub.publish(marker)
		rateSend.sleep()
		seq_trans = seq_trans + 1
		nbTerm = nbTerm + 1

def binsLoc_callback(clicked_point):
	""" This function is the main function to build a graph, its bins, and 
		the transition between bins using RVIZ as the interface to visualize
		and define the graph.
		Specifically, assuming, RVIZ is opened and shows a gridworld with
		obstacles and non-obstacls, This function behaves as follows:
			- If a mouse click event is raised at two cells above the origin,
				then a name of teh file to save the current configuration is
				requested in the terminal and the current conf is saved
			- If a mouse click event is raised at the origin, then the user
				will be asked to enumerate the bin that is about to be created.
				After this, if the user click on any cell, the cell will be considered
				as part of that bin. To finalize the construction of the
				current bin, the user must click one more time at the origin,
			- If a mouse click event is raised 2 cells below the origin, the user
				is expected to define the transitions between bins. CLick in two
				bins to define a link betwen=en the bins.
	"""
	global DATA_SAVE_DIR
	global lastClick
	global grid_map
	global binsDict
	global binsSave
	global transEnable
	global transPoints
	global bin_number
	global transVect
	global nbCell
	global curr_res
	recv_point = clicked_point.point
	# nbCell = 80
	trigPoint = (nbCell/2,nbCell/2)
	savePoint = (nbCell/2,nbCell/2+2)
	transPoint =(nbCell/2, nbCell/2-2)
	# curr_res = 5.0
	originPosition = Point(-(nbCell*curr_res)/2 , -(nbCell*curr_res)/2, 0.0)
	gridVal = pos2grid((recv_point.x,recv_point.y),originPosition,curr_res)
	print (gridVal)
	# print gridVal
	if gridVal == savePoint:
		saveFileName = raw_input("Bins file name : ")
		saveBinsInfo(DATA_SAVE_DIR+saveFileName,curr_res, originPosition)
		return
	if gridVal == transPoint and not transEnable:
		transEnable = True
		return
	if gridVal != transPoint and transEnable:
		transEnable = False
		binLast = -1
		binCurr = -1
		for (key, binsVal) in  binsDict.items():
			if lastClick in binsVal:
				binLast = key
			if gridVal in binsVal:
				binCurr = key
			if binLast != -1 and binCurr != -1:
				print ("Trans: ",(binLast,binCurr))
				transPoints.append((grid2pos(lastClick,originPosition,curr_res),
									grid2pos(gridVal,originPosition,curr_res)))
				transVect.append((binLast,binCurr))
				send_transitions(originPosition.z)
				return
		print "No bins match !"
		return
	if gridVal == trigPoint and not binsSave:
		binsSave = True
		bin_number = int(raw_input("Bin number (int) : "))
		print ("Starting saving a bin")
		binsDict[bin_number] = list()
		return
	if gridVal == trigPoint and binsSave:
		binsSave = False
		print ("Stop saving bin")
		return
	if gridVal != trigPoint and binsSave:
		if len(binsDict[bin_number]) == 0:
			binsDict[bin_number].append(gridVal)
			# print "Before send"
			send_bins_geom(curr_res,originPosition)
			return
		# print binsDict[bin_number]
		gridB = binsDict[bin_number][-1]
		# print gridB
		min_x = gridVal[0] if gridVal[0] < gridB[0] else gridB[0]
		max_x = gridVal[0] if gridVal[0] > gridB[0] else gridB[0]
		min_y = gridVal[1] if gridVal[1] < gridB[1] else gridB[1]
		max_y = gridVal[1] if gridVal[1] > gridB[1] else gridB[1]
		for i in range(min_x, max_x+1):
			for j in range(min_y, max_y+1):
				binsDict[bin_number].append((i,j))
		binsDict[bin_number].append(gridVal)
		send_bins_geom(curr_res,originPosition)
		return
	lastClick = gridVal


def send_obstacle_geom(obstacle_list, map_info):
	""" Provide to RVIZ the geometry of each obstacles and save the 
		obstacles in a file given by the user
	"""
	global DATA_SAVE_DIR
	global seq_obs
	global saveObstacle
	resolution = map_info.resolution
	z = map_info.origin.position.z

	rateSend = rospy.Rate(100)
	marker = Marker()
	marker.header.frame_id = "/odom"
	marker.header.stamp = rospy.Time.now()
	marker.ns = 'obstacles'
	marker.type = 1
	marker.action = 3
	obstaclePub.publish(marker)
	for i in range(5):
		rateSend.sleep()
	# Set attribute for markers
	marker.action = 0
	marker.pose.orientation.w  = 1.0
	marker.color.r = 1.0
	marker.color.g = 0.0
	marker.color.b = 0.0
	marker.color.a = 0.2
	nbTerm = 0
	#print len(obstacle_list)
	#print obstacle_list
	for (min_x,min_y,max_x,max_y) in obstacle_list:
		pos_min = grid2pos((min_x,min_y), map_info.origin.position, resolution)
		pos_max = grid2pos((max_x,max_y), map_info.origin.position, resolution)
		#print (pos_min,pos_max)
		marker.pose.position = Point((pos_min[0]+pos_max[0])/2.0,(pos_min[1]+pos_max[1])/2.0,z)
		marker.scale.x = (max_x-min_x+1)*resolution
		marker.scale.y = (max_y-min_y+1)*resolution
		marker.scale.z = 0.01
		marker.header.seq= seq_obs
		marker.id = nbTerm
		obstaclePub.publish(marker)
		rateSend.sleep()
		seq_obs = seq_obs + 1
		nbTerm = nbTerm + 1
	if not saveObstacle:
		return
	# Allow to save the map
	obstacle_file = raw_input("obstacle_file_name : ")
	obstacle_array = np.zeros((len(obstacle_list), 8))
	iterVal = 0
	for (min_x,min_y,max_x,max_y) in obstacle_list:
		pos_1 = grid2pos((min_x,min_y), map_info.origin.position, resolution)
		pos_2 = grid2pos((max_x,min_y), map_info.origin.position, resolution)
		pos_3 = grid2pos((max_x,max_y), map_info.origin.position, resolution)
		pos_4 = grid2pos((min_x,max_y), map_info.origin.position, resolution)
		obstacle_array[iterVal,:] = np.array([pos_1[0],pos_1[1],pos_2[0],pos_2[1],pos_3[0],pos_3[1],pos_4[0],pos_4[1]])
		iterVal = iterVal+1
	np.save(DATA_SAVE_DIR+obstacle_file, obstacle_array)


if __name__ == '__main__':
	# Parse command line arguments
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='./')
	parser.add_argument('--bin_unit_dim', type=float, default=5)
	parser.add_argument('--num_cells', type=float, default=80)
	args = parser.parse_args()

	# Save the data concerning the grid world
	DATA_SAVE_DIR = args.data_dir

	grid_map = OccupancyGrid()  	# Store the occupancy grid
	obstacle_geom = list()      	# Store the obstacles as (min_x,min_y,max_x,max_y)
	binsDict = dict()           	# Store the bins which represents the grid
	transVect = list()          	# Store the transition between ceels
	transPoints = list()        	# Store the points used for the transition

	transEnable = False         	# Enable RVIZ to show the transitions
	firstTime=True              	# Temporary variable for rviz interaction
	seq_obs = 0                 	# Counter for messages sent to RVIZ
	seq_bins = 0                	# Counter for messages sent to RVIZ
	seq_trans = 0               	# Counter for messages sent to RVIZ
	binsSave = False            	# Variable to save the bins in a file
	bin_number = 0              	# Count the number of bin registered so far
	lastClick = (0,0)           	# Save the last clicked cell

	saveObstacle = False        	# Activate saving the obstacles positions
	curr_res = args.bin_unit_dim	# Size of a cell in meter (m)
	nbCell = args.num_cells       	# Number of cell in the grid

	rospy.init_node('bins_obstacles_node', anonymous=True)
	# # Subscribe to the map topic
	rospy.Subscriber("/map", OccupancyGrid, grid2d_callback)
	# # Subscriber to the RVIZ click
	rospy.Subscriber("/clicked_point", PointStamped, binsLoc_callback)
	# Publisher of the obstacles
	obstaclePub = rospy.Publisher("/obs_loc", Marker, queue_size=100)
	# Publisher of the bins
	binsPub = rospy.Publisher("/bins_loc", Marker, queue_size=100)
	transPub = rospy.Publisher("/trans_loc", Marker, queue_size=100)

	rospy.spin()
