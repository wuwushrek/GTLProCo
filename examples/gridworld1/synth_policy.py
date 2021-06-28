#!/usr/bin/env python
import sys
import rospy

import numpy as np
import random 
import math

import json

from nav_msgs.msg import OccupancyGrid

from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker

from gazebo_msgs.msg import ModelState

from gazebo_msgs.srv import SpawnModel
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import SetModelState

from nav_msgs.msg import Odometry

from ast import literal_eval as make_tuple

import rvo2

def getObstaclesData(verticesArr):
	""" Given vertexes of obstalces, add the obstacles to and
		obstacle avoidance algorithm and return the added
		obstacles
	"""
	global sim          # The obstacle avoidance simulation object
	global binUnitDim   # Each bin dimension

	resList = list()
	valBinExtra = binUnitDim/4.0
	for i in range(verticesArr.shape[0]):
		listSub = [ (-valBinExtra,   -valBinExtra),
					(valBinExtra,    -valBinExtra),
					(valBinExtra,    valBinExtra),
					(-valBinExtra,   valBinExtra)]
		obsList = [(verticesArr[i,j]+sub[0],
					verticesArr[i,j+1]+sub[1]) \
						for j, sub in zip([0,2,4,6], listSub)]
		sim.addObstacle(obsList)    # Add the obstacle to the sim
		resList.append(obsList)     # Save the obstacle
	sim.processObstacles()          # Process the obstacles
	return resList

def getCenter(pos):
	""" Get the center of points representing a bin
	"""
	global binUnitDim

	minX = 500
	maxX = -500
	minY = 500
	maxY = -500
	for centerCube in pos:
		minX = min(minX, centerCube[0])
		minY = min(minY, centerCube[1])
		maxX = max(maxX, centerCube[0])
		maxY = max(maxY, centerCube[1])
	return ((minX +maxX)/2.0, (minY+maxY)/2.0,
				maxX-minX + binUnitDim, maxY-minY + binUnitDim)

def getBins(binsConf):
	""" Get the bin configuration from a dictionary load from a file
	"""
	global binsVir  # Contains a refined discretization of the grid for location of each agent

	dictDist = binsConf['bins']
	binsDict = dict()
	for elem, pos in dictDist.items():
		binsVir[int(elem)] = [(x[0],x[1]) for x in pos]
		binsDict[int(elem)] = [getCenter(pos)]
	transBin = [(val[0], val[1]) for val in binsConf['trans']]
	transPoint = [((val[0][0],val[0][1]),(val[1][0],val[1][1])) \
						for val in binsConf['transTrue']]
	return binsDict, transBin, transPoint

################### RVIZ visualization functions ##############################
def sendBins(bin_label=True):
	""" Draw the bins configuration in RVIZ
	"""
	global binUnitDim
	global binsVir
	global binsDict
	global cellsColor
	global subDivColor
	global binsTextColor
	global agentRadius
	global binsPub
	global obstaclesBin

	rateSend = rospy.Rate(100)
	marker = Marker()
	marker.header.frame_id = "odom"
	marker.header.stamp = rospy.Time.now()
	marker.ns = 'bins'
	marker.type = 1
	# First Delete old Markers
	marker.action = 3
	binsPub.publish(marker)
	for i in range(50):
		rateSend.sleep()
	# Then Add the bins Marker
	marker.action = 0
	marker.pose.orientation.w  = 1.0
	marker.scale.x = binUnitDim*0.8
	marker.scale.y = binUnitDim*0.8
	marker.scale.z = 0.001
	marker.color.a = 0.7
	nbElement = 0
	for (key, binsVal) in binsDict.items():
		if key in obstaclesBin:
			continue
		marker.color.r = cellsColor[key][0]
		marker.color.g = cellsColor[key][1]
		marker.color.b = cellsColor[key][2]
		for elem in binsVal:
			# RVIZ
			marker.scale.x = elem[2] * 0.99
			marker.scale.y = elem[3] * 0.99
			marker.pose.position = Point(elem[0],elem[1],0)
			marker.header.seq = nbElement
			marker.header.stamp = rospy.Time.now()
			marker.id = nbElement
			nbElement = nbElement + 1
			binsPub.publish(marker)
			rateSend.sleep()

	# # Send the subdivision inside the cells/bins
	# marker.scale.z = 0.001
	# marker.color.r = subDivColor[0]
	# marker.color.g = subDivColor[1]
	# marker.color.b = subDivColor[2]
	# marker.color.a = 0.7
	# marker.scale.x = binUnitDim
	# marker.scale.y = binUnitDim
	# for key, binsVal in binsVir.items():
	#   for elem in binsVal:
	#       marker.pose.position = Point(elem[0],elem[1],0)
	#       marker.header.seq = nbElement
	#       marker.header.stamp = rospy.Time.now()
	#       marker.id = nbElement
	#       nbElement = nbElement + 1
	#       binsPub.publish(marker)
	#       rateSend.sleep()

	# Send the text labels for the bins
	if not bin_label:
		return
	marker.type = 9
	marker.scale.z = (1/binUnitDim) * 1.3
	marker.color.r = binsTextColor[0]
	marker.color.g = binsTextColor[1]
	marker.color.b = binsTextColor[2]
	marker.color.a = 0.7 # 1
	marker.scale.x = (1/binUnitDim) * 2.2
	marker.scale.y = (1/binUnitDim) * 2.2
	for key, binsVal in binsDict.items():
		ranPos = random.choice(binsVal)
		marker.pose.position = Point(ranPos[0],ranPos[1],0)
		marker.text =  "Bin " + str(key)
		marker.id = nbElement
		marker.header.seq = nbElement
		marker.header.stamp = rospy.Time.now()
		nbElement = nbElement + 1
		binsPub.publish(marker)
		rateSend.sleep()

def sendObstacles():
	""" Show obstacles in RVIZ for visualization
	"""
	global obsPub
	global obstaclesList
	global nbSubDiv

	rateSend = rospy.Rate(100)
	marker = Marker()
	marker.header.frame_id = "odom"
	marker.header.stamp = rospy.Time.now()
	marker.ns = 'obstacles'
	marker.type = 1
	# First Delete old Markers
	marker.action = 3
	obsPub.publish(marker)
	for i in range(50):
		rateSend.sleep()
	# Then Add the bins Marker
	marker.action = 0
	marker.pose.orientation.w  = 1.0
	marker.scale.z = 0.001
	marker.color.r = 1.0
	marker.color.g = 0.0
	marker.color.b = 0.0
	marker.color.a = 1.0
	nbElement = 0
	for elem in obstaclesList:
		marker.scale.x = binUnitDim * nbSubDiv * 0.99
		marker.scale.y = binUnitDim * nbSubDiv * 0.99
		marker.pose.position = Point((elem[0][0]+elem[1][0])/2.0, (elem[1][1]+elem[2][1])/2.0,0)
		marker.header.seq = nbElement
		marker.header.stamp = rospy.Time.now()
		marker.id = nbElement
		nbElement = nbElement + 1
		obsPub.publish(marker)
		rateSend.sleep()

def sendTransitions():
	""" Show the transitions between bins in RVIZ for visualization
	"""
	global transPoint
	global binUnitDim
	global transPub
	global transColor

	rateSend = rospy.Rate(100)
	marker = Marker()
	marker.header.frame_id = "odom"
	marker.header.stamp = rospy.Time.now()
	marker.ns = 'transitions'
	marker.type = 4
	# First Delete old Markers
	marker.action = 3
	obsPub.publish(marker)
	for i in range(50):
		rateSend.sleep()
	# Then Add the bins Marker
	marker.action = 0
	marker.pose.orientation.w  = 1.0
	marker.scale.x = 0.1 * binUnitDim
	marker.scale.y = 0.1 * binUnitDim
	marker.scale.z = 0.1
	marker.color.a = 0.3
	marker.color.r = transColor[0]
	marker.color.g = transColor[1]
	marker.color.b = transColor[2]
	nbElement = 0
	for (pt1, pt2) in transPoint:
		marker.points = [Point(pt1[0],pt1[1],0), Point(pt2[0],pt2[1],0)]
		marker.header.seq = nbElement
		marker.header.stamp = rospy.Time.now()
		marker.id = nbElement
		nbElement = nbElement + 1
		transPub.publish(marker)
		rateSend.sleep()

def sendRobotPosVel(pos, vel, firstTime=False):
	""" Broadcast the robot position in RVIZ
	"""
	global robotSequence
	global agentRadius
	global robotPub
	global robotColor

	rateSend = rospy.Rate(1000)
	marker = Marker()
	marker.header.frame_id = "odom"
	marker.header.stamp = rospy.Time.now()
	marker.ns = 'robots'
	marker.action = 0
	marker.pose.orientation.w  = 1.0
	marker.color.a = 1.0
	nbElement = 0
	for p , v in zip(pos, vel):
		marker.color.r = robotColor[nbElement][0]
		marker.color.g = robotColor[nbElement][1]
		marker.color.b = robotColor[nbElement][2]
		marker.type = 2
		marker.scale.x = agentRadius
		marker.scale.y = agentRadius
		marker.scale.z = agentRadius
		marker.pose.position = Point(p[0],p[1],0)
		marker.points = []
		marker.pose.orientation.w  = 1.0
		marker.id = nbElement
		marker.header.seq = robotSequence
		marker.header.stamp = rospy.Time.now()
		robotPub.publish(marker)
		robotSequence = robotSequence+1
		# marker.type = 0
		# marker.pose = Pose()
		# marker.scale.x = marker.scale.x / 1.5
		# marker.scale.y = marker.scale.y / 1.5
		# marker.scale.z = marker.scale.x * 1.5
		# marker.points = [Point(p[0],p[1],0), Point(p[0]+v[0]*4,p[1]+v[1]*4,0)]
		# marker.id = nbElement + nbAgents
		# marker.header.seq = robotSequence
		# marker.header.stamp = rospy.Time.now()
		# robotPub.publish(marker)
		# robotSequence = robotSequence+1
		nbElement = nbElement + 1
		rateSend.sleep()

################## SWARM UTILITY FUNCTIONS ############################

def compute_V_des(X, goal, V_max, lastSpeed, distGoal=0.5):
	""" Compute the desired speed of each agent given
		the current position, the goal and the maximum allowed velocity
	"""
	V_des = []
	countReach = 0
	for curr , targ, vSpeed  in zip(X, goal, lastSpeed): 
		dif_x = [targ[k]-curr[k] for k in range(2)]
		norm = distance(dif_x, [0, 0])
		norm_dif_x = [dif_x[k]*V_max[k]/norm for k in range(2)]
		reachGoal = reach(curr, targ, distGoal)
		if reachGoal:
			countReach += 1
			V_des.append((0,0))
		else:
			if vSpeed is None:
				V_des.append((norm_dif_x[0], norm_dif_x[1]))
			else:
				V_des.append(vSpeed)
	return V_des, countReach == len(X)

def reach(p1, p2, bound=0.5):
	""" Check if the two points p1 and p2 are distant from each 
		other by less than 'bound'
	"""
	if distance(p1,p2)< bound:
		return True
	else:
		return False


def distance(pose1, pose2):
	""" Compute Euclidean distance for 2D 
	"""
	return np.sqrt((pose1[0]-pose2[0])**2+(pose1[1]-pose2[1])**2)+0.001


def updateGoal(tIndex):
	""" Update the goal, i.e. the target position of each robot
	"""
	global robotTimePosition
	return robotTimePosition[:,tIndex]

def updateEfficientPosition(agentDist, currentDist, goalPos):
	""" More efficient version of getRobotPosition, where we don't
		relocate the agent that are not moving from their current bin
	"""
	global binsVir
	newBinsVir = dict(binsVir)
	for nextBin, currBin, gPos in zip(agentDist, currentDist, goalPos):
		if nextBin == currBin:
			newList = list()
			for elem in newBinsVir[nextBin]:
				if distance(elem, gPos) > 0.001:
					newList.append(elem)
			newBinsVir[nextBin] = newList
	occupancyAgent = dict()
	robotPos = dict()
	for i in range(len(binsDict)):
		occupancyAgent[i] = [0 , list()]
	for i, elem in enumerate(agentDist):
		if currentDist[i] == elem:
			continue
		occupancyAgent[elem][0] = occupancyAgent[elem][0] + 1
		occupancyAgent[elem][1].append(i)
	for key, val in occupancyAgent.items():
		newPos = random.sample(newBinsVir[key] , k=val[0])
		for robId, robPos in zip(val[1], newPos):
			robotPos[robId] = robPos
	currPos = [robotPos.get(i, j) for i, j in enumerate(goalPos)]
	return currPos

def getRobotPosition(agentDist):
	""" Get position of each robot given a distribution of
		the robots over the bins
	"""
	global binsVir
	occupancyAgent = dict()
	robotPos = dict()
	for i in range(len(binsDict)):
		occupancyAgent[i] = [0 , list()]
	for i, elem in enumerate(agentDist):
		occupancyAgent[elem][0] = occupancyAgent[elem][0] + 1
		occupancyAgent[elem][1].append(i)
	for key, val in occupancyAgent.items():
		newPos = random.sample(binsVir[key] , k=val[0])
		for robId, robPos in zip(val[1], newPos):
			robotPos[robId] = robPos
	currPos = [robotPos[i] for i in range(nbAgents)]
	return currPos

def checkRobotReachBin(currPos, targetBin):
	""" Check if each robot reach their target bins
	"""
	countCorrect = 0
	for cPos, tBin in zip(currPos, targetBin):
		binsDicCenter = binsDict[tBin][0]
		if cPos[0] >= binsDicCenter[0] - binsDicCenter[2]/2.0 and cPos[1] >= binsDicCenter[1] - binsDicCenter[3]/2.0 and cPos[0] <= binsDicCenter[0] + binsDicCenter[2]/2.0 and cPos[1] <= binsDicCenter[1] + binsDicCenter[3]/2.0:
			countCorrect += 1
	return countCorrect == targetBin.shape[0]

# Get the optimal velocity based on current velocity and goal
def updateVelocity(pos, goal, vel, distGoal=0.5):
	""" Get the optimal velocity based on the current position, the current
		velocity, and the desired goal
	"""
	global VMAX
	global robotPosition
	global robotVelocity
	global obstacleSeen
	global sim
	V_des, rGoal = compute_V_des(pos, goal, VMAX, obstacleSeen, distGoal)
	NewobstacleSeen = []
	for i, v_des, rPos, rVel in zip(range(nbAgents), V_des, robotPosition, robotVelocity):
		sim.setAgentPosition(i, rPos)
		sim.setAgentVelocity(i, rVel)
		sim.setAgentPrefVelocity(i, v_des)
	sim.doStep()
	nextV = [sim.getAgentVelocity(i) for i in range(nbAgents)]
	# Hack to ease obstacle avoidance
	for desV, nV, obsSeen, currV in zip (V_des, nextV, obstacleSeen, robotVelocity):
		if np.abs(desV[0]) <= 1e-14 or np.abs(desV[1]) <= 1e-14:
			NewobstacleSeen.append(None)
		elif (np.abs(nV[0]/desV[0]) <= 1e-1 or np.abs(nV[1]/desV[1]) <= 1e-1) and obsSeen is None:
			NewobstacleSeen.append((np.sign(desV[0])*VMAX[0], np.sign(desV[1])*VMAX[1]))
		elif np.abs(nV[0]/ desV[0]) <= 1e-1 or np.abs(nV[1]/desV[1]) <= 1e-1:
			NewobstacleSeen.append(obsSeen)
		else:
			if np.abs(nV[0]) <= VMAX[0]/3.0 and obsSeen is not None:
				NewobstacleSeen.append(obsSeen)
			elif np.abs(nV[1]) <= VMAX[0]/3.0 and obsSeen is not None:
				NewobstacleSeen.append(obsSeen)
			else:
				NewobstacleSeen.append(None)
	obstacleSeen = NewobstacleSeen
	return nextV, rGoal

def NoCollisionupdateVelocity(pos, goal, vel, distGoal=0.5):
	""" Get the optimal velocity based on the current position, the current
		velocity, and the desired goal
	"""
	global VMAX
	global robotPosition
	global robotVelocity
	global obstacleSeen
	global sim
	V_des, rGoal = compute_V_des(pos, goal, VMAX, obstacleSeen, distGoal)
	# # NewobstacleSeen = []
	# # for i, v_des, rPos, rVel in zip(range(nbAgents), V_des, robotPosition, robotVelocity):
	# # 	sim.setAgentPosition(i, rPos)
	# # 	sim.setAgentVelocity(i, rVel)
	# # 	sim.setAgentPrefVelocity(i, v_des)
	# # sim.doStep()
	# # nextV = [sim.getAgentVelocity(i) for i in range(nbAgents)]
	# # Hack to ease obstacle avoidance
	# for desV, nV, obsSeen, currV in zip (V_des, nextV, obstacleSeen, robotVelocity):
	# 	if np.abs(desV[0]) <= 1e-14 or np.abs(desV[1]) <= 1e-14:
	# 		NewobstacleSeen.append(None)
	# 	elif (np.abs(nV[0]/desV[0]) <= 1e-1 or np.abs(nV[1]/desV[1]) <= 1e-1) and obsSeen is None:
	# 		NewobstacleSeen.append((np.sign(desV[0])*VMAX[0], np.sign(desV[1])*VMAX[1]))
	# 	elif np.abs(nV[0]/ desV[0]) <= 1e-1 or np.abs(nV[1]/desV[1]) <= 1e-1:
	# 		NewobstacleSeen.append(obsSeen)
	# 	else:
	# 		if np.abs(nV[0]) <= VMAX[0]/3.0 and obsSeen is not None:
	# 			NewobstacleSeen.append(obsSeen)
	# 		elif np.abs(nV[1]) <= VMAX[0]/3.0 and obsSeen is not None:
	# 			NewobstacleSeen.append(obsSeen)
	# 		else:
	# 			NewobstacleSeen.append(None)
	# obstacleSeen = NewobstacleSeen
	return V_des, rGoal

def commandVelocity(vNext):
	global robotPosition, robotVelocity, stepTime
	robotVelocity = vNext
	robotPosition = [ (px+stepTime*vx, py+stepTime*vy) for (px,py), (vx,vy) in zip(robotPosition, robotVelocity)]

##########################################################################

if __name__ == '__main__':
	# Parse command line arguments
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='./')
	parser.add_argument('--bins_name', type=str, default='bins_distrib.json')
	parser.add_argument('--obstacle_name', type=str, default='obstacles.npy')
	parser.add_argument('--swarm_traj', type=str, default='robotTraj.npy')
	parser.add_argument('--update_rate', type=int, default=5)
	parser.add_argument('--agent_radius', type=float, default=5)
	parser.add_argument('--vmax', type=float, default=3)
	parser.add_argument('--seed', type=int, default=101)
	# parser.add_argument('--num_episodes', type=int, default=100)
	# parser.add_argument('--num_run', type=int, default=200)
	# parser.add_argument('--id_force', type=ast.literal_eval, default={})
	# parser.add_argument('--coeff_u', type=float, default=1)
	args = parser.parse_args()

	# Set randomness seed for reproducibility
	# np.random.seed(args.seed)
	# random.seed(args.seed)

	########## Obtain the bins information and obstacles ###############
	DATA_SAVE_DIR = args.data_dir
	bins_name = args.bins_name
	obs_name = args.obstacle_name
	robot_pos_file = args.swarm_traj

	# Read the bins configuration
	with open(DATA_SAVE_DIR + bins_name, 'r') as fp:
		binsConf = json.load(fp)

	# Read the description of the obstacles present 
	verticesArr = np.load(DATA_SAVE_DIR + obs_name)

	# Total number of bins
	nBins = len(binsConf['bins'])
	obstaclesSet = binsConf['obstacles']
	coordToBin = dict()
	for key, value in binsConf['coordToBin'].items():
		coordToBin[make_tuple(key)] = value
	obstaclesBin = set()
	for key in obstaclesSet:
		key_val  = (key[0], key[1])
		obstaclesBin.add(coordToBin[key_val])
	print('Dim unit bins: ', binsConf['bin_unit_dim'])
	print('Number of subdivision cells: ', binsConf['n_sub_div'])

	# # Obtain the time evolution of the robot position in the bin coordinate
	robotTimePosition = np.load(DATA_SAVE_DIR + robot_pos_file)
	# robotTimePosition = np.zeros((10,1))

	####################################################################

	# Counter for ROS messages
	robotSequence = 0

	# Initialize the Simulation node
	rospy.init_node('swarm_simu', anonymous=True)

	# Create the robot marker publisher
	robotPub = rospy.Publisher('/robot_position', Marker, queue_size=1000)

	# Create the  bins publisher
	binsPub = rospy.Publisher('/bins_loc', Marker, queue_size=50)

	# Create the transition publisher
	transPub = rospy.Publisher('/trans_loc', Marker, queue_size=50)

	# Create a publisher for the obstacles
	obsPub = rospy.Publisher('/obs_loc', Marker, queue_size=50)

	################### Multi Agent Settings ##########################
	nbAgents = robotTimePosition.shape[0]
	updateRate = args.update_rate
	stepTime = 1.0/updateRate
	binUnitDim = float(binsConf['bin_unit_dim'])
	agentRadius = binUnitDim/1.25
	nbSubDiv = float(binsConf['n_sub_div'])
	VMAX = [args.vmax for i in range(nbAgents)]
	rosRate = rospy.Rate(updateRate)
	binsVir = dict()
	agentPublisher = dict()
	dictPosVel = dict()

	################## Define the RVO sim model ######################
	# timeStep:         The time step of the simulation. Must be positive.
	# neighborDist:     The default maximum distance (center point to center point) to other agents a new agent takes into account in the navigation. The larger this number, the longer he running time of the simulation. If the number is too low, the simulation will not be safe. Must be non-negative.
	# maxNeighbors:     The default maximum number of other agents a new agent takes into account in the navigation. The larger this number, the longer the running time of the simulation. If the number is too low, the simulation will not be safe.
	# timeHorizon:      The default minimal amount of time for which a new agent's velocities that are computed by the simulation are safe with respect to other agents. The larger this number, the sooner an agent will respond to the presence of other agents, but the less freedom the agent has in choosing its velocities. Must be positive.
	# timeHorizonObst:  The default minimal amount of time for which a new agent's velocities that are computed by the simulation are safe with respect to obstacles. The larger this number, the sooner an agent will respond to the presence of obstacles, but the less freedom the agent has in choosing its velocities. Must be positive.
	# radius:           The default radius of a new agent. Must be non-negative.
	# maxSpeed:         The default maximum speed of a new agent. Must be non-negative.
	# velocity:         The default initial two-dimensional linear velocity of a new agent (optional).
	sim = rvo2.PyRVOSimulator(stepTime, 2.0*agentRadius, 6, 5*stepTime, 5*stepTime,
									agentRadius,VMAX[0], (0,0))
	# Add each agent in the RVO simulation
	for i in range(nbAgents):
		sim.addAgent((0,0))

	################### Visualization settings for RVIZ ###############
	gC = [0,1,0]
	rC = [1,0,0]
	bC = [0,0,1]
	# robotColor = [[0,0,1] for i in range(nbAgents)] # 
	robotColor = [np.random.random(3) for i in range(nbAgents)]
	targetColor = gC
	initColor = [0,1,1]
	transColor = np.random.random(3)
	cellC = np.random.random(3)
	# cellC = [0.46323047, 0.58289193, 0.0845086 ]
	# cellC = [0.5644569, 0.58567775, 0.89103187]
	cellC = [0.16059486, 0.71912574, 0.72425261]
	print(cellC)
	cellsColor = [cellC for i in range(nBins)]
	subDivColor = np.random.random(3)
	binsTextColor = [0, 0, 0]
	# Set the target colors
	targetSet = [16, 17, 23, 28]
	for val in targetSet:
		cellsColor[val] = targetColor
	initSet = [0, 6, 24, 30]
	for val in initSet:
		cellsColor[val] = initColor

	obstaclesList = getObstaclesData(verticesArr) # Obtain the obstacles
	binsDict, transBin, transPoint = getBins(binsConf)

	sendBins(bin_label=True)
	sendTransitions()
	sendObstacles()

	# Send the initial position of the Robot in the gridworld
	currentBin = robotTimePosition[:,0]
	robotPosition = getRobotPosition(currentBin)
	for i , robotPos in enumerate(robotPosition):
		sim.setAgentPosition(i , robotPos)
	robotVelocity = [(0,0) for i in range(nbAgents)]
	obstacleSeen = [None for i in range(nbAgents)]
	sendRobotPosVel(robotPosition, robotVelocity)

	# Send the rest of the trajectory
	tIndex = 1
	trajLength = robotTimePosition.shape[1]
	targetBin = updateGoal(tIndex)
	targetPos = updateEfficientPosition(targetBin, currentBin, robotPosition)
	while tIndex <= trajLength:
		nextV, rGoal = NoCollisionupdateVelocity(robotPosition, targetPos, robotVelocity)
		if rGoal or checkRobotReachBin(robotPosition, targetBin):
			print (tIndex)
			tIndex += 1
			currentBin = targetBin
			targetBin = updateGoal(tIndex)
			targetPos = updateEfficientPosition(targetBin, currentBin, robotPosition)
		commandVelocity(nextV)
		sendRobotPosVel(robotPosition, robotVelocity)
	# sendRobotPosVel(robotPosition, robotVelocity)
