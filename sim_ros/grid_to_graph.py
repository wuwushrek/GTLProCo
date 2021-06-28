import numpy as np
import json
from geometry_msgs.msg import Point

def grid2pos(pos, origin, resolution):
	""" Return a continuous position (x,y)
		corresponding to a discrete position (i,j) in a grid characterized
		by the origin 'origin' and the cell resolution 'resolution'
		:param pos : a tuple (i,j) such that i represents the index on the x axis
						and j is the index on the y axis
		:param origin : a tuple (pos_x, pos_y) that represents the continuous
						coordinate of the origin of the gridworld which corresponds
						to (0,0) the lower left point in the gridworld
		:param resolution : Dimension of a cell in the grid
	"""
	new_x = pos[0] * resolution + resolution/2.0
	new_y = pos[1] * resolution + resolution/2.0
	return (new_x + origin.x, new_y + origin.y)


def pos2grid(val, origin, resolution):
	"""  Return a discrete position (i,j) in a gridworld
		to teh corresponding continuous position in the world
		:param val : a tuple (pos_x,pos_y) such that pos_x and pos_y represent
						the x and y axis continuous coordinate
		:param origin : a tuple (pos_x, pos_y) that represents the continuous
						coordinate of the origin of the gridworld which corresponds
						to (0,0) the lower left point in the gridworld
		:param resolution : Dimension of a cell in the grid
	"""
	val_x = int((val[0] - origin.x - resolution/2.0) / resolution)
	val_y = int((val[1] - origin.y - resolution/2.0) / resolution)
	return val_x,val_y

def getNeighbors(pos, width, height):
	""" Given a discrete pos=(i,j) in a gridworld, return the neighbor
		cells of that position
		:param pos : a tuple (i,j) such that i represents the index on the x axis
						and j is the index on the y axis
		:param width : The number of cells in the x direction
		:param height : The number ofc ells in the y direction
	"""
	listRes = set()
	if pos[0]+1 < width:
		listRes.add((pos[0]+1,pos[1]))
	if pos[0]-1 >= 0:
		listRes.add((pos[0]-1,pos[1]))
	if pos[1]+1 < height:
		listRes.add((pos[0],pos[1]+1))
	if pos[1]-1 >= 0:
		listRes.add((pos[0],pos[1]-1))
	return listRes

def originGrid(width, heigh, cell_dim):
	""" Return the continuous coordinate of the (0,0) index in a 
		grid world. THe coordinate corresponds to the lower left
		point in the grid
		:param width : the number of cells along the x axis
		:param height : the number of cells along the y axis
		:param cell_dim : the dimension of each cell
	"""
	return Point(-(width*cell_dim)/2, -(width*cell_dim)/2, 0.0)

def from_grid_to_bin_repr(width, height, bin_unit_dim, num_subdiv, 
								obstacles=set(), regroupCell=dict()):
	""" GIven a grid with its number of cells along the x axis
		given by with and the number of cells along the y axis
		given by height, compute the corresponding bins and
		and transition between bins
		:param width : the number of cells on the x axis
		:param height : the number of cells on the y axis
		:param bin_unit_dim : the dimension of a subdivision. A subdivision
			is a subcell of a cell that can contain a robot
		:param num_subdiv : the number of sub-divisions in each cells. It
			also characterizes the maximum number of agent that can reside
			in a cell.
		:param regroupCell : A dict with index keys (i,j) and which value is the
					set of index (k,l) such that (i,j) and (k,l) represent the same bin
					A key should not be seen inside values of any keys (including its own value)
	"""
	newRegroup = dict()
	# Perform some sanity check on regroupCell and sort the keys
	firstIter = True
	lastSetVal = None
	for (i,j), setVal in regroupCell.items():
		newSetVal = setVal.union(set([(i,j)]))
		# print(newSetVal)
		if not firstIter:
			assert newSetVal.isdisjoint(lastSetVal), 'Key and values pairs are not disjoint between each other'
		newOrder = sorted(newSetVal)
		newRegroup[newOrder[0]] = set(newOrder[1:])
		lastSetVal = newSetVal
		if firstIter:
			firstIter = False

	# Main body
	resolution = bin_unit_dim*num_subdiv
	origin = originGrid(width, height, resolution)
	binsDict = dict()
	binsDictFull = dict()
	alreadySeen = set()
	counterVal = 0
	tempDict = dict()
	for i in range(width):
		for j in range(height):
			if (i,j) in alreadySeen:
				continue
			bin_number = counterVal
			counterVal += 1
			binsDict[bin_number] = list()
			regroup = newRegroup.get((i,j), set())
			regroup.add((i,j))
			for (x,y) in regroup:
				binsDictFull[(x,y)] = list()
				alreadySeen.add((x,y))
				tempDict[(x,y)] = bin_number
				x_true = x * num_subdiv
				y_true = y * num_subdiv
				for k in range(num_subdiv):
					for l in range(num_subdiv):
						actualPos = grid2pos((x_true+k,y_true+l), origin, bin_unit_dim)
						binsDict[bin_number].append(actualPos)
						binsDictFull[(x,y)].append(actualPos)

	# Handle the transitions between the bins
	alreadySeen = set()
	transVect = set()
	transPoints = list()
	for i in range(width):
		for j in range(height):
			if (i,j) in alreadySeen:
				continue
			regroup = newRegroup.get((i,j), set())
			regroup.add((i,j))
			for (x,y) in regroup:
				alreadySeen.add((x,y))
				x_true = x * num_subdiv
				y_true = y * num_subdiv
				posxy = grid2pos((x_true,y_true), origin, bin_unit_dim)
				bin_number = tempDict[(x,y)]
				transVect.add((bin_number,bin_number))
				neighborList = getNeighbors((x,y), width, height)
				for (k,l) in neighborList:
					k_true = k * num_subdiv
					l_true = l * num_subdiv
					binN = tempDict[(k,l)]
					poskl = grid2pos((k_true,l_true), origin, bin_unit_dim)
					transVect.add((binN,binN))
					transVect.add((bin_number, binN))
					transVect.add((binN, bin_number))
					transPoints.append((posxy, poskl))

	# Handle the obstacles
	obstacle_array = np.zeros((1, 8))
	for (x,y) in obstacles:
		# nid_bin = tempDict[(x,y)]
		min_x, max_x, min_y, max_y = np.inf,-np.inf,np.inf,-np.inf
		# print(x,y)
		for (posij_x,posij_y) in binsDictFull[(x,y)]:
			# posij_x, posij_y = grid2pos((i,j), origin, resolution)
			# print(posij_x,posij_y)
			min_x = np.minimum(posij_x, min_x)
			max_x = np.maximum(max_x, posij_x)
			min_y = np.minimum(posij_y, min_y)
			max_y = np.maximum(max_y, posij_y)
		pos_1 = (min_x,min_y)
		pos_2 = (max_x,min_y)
		pos_3 = (max_x,max_y)
		pos_4 = (min_x,max_y)
		arrayPos = np.array([pos_1[0],pos_1[1],pos_2[0],pos_2[1],pos_3[0],pos_3[1],pos_4[0],pos_4[1]])
		# print(arrayPos)
		obstacle_array = np.concatenate((obstacle_array, arrayPos.reshape(1,-1)))

	return binsDict, transVect, transPoints, (obstacle_array[1:,:] if obstacles is not None else None), tempDict



if __name__ == '__main__':

	# ################# Gridworld example 1 ##################
	# nrows = 6
	# ncols = 6
	# bin_unit_dim = 1
	# nSubDiv = 5
	# obstacles = set([(2,1), (3,1), (1,2), (1,3), (2,4), (4,2), (4,3), (4,4)])
	# regroupCell = dict()
	# regroupCell[(2,2)] = set([(2,3), (3,2), (3,3)])
	# binsDict, transVect, transPoints, obsV, coordToBin = \
	# 	from_grid_to_bin_repr(nrows, ncols, bin_unit_dim, nSubDiv, obstacles=obstacles, regroupCell=regroupCell)
	# # Save the example
	# np.save('obstacles', obsV)
	# dictResult = dict()
	# dictResult['bin_unit_dim'] = bin_unit_dim
	# dictResult['n_sub_div'] = nSubDiv
	# dictResult['bins'] = binsDict
	# dictResult['trans'] = list(transVect)
	# dictResult['transTrue'] = transPoints
	# dictResult['coordToBin'] = {'{}'.format(t): v for t,v in coordToBin.items()}
	# dictResult['obstacles'] = [x for x in obstacles]
	# # print(dictResult)
	# with open('bins_distrib.json', 'w') as fp:
	# 	json.dump(dictResult, fp)

	################ Gridworld example 2 ###################
	nrows = 6
	ncols = 6
	bin_unit_dim = 0.5
	nSubDiv = 17
	obstacles = set([(0,1),(1,1),(4,1),(5,1),(1,3),(2,3),(3,4),(4,3),(1,4),(4,5)])
	regroupCell = dict()
	binsDict, transVect, transPoints, obsV, coordToBin = \
		from_grid_to_bin_repr(nrows, ncols, bin_unit_dim, nSubDiv, obstacles=obstacles, regroupCell=regroupCell)
	# Save the example
	np.save('obstacles', obsV)
	dictResult = dict()
	dictResult['bin_unit_dim'] = bin_unit_dim
	dictResult['n_sub_div'] = nSubDiv
	dictResult['bins'] = binsDict
	dictResult['trans'] = list(transVect)
	dictResult['transTrue'] = transPoints
	dictResult['coordToBin'] = {'{}'.format(t): v for t,v in coordToBin.items()}
	dictResult['obstacles'] = [x for x in obstacles]
	# print(dictResult)
	with open('bins_distrib.json', 'w') as fp:
		json.dump(dictResult, fp)