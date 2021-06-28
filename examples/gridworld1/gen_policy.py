import matplotlib
import matplotlib.pyplot as plt

from GTLProCo.LabelledGraph import LabelledGraph
from GTLProCo.GTLformula import *
from GTLProCo.gtlproco import *

import json
from ast import literal_eval as make_tuple

import tikzplotlib

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='./')
parser.add_argument('--bins_name', type=str, default='bins_distrib.json')

args = parser.parse_args()

DATA_SAVE_DIR = args.data_dir
bins_name = args.bins_name

# Read the bins configuration
with open(DATA_SAVE_DIR + bins_name, 'r') as fp:
	binsConf = json.load(fp)

# Parse the obstacle set and the bon to (x,y) coordinate
nBins = len(binsConf['bins'])
binsDistr = binsConf['bins']
transFun = binsConf['trans']
obstaclesSet = [(x[0], x[1]) for x in binsConf['obstacles']]
coordToBin = dict()
for key, value in binsConf['coordToBin'].items():
	coordToBin[make_tuple(key)] = value
obstaclesBin = set()
for key in obstaclesSet:
	key_val  = (key[0], key[1])
	obstaclesBin.add(coordToBin[key_val])

# Create the labelled graph
nodeSet = set([int(bV) for bV, val in binsDistr.items()])
mGraph = LabelledGraph(nodeSet)

# Add the transitions between nodes
s_id = 0 # Swarm id
mGraph.addSubswarm(s_id, set([(int(val[0]), int(val[1])) for val in binsConf['trans']]))

################## Define the label for the GTL specifications #################
# Define the capacity constraints specifications at the node n_id = 0
labelsDict = dict() 
labelsDict[0] = list()
for i, n_id in enumerate(nodeSet): # Define the upper node_ids bound constraints on the density distribution - capacity constraints
	labelsDict[0].append(getattr(mGraph, 'x_{}_{}'.format(s_id, n_id)))

# Define all the obstacles constraints as the label of node id 1
labelsDict[1] = list()
for i, n_id in enumerate(obstaclesBin):
	labelsDict[1].append(getattr(mGraph, 'x_{}_{}'.format(s_id, n_id)))

# Define the target constraints as the label of node id 2
targetNodes = [16, 17, 23, 28]
labelsDict[2] = list()
# labelsDict[2].append(-(mGraph.x_0_16+mGraph.x_0_17+mGraph.x_0_23+mGraph.x_0_28))
labelsDict[2].extend([-mGraph.x_0_16,-mGraph.x_0_17,-mGraph.x_0_23,-mGraph.x_0_28])

# Add the labels
for n_id, lLabel in labelsDict.items():
	mGraph.addNodeLabel(n_id, lLabel)

# Swarm and graph configuration information
swarm_ids = mGraph.eSubswarm.keys()
node_ids = mGraph.V.copy()

# print(mGraph)
################# Define the GTL formula over the label ###################

# Capacity constraints per bin
initState = [0,6,24,30]
cap = 0.15 # For a CAP OF 0.3
realCap = 0.2 # This takes into account the quantized error due to finite amount of agents
capGTL = AlwaysGTL(AtomicGTL([ (0.25 if (n_id in initState or n_id in targetNodes) else cap) for i, n_id in enumerate(nodeSet) ]))

# Constraints specifications
obstacleGTL = AlwaysGTL(AtomicGTL([ 0 for i, n_id in enumerate(obstaclesBin) ]))

# target destination constraints -(mGraph.x_0_0+mGraph.x_0_7+mGraph.x_0_27+mGraph.x_0_22) <= -1
lbTarget = 0.2
targetGTLatomic = EventuallyAlwaysGTL(AtomicGTL([-lbTarget,-lbTarget,-lbTarget,-lbTarget]))
# targetGTLatomic = EventuallyAlwaysGTL(AtomicGTL([-0.8]))

# Time horizom
Kp = 15
# Create the MILP representation of the constraints
m_milp = create_milp_constraints([capGTL, obstacleGTL, targetGTLatomic], [0,1,2], Kp, mGraph)
# m_milp = create_milp_constraints([obstacleGTL, targetGTLatomic], [1,2], Kp, mGraph)
# m_milp = create_milp_constraints([capGTL, obstacleGTL], [0,1], Kp, mGraph)


# print(m_milp)

# Define the intial density of the swarm
initPoint = { 0 : {n_id : (1.0/len(initState) if n_id in initState else 0) for n_id in nodeSet} }
# initPoint[0][14] = 1 # Initial density in bin 14 -> Every agents are in that bin

# Create the cost function for the MINLP solvers
def cost_fun(xDict, MDict, swarm_ids, node_ids):
	costValue = 0
	for s_id in swarm_ids:
		for n_id in node_ids:
				costValue += (1-MDict[(s_id, n_id, n_id)])
	return costValue
cost_fun = None

# # Use GTLproco Sewuential solver
# optCost, status, solveTime, xDictRes, MdictRes, ljRes, swarm_ids, node_ids = gtlproco_scp(m_milp, mGraph, Kp, 
#     	initPoint, initPoint, cost_fun=cost_fun, timeLimit=50, n_thread=0, verbose=True, verbose_solver=False,
#     		costTol=1e-6, bilTol=1e-6, maxIter=100, mu_lin=1e1, mu_period=1e-4)


# optCost, status, solveTime, xDictRes, MdictRes, ljRes, swarm_ids, node_ids = \
# 	create_minlp_gurobi(m_milp, mGraph, Kp, initPoint, initPoint, 
# 		cost_fun=cost_fun, timeLimit=50, n_thread=0, verbose=True)

# # Save the loop point
# loopPoint = 0
# for ind, val in ljRes.items():
# 	if np.ceil(val) == 1:
# 		loopPoint = ind
# 		break
# xDictRes[-1] = loopPoint

# print('-------------------------------------------')
# print(ljRes)
# print(optCost, status, solveTime)
# maxDiff = 0
# # print('Desired Density : ', desPoint)
# for s_id in swarm_ids:
# 	for t in range(Kp):
# 		for n_id in node_ids:
# 			# print (s_id, t, n_id, xDictRes[(s_id, n_id,t)])
# 			s = sum(MdictRes[(s_id, n_id, m_id, t)]*xDictRes[(s_id, m_id,t)] for m_id in node_ids)
# 			maxDiff = np.maximum(maxDiff, np.abs(xDictRes[(s_id, n_id,t+1)]-s))
# print('Max diff = ', maxDiff)
# print('-------------------------------------------')


# # Save the policy and the intial density distribution
# with open('policy.json', 'w') as fileToSave:
# 	json.dump({str(k) : Mv for k, Mv in MdictRes.items()}, fileToSave, indent=4, sort_keys=True)
# with open('density.json', 'w') as fileToSave:
# 	json.dump({str(k) : Xv for k, Xv in xDictRes.items()}, fileToSave, indent=4, sort_keys=True)
# with open('density.json', 'w') as fileToSave:
# 	json.dump({str(k) : Xv for k, Xv in xDictRes.items()}, fileToSave, indent=4, sort_keys=True)

# exit()

######################## Load and Test the policy ###############################
#################################################################################

# Randomness for reproducibility
np.random.seed(401)

def compute_next_move_agent(s_id, tIndex, currBin, Mt, node_ids):
	""" Given the current bin of an agent, the current time index 
		and the corresponding swarm identifier with the time-varying Markov 
		matrix, compute the next bin the agent should reach
	"""
	while True:
		rand_z = np.random.random()
		curr_sum = 0
		for n_id in node_ids:
			new_curr_sum = curr_sum +  Mt[(s_id,n_id,currBin,tIndex)]
			if rand_z >= curr_sum and rand_z < new_curr_sum:
				return n_id
			curr_sum = new_curr_sum

# def compute_next_moves(agentBins, actualTime, MdictRes, node_ids):
# 	arrayNodes = np.array([n_id for n_id in node_ids])
# 	newAgentBins = np.zeros(agentBins.shape[0])
# 	for n_id in node_ids:
# 		agentIndexes  = np.where(agentBins == n_id)[0]
# 		if agentIndexes.shape[0] <= 0:
# 			continue
# 		agentDistr = np.random.choice(arrayNodes, agentIndexes.shape[0], p= np.array([MdictRes[(0,m_id,n_id,actualTime)] for m_id in node_ids]))
# 		for j in range(agentIndexes.shape[0]):
# 			newAgentBins[agentIndexes[j]] = agentDistr[j]
# 	return newAgentBins



# Load the polciy and density distribution from files
with open('policy.json', 'r') as fileToSave:
	MdictRes = json.load(fileToSave)
	MdictRes = {  make_tuple(key) : val for key, val in MdictRes.items() }
with open('density.json', 'r') as fileToSave:
	xDictRes = json.load(fileToSave)
	xDictRes = {   make_tuple(key) : val for key, val in xDictRes.items() }

# Define the number of agent in the swarm
# nbAgents = 20000
nbAgents = 1000

# Find the initial swarm distribution
nodesArray = np.zeros(len(node_ids))
initDist = np.zeros(len(node_ids))
for n_id in node_ids:
	initDist[n_id] = initPoint[0][n_id]
	nodesArray[n_id] = n_id
initialBinDist = np.random.choice(nodesArray, nbAgents, p=initDist)

# Define the robot positions (in term of bins location) and time horizon
timeHorizon = 30
robotPositions = np.zeros((nbAgents, timeHorizon), dtype=np.int64)
robotPositions[:,0] = initialBinDist

# Get the loop index
print(xDictRes[-1])
loopIndex = np.ceil(xDictRes[-1][0])
print( 'Loop index : {}'.format(loopIndex))
# for t in range(Kp+1):
# 	print('x[{}] = {}'.format(t, {n_id : xDictRes[(0,n_id,t)] for n_id in node_ids}) )

# Simulate a trajectory for the swarm
actualTime = 0 
for t in range(actualTime+1,timeHorizon):
	if actualTime == Kp:
		actualTime = loopIndex - 1
	# robotPositions[:,t] = compute_next_moves(robotPositions[:,t-1], actualTime, MdictRes, node_ids)
	for j in range(nbAgents):
		robotPositions[j,t] = compute_next_move_agent(0, actualTime, robotPositions[j,t-1], MdictRes, node_ids)
	actualTime += 1

# Save the obtained trajectory for visualization
np.save('robotTraj.npy', robotPositions)
# exit()

######################## Plot the performance of the trajectory ###############################
###############################################################################################

robotPositions = np.load('robotTraj.npy')
nbAgents = robotPositions.shape[0]
binsIds = np.unique(robotPositions)
denEvolution = np.zeros((len(node_ids), robotPositions.shape[1]))
for t in range(robotPositions.shape[1]):
	nodeVals, countVals =  np.unique(robotPositions[:,t], return_counts= True)
	for j in range(nodeVals.shape[0]):
		denEvolution[nodeVals[j], t] = countVals[j] / nbAgents
	assert np.abs(sum(denEvolution[:,t]) - 1) <= 1e-6
print(denEvolution)
print(robotPositions)

# Plot the evolution of bins in the obstacles bins
markersize = 4
linewidth = 2
obstacleColor, obstacleMarker ='red', 'o'
targetSpec = {16 : ('blue','*'), 17 : ('green', 'D'), 23 : ('cyan', '^') , 28 : ('magenta', 'x')}
plt.figure()
for i, n_id in enumerate(obstaclesBin):
	plt.plot(denEvolution[n_id,:], color=obstacleColor, linewidth=linewidth, marker= obstacleMarker, markersize=markersize, label= (None if i>0 else 'Obstacle Bins'))
for i, n_id in enumerate(targetNodes):
	plt.plot(denEvolution[n_id,:], color=targetSpec[n_id][0], linewidth=linewidth, marker=targetSpec[n_id][1], markersize=markersize, label = 'Target Bin {}'.format(n_id))

plt.ylabel('Density distribution')
plt.xlabel('Time steps')
plt.legend()
plt.tight_layout()
plt.grid(True)
# plt.savefig('figs/evol_obs_target.svg', dpi=300, transparent=True)
# tikzplotlib.clean_figure()
# tikzplotlib.save('figs/evol_obs_target.tex')

# Plot the density evolution in the starting bins
plt.figure()
initSpec = {0 : ('blue','*'), 6 : ('green', 'D'), 24 : ('cyan', '^') , 30 : ('magenta', 'x')}
for i, n_id in enumerate(initState):
	plt.plot(denEvolution[n_id,:], color=initSpec[n_id][0], linewidth=linewidth, marker=initSpec[n_id][1], markersize=markersize, label = 'Starting Bin {}'.format(n_id))
plt.ylabel('Density distribution')
plt.xlabel('Time steps')
plt.legend()
plt.tight_layout()
plt.grid(True)
# plt.savefig('figs/evol_initstate.svg', dpi=300, transparent=True)
# tikzplotlib.clean_figure()
# tikzplotlib.save('figs/evol_initstate.tex')

# Plot the evolution in the bins to illustrate capacity constraints
plt.figure()
capSpec = {8 : ('blue','*'), 12 : ('green', 'D'), 18 : ('cyan', '^') , 26 : ('magenta', 'x')}
for n_id in capSpec:
	plt.plot(denEvolution[n_id,:], color=capSpec[n_id][0], linewidth=linewidth, marker=capSpec[n_id][1], markersize=markersize, label = 'Bin {}'.format(n_id))
plt.ylabel('Density distribution')
plt.xlabel('Time steps')
plt.legend()
plt.tight_layout()
plt.grid(True)


plt.show()