from .LabelledGraph import *
from .GTLformula import *

import numpy as np
import random
import gurobi as gp

def random_connected_graph(nbNode, maxEdgePerNode):
	""" Generate a connected graph with each node having a maximum 
		number of edge specified by maxEdgePerNode.
		:param nbNode : Number of node of the graph
		:maxEdgePerNode : Maximum number of edge per node
	"""
	nodeSet = set ([i for i in range(nbNode)])
	source = np.random.randint(low=0, high=nbNode)
	cSet = set([source])
	resEdge = set()
	dictEdgeCnt = dict()
	for u in nodeSet:
		dictEdgeCnt[u] = 0
	while len(cSet) != len(nodeSet):
		v = np.random.choice(np.array(list(cSet)), 1)[0]
		u = np.random.choice(np.array(list(nodeSet-cSet)), 1)[0]
		resEdge.add((v,u))
		resEdge.add((u,u))
		resEdge.add((u,v))
		resEdge.add((v,v))
		if (u,v) not in resEdge:
			dictEdgeCnt[u] = dictEdgeCnt[u] + 1
		if (v,u) not in resEdge:
			dictEdgeCnt[v] = dictEdgeCnt[v] + 1
		if (u,u) not in resEdge:
			dictEdgeCnt[u] = dictEdgeCnt[u] + 1
		if (v,v) not in resEdge:
			dictEdgeCnt[v] = dictEdgeCnt[v] + 1
		cSet.add(u)

	# Add remaining edge on the built tree
	for u in nodeSet:
		nEdge = np.random.randint(low=0, high=maxEdgePerNode-dictEdgeCnt[u])
		nextN = np.random.choice(np.array(list(nodeSet)), nEdge, replace=False)
		for v in nextN:
			resEdge.add((u,v))
	return nodeSet, resEdge

def add_bounded_label_constr(lG):
	""" Label a given graph with node label [x, -x] in order to enforce
		time evolution demsity constraints
	"""
	for s_id in lG.eSubswarm:
		for n_id in lG.V:
			lG.addNodeLabel(n_id, 
				[getattr(lG, 'x_{}_{}'.format(s_id, n_id)),
				-getattr(lG, 'x_{}_{}'.format(s_id, n_id))])


def create_random_graph_and_reach_avoid_spec(nbNode, maxEdgePerNode, minWidth=0.1):
	""" Create a random graph with the number of node specified by nbNode
		and the maximum number of edge per node specified by maxEdgePerNode.
		Further the graph is labelled with the density of the unique swarm
		at each node.
		Besides, Eventually always (reach) and Always Eventually formula
		are applied to each node as the GTL formula to satisfied
	"""
	nodeSet, resEdge = random_connected_graph(nbNode, maxEdgePerNode)
	# Create the node label
	lG = LabelledGraph(nodeSet)
	# Add a single subswarm to the swarm
	lG.addSubswarm(0, resEdge)
	# Add the label of each node in the graph
	add_bounded_label_constr(lG)
	probV = np.random.rand(nbNode)
	probV = probV / sum(probV)
	safeSet_lb = np.random.uniform(low=np.zeros(nbNode), high=probV)
	safeSet_ub = np.random.uniform(low=np.minimum(probV+minWidth,1), high=np.full(nbNode, 1.0))
	# Pick a random desired distribution inside the safe set
	mOpt = gp.Model('Vector with constraint and sum to 1')
	mOpt.Params.OutputFlag = False
	xV = [mOpt.addVar(lb=safeSet_lb[i], ub=safeSet_ub[i]) for i in range(nbNode)]
	mOpt.addConstr(sum(xV) == 1)
	mOpt.setObjective(sum((xv - s_lb)*(xv - s_lb) for (xv,s_lb) in zip(xV, safeSet_lb)))
	mOpt.optimize()
	initPoint = { (0, v) : xv.x for v, xv in enumerate(xV)}

	mOpt.setObjective(sum((xv - s_lb)*(xv - s_lb) for (xv,s_lb) in zip(xV, safeSet_ub)))
	mOpt.optimize()
	desPoint = { (0, v) : xv.x for v, xv in enumerate(xV)}

	# print (initPoint)
	# print(desPoint)
	# Build the GTL formula at each node
	dictAtomicSafe= dict()
	dictAtomicDes = dict()
	dictFormula = dict()
	k=0
	for v in nodeSet:
		dictAtomicSafe[v] = AtomicGTL([safeSet_ub[v], -safeSet_lb[v]])
		dictAtomicDes[v] = AtomicGTL([desPoint[(0,v)], -desPoint[(0,v)]])
		dictFormula[v] = AndGTL([AlwaysGTL(dictAtomicSafe[v]), EventuallyAlwaysGTL(dictAtomicDes[v])])
	return lG, list(dictFormula.values()), list(dictFormula.keys()), dictAtomicSafe, initPoint, desPoint


if __name__ == "__main__":
	from .gtlproco import create_minlp_gurobi, create_gtl_proco
	# np.random.seed(201)
	nNode = 30
	nEdge = np.random.randint(2, 3)
	Kp = 10
	lG, gtl, nodes, safeGTL, initPoint, desPoint = create_random_graph_and_reach_avoid_spec(nNode, nEdge)
	# print_milp_repr(gtl, nodes, Kp, lG, initTime=0)
	m_milp = create_milp_constraints(gtl, nodes, Kp, lG)
	def cost_fun(xDict, MDict, swarm_ids, node_ids):
		costValue = 0
		for s_id in swarm_ids:
			for n_id in node_ids:
				for m_id in node_ids:
					costValue += MDict[(s_id, n_id, m_id)]
		return costValue
	# cost_fun = None

	optCost, status, solveTime, xDictRes, MdictRes, ljRes, swarm_ids, node_ids = \
		create_gtl_proco(m_milp, lG, Kp, initPoint, initPoint, cost_fun=cost_fun, solve=True, 
					maxIter=20, epsTol=1e-5, lamda=1, devM=1, rho0=1e-2, 
					rho1=0.75, rho2=0.95, alpha=3.0, beta=1.5,
					timeLimit=5, n_thread=0, verbose=False, autotune=False)
	print(ljRes)
	print(optCost, status, solveTime)
	maxDiff = 0
	# print('Desired Density : ', desPoint)
	for s_id in swarm_ids:
		for t in range(Kp):
			for n_id in node_ids:
				s = sum(MdictRes[(s_id, n_id, m_id, t)]*xDictRes[(s_id, m_id,t)] for m_id in node_ids)
				maxDiff = np.maximum(maxDiff, np.abs(xDictRes[(s_id, n_id,t+1)]-s))
	print('Max diff = ', maxDiff)

	optCost, status, solveTime, xDictRes, MdictRes, ljRes, swarm_ids, node_ids = \
		create_minlp_gurobi(m_milp, lG, Kp, initPoint, initPoint, 
			cost_fun=cost_fun, timeLimit=5, n_thread=0, verbose=False)
	print(ljRes)
	print(optCost, status, solveTime)
	maxDiff = 0
	# print('Desired Density : ', desPoint)
	for s_id in swarm_ids:
		for t in range(Kp):
			for n_id in node_ids:
				s = sum(MdictRes[(s_id, n_id, m_id, t)]*xDictRes[(s_id, m_id,t)] for m_id in node_ids)
				maxDiff = np.maximum(maxDiff, np.abs(xDictRes[(s_id, n_id,t+1)]-s))
	print('Max diff = ', maxDiff)
	# print('Desired Density : ', desPoint)
	# for s_id in swarm_ids:
	# 	for t in range(Kp+1):
	# 		xt = np.zeros(nNode)
	# 		for n_id in node_ids:
	# 			xt[n_id] = xDictRes[(s_id, n_id, t)]
	# 		print ('S_id: {}, Time: {}, {}'.format(s_id, t, xt))
			# for m_id in node_ids:
			# 	print('M[{}][{}] = {}'.format(n_id, m_id, MdictRes[(s_id, n_id, m_id, 4)]))