import numpy as np

from .LabelledGraph import LabelledGraph
from .GTLformula import *

from .utils_fun import create_random_graph_and_reach_avoid_spec

from .gtlproco import create_minlp_gurobi
from .gtlproco import create_minlp_scip
from .gtlproco import create_minlp_pyomo
from .gtlproco import create_gtl_proco
from .gtlproco import create_reach_avoid_problem_convex
from .gtlproco import create_reach_avoid_problem_lp

# Save the path for the non convex solver of pyomo
pyomo_path = "/home/franckd/GTLProCo/minlp_solver/"

def solve_problem(solver_name, pb_infos):
	global pyomo_path

	lG, gtl, nodes, safeGTL, initPoint, desPoint, m_milp, Kp = pb_infos
	timeL = 60*5 # 5 min
	n_thread = 0 # Solver decides the number of thread
	verb = False
	costF = None
	if solver_name == 'GTLProco':
		optCost, status, solveTime, xDictRes, MdictRes, ljRes, swarm_ids, node_ids = \
				create_gtl_proco(m_milp, lG, Kp, initPoint, initPoint, cost_fun=costF,
						solve=True, maxIter=200, epsTol=1e-4, lamda=1, devM=1, rho0=1e-3, 
						rho1=0.75, rho2=0.95, alpha=3.0, beta=3,
						timeLimit=timeL, n_thread=n_thread, verbose=verb, autotune=False
				)
	elif solver_name == 'Gurobi_MINLP':
		optCost, status, solveTime, xDictRes, MdictRes, ljRes, swarm_ids, node_ids = \
				create_minlp_gurobi(m_milp, lG, Kp, initPoint, initPoint, 
						cost_fun=costF, timeLimit=timeL, n_thread=n_thread, verbose=verb)
	elif solver_name == 'SCIP':
		optCost, status, solveTime, xDictRes, MdictRes, ljRes, swarm_ids, node_ids =\
			create_minlp_scip(m_milp, lG, Kp, initPoint, initPoint,
						cost_fun=costF, solve=True,
						timeLimit=timeL, n_thread=n_thread, verbose=verb)
	elif solver_name == 'couenne' or solver_name == 'bonmin':
		optCost, status, solveTime, xDictRes, MdictRes, ljRes, swarm_ids, node_ids =\
			create_minlp_pyomo(m_milp, lG, Kp, initPoint, initPoint,
						cost_fun=costF, solve=True, solver=solver_name,
						solverPath=pyomo_path, timeLimit=timeL, n_thread=n_thread, verbose=verb)
	elif solver_name == 'GTLProco_SDP':
		swarm_ids = swarm_ids = lG.eSubswarm.keys()
		node_ids = lG.V.copy()
		optCost, status, solveTime, MDictRes = \
			create_reach_avoid_problem_convex(safeGTL.values(), safeGTL.keys(), desPoint, lG, 
				cost_fun=costF, cost_M=None, solve=True, timeLimit=timeL, 
				n_thread=n_thread, verbose=verb, sdp_solver=True)
	elif solver_name == 'GTLProco_LP':
		swarm_ids = swarm_ids = lG.eSubswarm.keys()
		node_ids = lG.V.copy()
		optCost, status, solveTime, MDictRes = \
			create_reach_avoid_problem_lp(safeGTL.values(), safeGTL.keys(), desPoint, lG, 
				cost_fun=costF, cost_M=None, solve=True, timeLimit=timeL, 
				n_thread=n_thread, verbose=verb)
	else:
		assert False, " Solver not implemented "

	# Gather information on the convex constraint satisfaction
	maxDiff = 0
	if status == -1 or status == 0:
		maxDiff = -1
	if status == 1 and not (solver_name == 'GTLProco_SDP' or solver_name == 'GTLProco_LP'):
		for s_id in swarm_ids:
			for t in range(Kp):
				for n_id in node_ids:
					s = sum(MdictRes[(s_id, n_id, m_id, t)]*xDictRes[(s_id, m_id,t)] for m_id in node_ids)
					maxDiff = np.maximum(maxDiff, np.abs(xDictRes[(s_id, n_id,t+1)]-s))
	print ('----------------- {} -----------------'.format(solver_name))
	print ('NODES : {}, STATUS : {}, OPT COST : {}, SOLVE TIME : {}, MAXDIFF : {}.'.format(len(node_ids), status, optCost, solveTime, maxDiff))
	print ('----------------- END {} -----------------'.format(solver_name))
	return np.array([len(node_ids), status, optCost, solveTime, maxDiff]).reshape(1,-1)

# Set seed for reproductibility
# np.random.seed(101)
solverList = ['GTLProco', 'GTLProco_SDP', 'GTLProco_LP', 'Gurobi_MINLP', 'SCIP', 'couenne', 'bonmin']
maxPS = 101
sizeProblem = [ i for i in range(5, maxPS, 5)]
timeHorizon = [ np.maximum(5 + int(i/10.0),15) for i in range(5, maxPS, 5)]
nbTry = 5

import sys

save_dir = sys.argv[1]
print('File name: ', save_dir)

dictRes = dict()

for _ in range(nbTry):
	setNotToSolve = set()
	for sizePb, Kp in zip(sizeProblem, timeHorizon):
		lG, gtl, nodes, safeGTL, initPoint, desPoint = \
			create_random_graph_and_reach_avoid_spec(sizePb, 5, True)
		m_milp = create_milp_constraints(gtl, nodes, Kp, lG)
		params_solver = (lG, gtl, nodes, safeGTL, initPoint, desPoint, m_milp, Kp)
		for solver_name in solverList:
			if solver_name in setNotToSolve:
				continue
			try:
				res = solve_problem(solver_name, params_solver)
				if solver_name not in dictRes:
					dictRes[solver_name] = res
				else:
					dictRes[solver_name] = np.concatenate((dictRes[solver_name],res))
				if res[0,1] == -1: # Time Limit elapsed
					setNotToSolve.add(solver_name)
				# Don't run the sdp or lp if they take more time than the sequential solver
				if solver_name == 'GTLProco_SDP' or solver_name == 'GTLProco_LP':
					if res[0,-2] > dictRes['GTLProco'][-1,-2]:
						setNotToSolve.add(solver_name)
			except:
				setNotToSolve.add(solver_name)
				print('Error occured : {}, Size : {}, Kp : {}'.format(solver_name, sizePb, Kp))

		np.savez(save_dir, **dictRes)
