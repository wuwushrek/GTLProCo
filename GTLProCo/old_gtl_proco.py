from .LabelledGraph import LabelledGraph
from .GTLformula import *

# Import the optimizer tools
import gurobipy as gp
import cvxpy as cp
import time

def create_linearize_constr(mOpt, xDen, MarkovM, dictSlack, dictConstr, swam_ids, Kp, node_ids):
	""" Create a gurobi prototype of linearized nonconvex constraint
		around the solution of the previous iteration.
		Additionally, create the slack variables for the constraints.
	"""
	# Add the slack variable 
	for s_id in swam_ids:
		for t in range(Kp):
			for n_id in node_ids:
				dictSlack[(s_id, n_id, t)] = \
					mOpt.addVar(name='Z[{}][{}]({})'.format(s_id,n_id,t))
				dictSlack[(s_id, -n_id-1, t)] = \
					mOpt.addVar(name='nZ[{}][{}]({})'.format(s_id,n_id,t))

	# Create the linearized constraints
	for s_id in swam_ids:
		for t in range(Kp):
			for n_id in node_ids:
				dictConstr[(s_id, n_id, t)] = mOpt.addLConstr(
					gp.LinExpr([1,1,-1]+ [1.0/len(node_ids) for i in node_ids] + [1.0/len(node_ids) for i in node_ids], 
						[xDen[(s_id,n_id,t+1)], dictSlack[(s_id, n_id, t)], dictSlack[(s_id, -n_id-1, t)]] + \
						[ xDen[(s_id,m_id,t)] for m_id in node_ids] + \
						[ MarkovM[(s_id,n_id,m_id,t)] for m_id in node_ids]
					), 
					gp.GRB.EQUAL, 
					0, 
					name='Lin[{}][{}]({})'.format(s_id,n_id,t))


def create_l1_trust_region_constr(mOpt, xDen, swam_ids, Kp, node_ids):
	dictConstrVar = dict()
	dictConstrL1 = dict()
	extraVar = dict()
	for s_id in swam_ids:
		for t in range(Kp+1):
			for n_id in node_ids:
				extraVar[(s_id,n_id,t)] = mOpt.addVar(lb=0, name='t[{}][{}][{}]'.format(s_id,n_id,t))
				dictConstrVar[(s_id,n_id,t)] = (mOpt.addLConstr( gp.LinExpr([1,-1], [xDen[s_id,n_id,t],extraVar[(s_id,n_id,t)]]), gp.GRB.LESS_EQUAL, 0),
								mOpt.addLConstr( gp.LinExpr([-1,-1], [xDen[s_id,n_id,t],extraVar[(s_id,n_id,t)]]), gp.GRB.LESS_EQUAL, 0))
			dictConstrL1[(s_id, t)] = mOpt.addLConstr( gp.LinExpr([1 for i in node_ids], [extraVar[(s_id,n_id,t)] for n_id in node_ids]), gp.GRB.LESS_EQUAL, 0)
	return extraVar, dictConstrVar, dictConstrL1


def find_initial_feasible_Markov(lG, Kp, xk, verbose=True):
	# Create the Gurobi model
	mOpt = gp.Model('Initial Markov matrix problem')
	mOpt.Params.OutputFlag = verbose
	mOpt.Params.NumericFocus = 3
	# Swarm and graph configuration information
	swarm_ids = lG.eSubswarm.keys()
	node_ids = lG.V.copy()
	# Create the different Markov matrices
	MarkovM = dict()
	for s_id in swarm_ids:
		nedges = lG.getSubswarmNonEdgeSet(s_id)
		for n_id in node_ids:
			for m_id in node_ids:
				for t in range(Kp):
					if (m_id, n_id) in nedges:
						MarkovM[(s_id, n_id, m_id, t)] = \
							mOpt.addVar(lb=0, ub=0, name='M[{}][{}][{}][{}]'.format(s_id, n_id, m_id,t))
					else:
						MarkovM[(s_id, n_id, m_id, t)] = \
							mOpt.addVar(lb=0, ub=1, name='M[{}][{}][{}][{}]'.format(s_id, n_id, m_id,t))
	# Add the constraints on the Markov matrices
	for s_id in swarm_ids:
		for n_id in node_ids:
			for t in range(Kp):
				mOpt.addConstr(gp.quicksum([MarkovM[(s_id, m_id, n_id, t)] for m_id in node_ids]) == 1)

	# Additional constraints on the swarm density evolution due to Mk
	slVar = dict()
	for s_id in swarm_ids:
		for t in range(Kp):
			for n_id in node_ids:
				slVar[(s_id,n_id,t)] = mOpt.addVar(lb=0, name='slack[{}][{}][{}]'.format(s_id,n_id,t))
				slVar[(s_id,-n_id-1,t)] = mOpt.addVar(lb=0, name='slack[{}][{}][{}]'.format(s_id,-n_id-1,t))
				mOpt.addLConstr(
					gp.LinExpr([1, -1]+ [xk[s_id,m_id,t] for m_id in node_ids],\
						[slVar[(s_id,n_id,t)],slVar[(s_id,-n_id-1,t)]] + [MarkovM[(s_id,n_id,m_id,t)] for m_id in node_ids]
					), 
					gp.GRB.EQUAL, 
					xk[s_id,n_id,t+1], 
					name='Mkv[{}][{}]({})'.format(s_id,n_id,t))

	mOpt.setObjective(gp.quicksum([slackVar for _,slackVar in slVar.items()]), gp.GRB.MINIMIZE)
	mOpt.optimize()
	dictM = {(s_id, n_id, m_id, t) : Mval.x for (s_id, n_id, m_id,t), Mval in MarkovM.items()}
	return dictM

def find_initial_feasible_density(milp_expr, lG, Kp, init_den_lb=None, init_den_ub=None, verbose=True):
	""" Compute an initial feasible density as closest as possible to the given feasible
		Markov chain
	"""
	# Create the Gurobi model
	mOpt = gp.Model('Initial feasible density')
	mOpt.Params.OutputFlag = verbose
	mOpt.Params.Presolve = 2
	mOpt.Params.NumericFocus = 3
	# mOpt.Params.MIPFocus = 2
	# mOpt.Params.Heuristics = 0.01 # Less time to find feasible solution --> the problem is always feasible
	# mOpt.Params.Crossover = 0
	# mOpt.Params.CrossoverBasis = 0
	# # mOpt.Params.BarHomogeneous = 0
	# mOpt.Params.FeasibilityTol = 1e-6
	# mOpt.Params.OptimalityTol = 1e-6
	# mOpt.Params.MIPGap = 1e-3
	# mOpt.Params.MIPGapAbs = 1e-6

	# Obtain the milp encoding
	(newCoeffs, newVars, rhsVals, nVar), (lCoeffs, lVars, lRHS) = milp_expr

	# Get the boolean and continuous variables from the MILP expressions of the GTL
	bVars, rVars, lVars, denVars = getVars(newVars, lVars)

	# Swarm and graph configuration information
	swarm_ids = lG.eSubswarm.keys()
	node_ids = lG.V.copy()

	# Create the different density variables
	xDen = dict()
	for s_id in swarm_ids:
		for n_id in node_ids:
			for t in range(Kp+1):
				if t == 0 and init_den_lb is not None and init_den_ub is not None:
					xDen[(s_id, n_id, t)] = mOpt.addVar(lb=init_den_lb[s_id][n_id], ub=init_den_ub[s_id][n_id], 
															name='x[{}][{}]({})'.format(s_id, n_id, t))
				else:
					xDen[(s_id, n_id, t)] = mOpt.addVar(lb=0, ub=1, 
															name='x[{}][{}]({})'.format(s_id, n_id, t))
	# Create the boolean variables from the GTL formula -> add them to the xDen dictionary
	for ind in bVars:
		xDen[ind] = mOpt.addVar(vtype=gp.GRB.BINARY, name='b[{}]'.format(ind[0]))

	# Create the tempory real variables from the GTL formula -> add them to the xDen dictionary
	for ind in rVars:
		xDen[ind] = mOpt.addVar(lb=0, ub=1.0, name='r[{}]'.format(ind[0]))

	# Create the binary variables from the loop constraint
	for ind in lVars:
		xDen[ind] = mOpt.addVar(vtype=gp.GRB.BINARY, name='l[{}]'.format(ind[0]))

	# Trivial constraints from the Markov matrix
	for s_id in swarm_ids:
		nedges = lG.getSubswarmNonEdgeSet(s_id)
		for t in range(Kp):
			for n_id in node_ids:
				mOpt.addConstr(xDen[(s_id, n_id, t+1)] <= gp.quicksum([ 0 if (m_id, n_id) in nedges else xDen[(s_id,m_id,t)] for m_id in node_ids]))
				mOpt.addConstr(xDen[(s_id, n_id, t)] <= gp.quicksum([ 0 if (n_id, m_id) in nedges else xDen[(s_id,m_id,t+1)] for m_id in node_ids]))

	# Add the constraint on the density distribution
	listContr = create_den_constr(xDen,swarm_ids, Kp, node_ids)

	# Add the MILP constraint on the density distribution
	listContr.extend(create_gtl_constr(xDen, milp_expr))
	mOpt.addConstrs((c for c in listContr))

	# Set the cost function
	mOpt.setObjective(gp.quicksum( [(ind[0]+1)*xDen[ind] for ind in lVars] ), gp.GRB.MINIMIZE)
	mOpt.optimize()
	# mOpt.display()
	resxk = {kVal : xVal.x for kVal, xVal in xDen.items()}
	return resxk

def parameterized_feasible_Markov(lG, Kp, verbose=True):
	# Create the Gurobi model
	mOpt = gp.Model('Initial Markov matrix problem')
	mOpt.Params.OutputFlag = verbose
	mOpt.Params.NumericFocus = 3
	# Swarm and graph configuration information
	swarm_ids = lG.eSubswarm.keys()
	node_ids = lG.V.copy()
	# Create the different Markov matrices
	MarkovM = dict()
	for s_id in swarm_ids:
		nedges = lG.getSubswarmNonEdgeSet(s_id)
		for n_id in node_ids:
			for m_id in node_ids:
				for t in range(Kp):
					if (m_id, n_id) in nedges:
						MarkovM[(s_id, n_id, m_id, t)] = \
							mOpt.addVar(lb=0, ub=0, name='M[{}][{}][{}][{}]'.format(s_id, n_id, m_id,t))
					else:
						MarkovM[(s_id, n_id, m_id, t)] = \
							mOpt.addVar(lb=0, ub=1, name='M[{}][{}][{}][{}]'.format(s_id, n_id, m_id,t))
	# Add the constraints on the Markov matrices
	for s_id in swarm_ids:
		for n_id in node_ids:
			for t in range(Kp):
				mOpt.addConstr(gp.quicksum([MarkovM[(s_id, m_id, n_id, t)] for m_id in node_ids]) == 1)

	# Additional constraints on the swarm density evolution due to Mk
	slVar = dict()
	Mconstr =  dict()
	for s_id in swarm_ids:
		for t in range(Kp):
			for n_id in node_ids:
				slVar[(s_id,n_id,t)] = mOpt.addVar(lb=0, name='slack[{}][{}][{}]'.format(s_id,n_id,t))
				slVar[(s_id,-n_id-1,t)] = mOpt.addVar(lb=0, name='slack[{}][{}][{}]'.format(s_id,-n_id-1,t))
				Mconstr[(s_id,n_id,t)] = mOpt.addLConstr(
					gp.LinExpr([1, -1]+ [1.0/len(node_ids) for m_id in node_ids],\
						[slVar[(s_id,n_id,t)],slVar[(s_id,-n_id-1,t)]] + [MarkovM[(s_id,n_id,m_id,t)] for m_id in node_ids]
					), 
					gp.GRB.EQUAL, 
					1.0, 
					name='Mkv[{}][{}]({})'.format(s_id,n_id,t))

	mOpt.setObjective(gp.quicksum([slackVar for _,slackVar in slVar.items()]), gp.GRB.MINIMIZE)
	# mOpt.optimize()
	# dictM = {(s_id, n_id, m_id, t) : Mval.x for (s_id, n_id, m_id,t), Mval in MarkovM.items()}
	return mOpt, MarkovM, Mconstr

def parameterized_feasible_density(milp_expr, lG, Kp, init_den_lb=None, init_den_ub=None, verbose=True):
	""" Compute an initial feasible density as closest as possible to the given feasible
		Markov chain
	"""
	# Create the Gurobi model
	mOpt = gp.Model('Pertubated feasible density')
	mOpt.Params.OutputFlag = verbose
	mOpt.Params.Presolve = 2
	# mOpt.Params.NumericFocus = 3
	# mOpt.Params.MIPFocus = 2
	# mOpt.Params.Heuristics = 0.01 # Less time to find feasible solution --> the problem is always feasible
	# mOpt.Params.Crossover = 0
	# mOpt.Params.CrossoverBasis = 0
	# # mOpt.Params.BarHomogeneous = 0
	# mOpt.Params.FeasibilityTol = 1e-6
	# mOpt.Params.OptimalityTol = 1e-6
	# mOpt.Params.MIPGap = 1e-3
	# mOpt.Params.MIPGapAbs = 1e-6

	# Obtain the milp encoding
	(newCoeffs, newVars, rhsVals, nVar), (lCoeffs, lVars, lRHS) = milp_expr

	# Get the boolean and continuous variables from the MILP expressions of the GTL
	bVars, rVars, lVars, denVars = getVars(newVars, lVars)

	# Swarm and graph configuration information
	swarm_ids = lG.eSubswarm.keys()
	node_ids = lG.V.copy()

	# Create the different density variables
	xDen = dict()
	for s_id in swarm_ids:
		for n_id in node_ids:
			for t in range(Kp+1):
				if t == 0 and init_den_lb is not None and init_den_ub is not None:
					xDen[(s_id, n_id, t)] = mOpt.addVar(lb=init_den_lb[s_id][n_id], ub=init_den_ub[s_id][n_id], 
															name='x[{}][{}]({})'.format(s_id, n_id, t))
				else:
					xDen[(s_id, n_id, t)] = mOpt.addVar(lb=0, ub=1, 
															name='x[{}][{}]({})'.format(s_id, n_id, t))
	# Create the boolean variables from the GTL formula -> add them to the xDen dictionary
	for ind in bVars:
		xDen[ind] = mOpt.addVar(vtype=gp.GRB.BINARY, name='b[{}]'.format(ind[0]))

	# Create the tempory real variables from the GTL formula -> add them to the xDen dictionary
	for ind in rVars:
		xDen[ind] = mOpt.addVar(lb=0, ub=1.0, name='r[{}]'.format(ind[0]))

	# Create the binary variables from the loop constraint
	for ind in lVars:
		xDen[ind] = mOpt.addVar(vtype=gp.GRB.BINARY, name='l[{}]'.format(ind[0]))

	# Add the constraint on the density distribution
	listContr = create_den_constr(xDen,swarm_ids, Kp, node_ids)

	# Add the constraint on the pertubated region around a previous solution
	dictConstrVar = dict()
	dictConstrL1 = dict()
	extraVar = dict()
	for s_id in swarm_ids:
		for t in range(Kp+1):
			for n_id in node_ids:
				extraVar[(s_id,n_id,t)] = mOpt.addVar(lb=0, name='t[{}][{}][{}]'.format(s_id,n_id,t))
				dictConstrVar[(s_id,n_id,t)] = (mOpt.addLConstr( gp.LinExpr([1,-1], [xDen[s_id,n_id,t],extraVar[(s_id,n_id,t)]]), gp.GRB.LESS_EQUAL, 0),
								mOpt.addLConstr( gp.LinExpr([-1,-1], [xDen[s_id,n_id,t],extraVar[(s_id,n_id,t)]]), gp.GRB.LESS_EQUAL, 0))
			dictConstrL1[(s_id, t)] = mOpt.addLConstr( gp.LinExpr([1 for i in node_ids], [extraVar[(s_id,n_id,t)] for n_id in node_ids]), gp.GRB.LESS_EQUAL, 0)

	# Add the MILP constraint on the density distribution
	listContr.extend(create_gtl_constr(xDen, milp_expr))
	mOpt.addConstrs((c for c in listContr))

	# Set the cost function
	mOpt.setObjective(gp.quicksum([ slVar for kVal, slVar in extraVar.items()]), gp.GRB.MAXIMIZE)
	# mOpt.optimize()

	# resxk = {kVal : xVal.x for kVal, xVal in xDen.items()}
	return mOpt,xDen, dictConstrVar, dictConstrL1

def find_pertubated_solution(xOptPb, xVar, xkConstrVar, xkConstrL1, MoptPb, Mvar, MkConstr, xk, l1Dist, node_ids):
	###### Solve the problem to find the new feasible density farthest from previous one
	# First set the absolute value constraint
	for (s_id,n_id,t), (cV_r, cV_l) in  xkConstrVar.items():
		cV_r.RHS = xk.get((s_id,n_id,t), 0)
		cV_l.RHS = -xk.get((s_id,n_id,t), 0)
	# Set the L1 norm distance
	for (s_id, t), cV in xkConstrL1.items():
		cV.RHS = l1Dist
	xOptPb.optimize()
	# Gather the new x debnsity value
	dictXk = {kVal : xVal.x for kVal, xVal in xVar.items()} 
	###### Solve the problem to find the new feasible Markov matrix corresponding to the synthesized xk
	for (s_id,n_id,t), cVar in MkConstr.items():
		cVar.rhs = dictXk[(s_id,n_id,t+1)]
		for m_id in node_ids:
			MoptPb.chgCoeff(cVar, Mvar[(s_id, n_id, m_id, t)], dictXk[(s_id, m_id, t)])
	MoptPb.optimize()
	dictMk = { kVal : Mval.x for kVal, Mval in Mvar.items()}
	return dictXk, dictMk


def create_linearized_problem_scp(milp_expr, lG, Kp, cost_fun, init_den_lb=None, init_den_ub=None,
									timeLimit=5, n_thread=0, verbose=True, mu_lin=10, mu_dist=1):
	# Create the Gurobi model
	mOpt = gp.Model('Linearized problem')
	mOpt.Params.OutputFlag = verbose
	mOpt.Params.Presolve = 2
	mOpt.Params.NumericFocus = 3
	# mOpt.Params.MIPFocus = 1
	mOpt.Params.Crossover = 0
	mOpt.Params.CrossoverBasis = 0
	# # mOpt.Params.BarHomogeneous = 0
	# mOpt.Params.FeasibilityTol = 1e-6
	# mOpt.Params.OptimalityTol = 1e-6
	# # mOpt.Params.MIPGap = 1e-3
	# # mOpt.Params.MIPGapAbs = 1e-6
	mOpt.Params.Threads = n_thread
	mOpt.Params.TimeLimit = timeLimit

	# Obtain the milp encoding
	(newCoeffs, newVars, rhsVals, nVar), (lCoeffs, lVars, lRHS) = milp_expr

	# Get the boolean and continuous variables from the MILP expressions of the GTL
	bVars, rVars, lVars, denVars = getVars(newVars, lVars)

	# Swarm and graph configuration information
	swarm_ids = lG.eSubswarm.keys()
	node_ids = lG.V.copy()

	# Create the different density variables
	xDen = dict()
	for s_id in swarm_ids:
		for n_id in node_ids:
			for t in range(Kp+1):
				if t == 0 and init_den_lb is not None and init_den_ub is not None:
					xDen[(s_id, n_id, t)] = mOpt.addVar(lb=init_den_lb[s_id][n_id], ub=init_den_ub[s_id][n_id], 
															name='x[{}][{}]({})'.format(s_id, n_id, t))
				else:
					xDen[(s_id, n_id, t)] = mOpt.addVar(lb=0, ub=1, 
															name='x[{}][{}]({})'.format(s_id, n_id, t))
	# Create the boolean variables from the GTL formula -> add them to the xDen dictionary
	for ind in bVars:
		xDen[ind] = mOpt.addVar(vtype=gp.GRB.BINARY, name='b[{}]'.format(ind[0]))

	# Create the tempory real variables from the GTL formula -> add them to the xDen dictionary
	for ind in rVars:
		xDen[ind] = mOpt.addVar(lb=0, ub=1.0, name='r[{}]'.format(ind[0]))

	# Create the binary variables from the loop constraint
	for ind in lVars:
		xDen[ind] = mOpt.addVar(vtype=gp.GRB.BINARY, name='l[{}]'.format(ind[0]))

	# Create the different Markov matrices
	MarkovM = dict()
	for s_id in swarm_ids:
		nedges = lG.getSubswarmNonEdgeSet(s_id)
		for n_id in node_ids:
			for m_id in node_ids:
				for t in range(Kp):
					if (m_id, n_id) in nedges:
						MarkovM[(s_id, n_id, m_id, t)] = \
							mOpt.addVar(lb=0, ub=0, name='M[{}][{}][{}]({})'.format(s_id, n_id, m_id, t))
					else:
						MarkovM[(s_id, n_id, m_id, t)] = \
							mOpt.addVar(lb=0, ub=1, name='M[{}][{}][{}]({})'.format(s_id, n_id, m_id, t))

	# Trivial constraints from the Markov matrix
	for s_id in swarm_ids:
		nedges = lG.getSubswarmNonEdgeSet(s_id)
		for t in range(Kp):
			for n_id in node_ids:
				mOpt.addConstr(xDen[(s_id, n_id, t+1)] <= gp.quicksum([ 0 if (m_id, n_id) in nedges else xDen[(s_id,m_id,t)] for m_id in node_ids]))
				# mOpt.addConstr(xDen[(s_id, n_id, t)] <= gp.quicksum([ 0 if (n_id, m_id) in nedges else xDen[(s_id,m_id,t+1)] for m_id in node_ids]))


	# Create the linearized constraints
	dictSlack = dict()
	dictConstr =  dict()
	create_linearize_constr(mOpt, xDen, MarkovM, dictSlack, dictConstr, swarm_ids, Kp, node_ids)
	listContr = []

	# Add the constraint on the density distribution
	listContr.extend(create_den_constr(xDen, swarm_ids, Kp, node_ids))

	# Add the constraints on the Markov matrices
	listContr.extend(create_markov_constr(MarkovM, swarm_ids, Kp, node_ids))

	# Add the mixed-integer constraints
	listContr.extend(create_gtl_constr(xDen, milp_expr))

	# Create trust region constraints
	extraVar, dictConstrVar, dictConstrL1 = create_l1_trust_region_constr(mOpt, xDen, swarm_ids, Kp, node_ids)

	# Add all the constraints
	mOpt.addConstrs((c for c in listContr))

	costVal = get_cost_function(cost_fun, xDen, MarkovM, swarm_ids, Kp, node_ids)

	# set_cost(cVal, xDen, mD, nids, sids)
	mOpt.setObjective(gp.quicksum([*[mu_lin*sVar for _, sVar in dictSlack.items()], costVal, *[mu_dist*(ind[0]+1)*xDen[ind] for ind in lVars]]), gp.GRB.MINIMIZE)

	# mOpt.setObjective(costVal + mu_lin * maxNorm['max'])
	mOpt.update()

	return mOpt, xDen, MarkovM, lVars, dictConstr, dictConstrVar, dictConstrL1

def update_linearized_problem_scp(mOpt, xDen, MarkovM, dictConstr, trustRegion, dictConstrVar, dictConstrL1, xk, Mk, node_ids):
	""" Given the linearized bilinear constraints, update such a constraint with respect to the previous solution
		xk and Mk.
	"""
	# First set the trust region constraints
	for (s_id,n_id,t), (cV_r, cV_l) in  dictConstrVar.items():
		cV_r.RHS = xk[(s_id,n_id,t)]
		cV_l.RHS = -xk[(s_id,n_id,t)]
	# Set the trust region on the L1 norm
	for (s_id, t), cV in dictConstrL1.items():
		cV.RHS = trustRegion

	# Set the coefficients of the linearized constraints
	for (s_id, n_id, t), c in dictConstr.items():
		c.RHS = -sum([Mk[(s_id, n_id, m_id, t)]*xk[(s_id, m_id, t)] for m_id in node_ids])
		for m_id in node_ids:
			mOpt.chgCoeff(c, MarkovM[(s_id, n_id, m_id, t)], -xk[(s_id, m_id, t)])
			mOpt.chgCoeff(c, xDen[(s_id, m_id, t)], - Mk[(s_id, n_id, m_id, t)])

def compute_true_linearized_fun(xk, Mk, xk1, Mk1, swam_ids, Kp, node_ids):
	fklist = np.array([sum( [xk[(s_id,m_id,t)]*Mk1[(s_id,n_id,m_id,t)]+Mk[(s_id,n_id,m_id,t)]*(xk1[(s_id,m_id,t)]-xk[(s_id,m_id,t)]) for m_id in node_ids])\
							 for s_id in swam_ids for t in range(Kp) for n_id in node_ids])
	flist = np.array([sum([xk1[(s_id,m_id,t)]*Mk1[(s_id,n_id,m_id,t)] for m_id in node_ids]) \
						for s_id in swam_ids for t in range(Kp) for n_id in node_ids])
	x1nextlist = np.array([xk1[(s_id,n_id,t+1)] for s_id in swam_ids for t in range(Kp) for n_id in node_ids])
	xnextlist = np.array([xk[(s_id,n_id,t+1)] for s_id in swam_ids for t in range(Kp) for n_id in node_ids])
	prodxlist = np.array([sum([xk[(s_id,m_id,t)]*Mk[(s_id,n_id,m_id,t)] for m_id in node_ids]) \
						for s_id in swam_ids for t in range(Kp) for n_id in node_ids])
	return flist, fklist, np.linalg.norm(x1nextlist-flist,1), np.linalg.norm(xnextlist-prodxlist,1) 

def gtl_proco_scp(milp_expr, lG, Kp, init_den_lb=None, init_den_ub=None,
					cost_fun=None, maxIter=10, epsTol=1e-6, mu_lin=1e1, mu_dist=1,
					trust_lim= 1e-4, trust_mult= 1.5, trust_div=1.5,
					rho0 = 0.2, rho1 = 0.8 ,
					timeLimit=5, n_thread=0, verbose=True):
	
	# Get the node ids and swarm ids
	swarm_ids = lG.eSubswarm.keys()
	node_ids = lG.V.copy()

	# xFeasOpt, xFeasVar, xFeasConstrVar, xFeasConstrL1 = parameterized_feasible_density(milp_expr, lG, Kp, init_den_lb=init_den_lb, init_den_ub=init_den_ub, verbose=verbose)
	# MFeasOpt, MarkovMVar, MarkovMconstr = parameterized_feasible_Markov(lG, Kp, verbose=verbose)

	# dictXk, dictMk = find_pertubated_solution(xFeasOpt, xFeasVar, xFeasConstrVar, xFeasConstrL1, MFeasOpt, MarkovMVar, MarkovMconstr, dict(), 1, node_ids)
	# # exit()

	# Initialize the SCP with a feasible solution to the GTL constraints
	dictXk = find_initial_feasible_density(milp_expr, lG, Kp, init_den_lb=init_den_lb, init_den_ub=init_den_ub, verbose=verbose)

	# Initialize the SCP with the closes Markov Matrix
	dictMk = find_initial_feasible_Markov(lG, Kp, dictXk, verbose=verbose)

	# exit()

	# Construct the linearized problem
	linProb, xLin, Mlin, lVars, linConstr, dictConstrVar, dictConstrL1 = create_linearized_problem_scp(milp_expr, lG, Kp, cost_fun, 
							init_den_lb=init_den_lb, init_den_ub=init_den_ub,
							timeLimit=timeLimit, n_thread=n_thread, verbose=verbose, 
							mu_lin=mu_lin, mu_dist=mu_dist)

	# Initialize the periodicy coefficient
	ljRes = {ind0 : dictXk[ind0] for ind0 in lVars}

	# Set the maximum trust region and intial trust region
	trust_max = len(node_ids)
	trust_init = 1.0 # len(node_ids)

	# Initialize the trust region
	rk = trust_init

	# Solving parameters and status
	solve_time = 0
	status = -1
	optCost = 0

	for i in range(maxIter):
		# Check if teh time limit is elapsed
		if solve_time >= timeLimit:
			status = -1
			break
		# maxDiff = 0
		# for s_id in swarm_ids:
		# 	for t in range(Kp):
		# 		for n_id in node_ids:
		# 			print (s_id, t, n_id, dictXk[(s_id, n_id,t)])
		# 			s = sum(dictMk[(s_id, n_id, m_id, t)]*dictXk[(s_id, m_id,t)] for m_id in node_ids)
		# 			maxDiff = np.maximum(maxDiff, np.abs(dictXk[(s_id, n_id,t+1)]-s))
		# print('-----------------------------Max diff = ', maxDiff)
		# linProb.reset()
		# Update the starting point of the linearized problem
		# for kVal, xVal in xLin.items():
		# 	xVal.start = dictXk[kVal]
		# for kVal, Mval in Mlin.items():
		# 	Mval.start = dictMk[kVal]
		# Update the optimization problem and solve it
		update_linearized_problem_scp(linProb, xLin, Mlin, linConstr, rk, dictConstrVar, dictConstrL1, dictXk, dictMk, node_ids)
		# linProb.update()
		# Measure optimization time
		cur_t = time.time()
		linProb.optimize()
		# linProb.display()
		solve_time += time.time() - cur_t
		# linProb.display()
		if linProb.status == gp.GRB.OPTIMAL:
			status = 1
		elif linProb.status == gp.GRB.TIME_LIMIT:
			status = -1
			break
		else:
			status = 0
			# break
		dictMk1 = {(s_id, n_id, m_id, t) : Mval.x for (s_id, n_id, m_id, t), Mval in Mlin.items()}
		dictXk1 = {kVal : xVal.x for kVal, xVal in xLin.items()}

		truef, linf, x1diffbil, xdiffbil = compute_true_linearized_fun(dictXk, dictMk, dictXk1, dictMk1, swarm_ids, Kp, node_ids)
		finish_bil = x1diffbil <= 1e-6*len(node_ids)*Kp*len(swarm_ids) or xdiffbil <= 1e-6*len(node_ids)*Kp*len(swarm_ids)

		# # In case the current solution doesn't improve and is not feasible pertubate the previous solution
		# if (i > 0) and (np.abs(linProb.objVal-attainedCost) < epsTol) and diffLin == 0 and x1diffbil > 1e-6*len(node_ids)*Kp*len(swarm_ids):
		# 	dictXk, _ = find_pertubated_solution(xFeasOpt, xFeasVar, xFeasConstrVar, xFeasConstrL1, MFeasOpt, MarkovMVar, MarkovMconstr, dictXk, 1.0 + np.random.rand(), node_ids)
		# 	continue

		diffLin = np.linalg.norm(truef-linf, 1)
		# noImprov = (i > 0) and (np.abs(linProb.objVal-attainedCost) < epsTol) and diffLin == 0
		noImprov = (i > 0) and (np.abs(linProb.objVal-attainedCost) < epsTol) and finish_bil
		actualCost  = get_cost_function(cost_fun, dictXk1, dictMk1, swarm_ids, Kp, node_ids)
		attainedCost = linProb.objVal
		# rho_k = (np.abs(actualCost-attainedCost) + diffLin)/(np.abs(attainedCost)+np.linalg.norm(linf, 1))
		rho_lin_k  = diffLin/np.linalg.norm(linf, 1)
		rho_bil_k  =  1 if (xdiffbil == 0 and x1diffbil ==0) else (np.inf if xdiffbil == 0 else x1diffbil / xdiffbil)
		# rho_k = rho_lin_k * rho_bil_k
		rho_k =  rho_lin_k

		if rho_k > rho1 or rho_bil_k > 1:
			# trust_div = 0 if rho_bil_k == np.inf else rho_bil_k
			# rk  =  rk / trust_div
			rk  =  rk / (0.1+rho_bil_k) if rho_bil_k < np.inf else 0
			if verbose:
				print('[Iteration {}] : Reject current solution -> Coarse Linearization'.format(i))
		else:
			dictXk = dictXk1
			dictMk = dictMk1
			optCost = actualCost
			ljRes = {ind0 : xLin[ind0].x for ind0 in lVars}
			# rk = np.minimum(rk * trust_mult if rho_k < rho0 else rk, trust_max)
			rk = np.minimum(rk / rho_bil_k if rho_bil_k < np.inf else 0, trust_max)
			if verbose:
				print('[Iteration {}] : Accept current solution'.format(i))

		if verbose:
			print('[Iteration {}] : Rho lin = {}, Rho bil = {}, bil diff xk = {}, bil diff xk1 = {}'.format(i, rho_lin_k, rho_bil_k, xdiffbil, x1diffbil))
			print('[Iteration {}] : Trust region = {}, Rho_k = {}, actual Cost = {}, attained Cost = {}, Lin Error = {}, Solve Time = {}'.format(
					i, rk, rho_k, actualCost, attainedCost, diffLin, solve_time))

		if rk < trust_lim:
			if verbose:
				print('[Iteration {}] : Minimum trust region reached'.format(i))
			break

		if noImprov:
			if verbose:
				print('[Iteration {}] : No improvement in the solution'.format(i))
			break


	return optCost, status, solve_time, dictXk, dictMk, ljRes, swarm_ids, node_ids



# def create_linearized_problem_scp(milp_expr, lG, Kp, cost_fun, init_den_lb=None, init_den_ub=None,
# 									timeLimit=5, n_thread=0, verbose=True, mu_lin=10, mu_spec=100):
# 	# Create the Gurobi model
# 	mOpt = gp.Model('Linearized problem')
# 	mOpt.Params.OutputFlag = verbose
# 	# mOpt.Params.Presolve = 2
# 	# mOpt.Params.MIPFocus = 2
# 	# mOpt.Params.Heuristics = 0.01 # Less time to find feasible solution --> the problem is always feasible
# 	# mOpt.Params.Crossover = 0
# 	# mOpt.Params.CrossoverBasis = 0
# 	# # mOpt.Params.BarHomogeneous = 0
# 	# mOpt.Params.FeasibilityTol = 1e-6
# 	# mOpt.Params.OptimalityTol = 1e-6
# 	# mOpt.Params.MIPGap = 1e-3
# 	# mOpt.Params.MIPGapAbs = 1e-6
# 	mOpt.Params.Threads = n_thread
# 	mOpt.Params.TimeLimit = timeLimit

# 	# Obtain the milp encoding
# 	(newCoeffs, newVars, rhsVals, nVar), (lCoeffs, lVars, lRHS) = milp_expr

# 	# Get the boolean and continuous variables from the MILP expressions of the GTL
# 	bVars, rVars, lVars, varSlack, denVars = getVars(newVars, lVars)

# 	# Swarm and graph configuration information
# 	swarm_ids = lG.eSubswarm.keys()
# 	node_ids = lG.V.copy()

# 	# Create the different density variables
# 	xDen = dict()
# 	for s_id in swarm_ids:
# 		for n_id in node_ids:
# 			for t in range(Kp+1):
# 				if t == 0 and init_den_lb is not None and init_den_ub is not None:
# 					xDen[(s_id, n_id, t)] = mOpt.addVar(lb=init_den_lb[s_id][n_id], ub=init_den_ub[s_id][n_id], 
# 															name='x[{}][{}]({})'.format(s_id, n_id, t))
# 				else:
# 					xDen[(s_id, n_id, t)] = mOpt.addVar(lb=0, ub=1, 
# 															name='x[{}][{}]({})'.format(s_id, n_id, t))
# 	# Create the boolean variables from the GTL formula -> add them to the xDen dictionary
# 	for ind in bVars:
# 		xDen[ind] = mOpt.addVar(vtype=gp.GRB.BINARY, name='b[{}]'.format(ind[0]))

# 	# Create the tempory real variables from the GTL formula -> add them to the xDen dictionary
# 	for ind in rVars:
# 		xDen[ind] = mOpt.addVar(lb=0, ub=1.0, name='r[{}]'.format(ind[0]))

# 	# Create the binary variables from the loop constraint
# 	for ind in lVars:
# 		xDen[ind] = mOpt.addVar(vtype=gp.GRB.BINARY, name='l[{}]'.format(ind[0]))

# 	# Create the slack variables 
# 	# assert len(varSlack) == 1, 'Only one slack variable for the spec are allowed'
# 	dictSlackSpec = dict()
# 	for ind in varSlack:
# 		xDen[ind] = mOpt.addVar(lb=0, name='|slack|[{}]'.format(ind[0]))
# 		dictSlackSpec[ind] = xDen[ind]
# 		# slackVar = xDen[ind]

# 	# Create the different Markov matrices
# 	MarkovM = dict()
# 	for s_id in swarm_ids:
# 		nedges = lG.getSubswarmNonEdgeSet(s_id)
# 		for n_id in node_ids:
# 			for m_id in node_ids:
# 				for t in range(Kp):
# 					if (m_id, n_id) in nedges:
# 						MarkovM[(s_id, n_id, m_id, t)] = \
# 							mOpt.addVar(lb=0, ub=0, name='M[{}][{}][{}]({})'.format(s_id, n_id, m_id, t))
# 					else:
# 						MarkovM[(s_id, n_id, m_id, t)] = \
# 							mOpt.addVar(lb=0, ub=1, name='M[{}][{}][{}]({})'.format(s_id, n_id, m_id, t))

# 	# Create the linearized constraints
# 	dictSlack = dict()
# 	dictConstr =  dict()
# 	create_linearize_constr(mOpt, xDen, MarkovM, dictSlack, dictConstr, swarm_ids, Kp, node_ids)
# 	listContr = []

# 	# Add the constraint on the density distribution
# 	listContr.extend(create_den_constr(xDen,swarm_ids, Kp, node_ids))

# 	# Add the constraints on the Markov matrices
# 	listContr.extend(create_markov_constr(MarkovM, swarm_ids, Kp, node_ids))

# 	# Add the mixed-integer constraints
# 	listContr.extend(create_gtl_constr(xDen, milp_expr))

# 	# Add all the constraints
# 	mOpt.addConstrs((c for c in listContr))

# 	costVal = get_cost_function(cost_fun, xDen, MarkovM, swarm_ids, Kp, node_ids)

# 	# set_cost(cVal, xDen, mD, nids, sids)
# 	mOpt.setObjective(gp.quicksum([*[mu_lin*sVar for _, sVar in dictSlack.items()], 
# 									costVal, 
# 									*[mu_spec*slackVar for _,slackVar in dictSlackSpec.items()]]), 
# 						gp.GRB.MINIMIZE)
# 	mOpt.update()

# 	return mOpt, xDen, MarkovM, dictConstr

# def update_linearized_problem_scp(mOpt, xDen, MarkovM, initRange, dictConstr, trustRegion, xk, Mk, node_ids):
# 	""" Given the linearized bilinear constraints, update such a constraint with respect to the previous solution
# 		xk and Mk.
# 	"""
# 	# First set the trust region constraints
# 	for (s_id, n_id, m_id, t), iVal in initRange.items():
# 		# # Maybe a trust region on the density too ? TODO and TO CHECK
# 		# xDen[(s_id, n_id, t+1)].lb = \
# 		# 		np.maximum(xk[(s_id, n_id,t+1)]-trustRegion, 0)
# 		# xDen[(s_id, n_id, t+1)].ub = \
# 		# 		np.minimum(xk[(s_id, n_id,t+1)]+trustRegion, 1)
# 		MarkovM[(s_id, n_id, m_id, t)].lb = \
# 			np.minimum(np.maximum(Mk[(s_id, n_id, m_id, t)]-trustRegion, iVal[0]), iVal[1])
# 		MarkovM[(s_id, n_id, m_id, t)].ub = \
# 			np.maximum(np.minimum(Mk[(s_id, n_id, m_id, t)]+trustRegion, iVal[1]), iVal[0])

# 	# Set the coefficients of the linearized constraints
# 	for (s_id, n_id, t), c in dictConstr.items():
# 		c.RHS = -sum([Mk[(s_id, n_id, m_id, t)]*xk[(s_id, m_id, t)] for m_id in node_ids])
# 		for m_id in node_ids:
# 			mOpt.chgCoeff(c, MarkovM[(s_id, n_id, m_id, t)], -xk[(s_id, m_id, t)])
# 			mOpt.chgCoeff(c, xDen[(s_id, m_id, t)], - Mk[(s_id, n_id, m_id, t)])

# def find_initial_feasible_Markov(lG, Kp, verbose=True):
# 	# Create the Gurobi model
# 	mOpt = gp.Model('Initial Markov matrix problem')
# 	mOpt.Params.OutputFlag = verbose
# 	mOpt.Params.NumericFocus = 3
# 	# Swarm and graph configuration information
# 	swarm_ids = lG.eSubswarm.keys()
# 	node_ids = lG.V.copy()
# 	# Create the different Markov matrices
# 	MarkovM = dict()
# 	for s_id in swarm_ids:
# 		nedges = lG.getSubswarmNonEdgeSet(s_id)
# 		for n_id in node_ids:
# 			for m_id in node_ids:
# 				if (m_id, n_id) in nedges:
# 					MarkovM[(s_id, n_id, m_id)] = \
# 							mOpt.addVar(lb=0, ub=0, name='M[{}][{}][{}]'.format(s_id, n_id, m_id))
# 				else:
# 					MarkovM[(s_id, n_id, m_id)] = \
# 							mOpt.addVar(lb=0, ub=1, name='M[{}][{}][{}]'.format(s_id, n_id, m_id))
# 	# Add the constraints on the Markov matrices
# 	for s_id in swarm_ids:
# 		for n_id in node_ids:
# 			mOpt.addConstr(gp.quicksum([MarkovM[(s_id, m_id, n_id)] for m_id in node_ids]) == 1)

# 	mOpt.setObjective(0)
# 	mOpt.optimize()
# 	dictM = {(s_id, n_id, m_id, t) : Mval.x for (s_id, n_id, m_id), Mval in MarkovM.items() for t in range(Kp)}
# 	return dictM

# def verification_step_scp(verifOpt, contrSet, xVar, Mk, Kp, swarm_ids, node_ids):
# 	# Update the constraint of verifOpt to find if the solution xk satisfies the specification
# 	# and quantify how unsatisfied the specifications a
# 	for (s_id, t, n_id), cVal in  contrSet.items():
# 		for m_id in node_ids:
# 			verifOpt.chgCoeff(cVal, xVar[(s_id, m_id, t)], Mk[(s_id, n_id, m_id, t)])
# 	# Solve the feasible problem
# 	verifOpt.optimize()
# 	# Save the obtained solution and the cost of satisfying the spec
# 	resxk = {(s_id,n_id,t) : xVar[(s_id,n_id,t)].x for s_id in swarm_ids for n_id in node_ids for t in range(Kp+1)}
# 	return resxk, verifOpt.objVal

# def create_verif_problem_scp(milp_expr, lG, Kp, init_den_lb=None, init_den_ub=None, verbose=True):
# 	""" Given the MILP constraints corresponding by the GTL specifications 
# 		and given the initial density distribution, construct an optimization problem to
# 		find if the density satisfies the specifications. 
# 		And in case it does not satisfy, provide how far it is from satisfying the specifications
# 	"""
# 	# Create the Gurobi model
# 	mOpt = gp.Model('Verification problem')
# 	mOpt.Params.OutputFlag = verbose
# 	# mOpt.Params.Presolve = 2
# 	# mOpt.Params.MIPFocus = 2
# 	# mOpt.Params.Heuristics = 0.01 # Less time to find feasible solution --> the problem is always feasible
# 	# mOpt.Params.Crossover = 0
# 	# mOpt.Params.CrossoverBasis = 0
# 	# # mOpt.Params.BarHomogeneous = 0
# 	# mOpt.Params.FeasibilityTol = 1e-6
# 	# mOpt.Params.OptimalityTol = 1e-6
# 	# mOpt.Params.MIPGap = 1e-3
# 	# mOpt.Params.MIPGapAbs = 1e-6

# 	# Obtain the milp encoding
# 	(newCoeffs, newVars, rhsVals, nVar), (lCoeffs, lVars, lRHS) = milp_expr

# 	# Get the boolean and continuous variables from the MILP expressions of the GTL
# 	bVars, rVars, lVars, denVars = getVars(newVars, lVars)

# 	# Swarm and graph configuration information
# 	swarm_ids = lG.eSubswarm.keys()
# 	node_ids = lG.V.copy()

# 	# Create the different density variables
# 	xDen = dict()
# 	for s_id in swarm_ids:
# 		for n_id in node_ids:
# 			for t in range(Kp+1):
# 				if t == 0 and init_den_lb is not None and init_den_ub is not None:
# 					xDen[(s_id, n_id, t)] = mOpt.addVar(lb=init_den_lb[s_id][n_id], ub=init_den_ub[s_id][n_id], 
# 															name='x[{}][{}]({})'.format(s_id, n_id, t))
# 				else:
# 					xDen[(s_id, n_id, t)] = mOpt.addVar(lb=0, ub=1, 
# 															name='x[{}][{}]({})'.format(s_id, n_id, t))
# 	# Create the boolean variables from the GTL formula -> add them to the xDen dictionary
# 	for ind in bVars:
# 		xDen[ind] = mOpt.addVar(vtype=gp.GRB.BINARY, name='b[{}]'.format(ind[0]))

# 	# Create the tempory real variables from the GTL formula -> add them to the xDen dictionary
# 	for ind in rVars:
# 		xDen[ind] = mOpt.addVar(lb=0, ub=1.0, name='r[{}]'.format(ind[0]))

# 	# Create the binary variables from the loop constraint
# 	for ind in lVars:
# 		xDen[ind] = mOpt.addVar(vtype=gp.GRB.BINARY, name='l[{}]'.format(ind[0]))

# 	# Create the slack variables 
# 	dictSlackSpec = dict()
# 	for ind in varSlack:
# 		xDen[ind] = mOpt.addVar(lb=0, name='|slack|[{}]'.format(ind[0]))
# 		dictSlackSpec[ind] = xDen[ind]
# 		# slackVar = xDen[ind]

# 	# Add the constraint on the density distribution
# 	listContr = create_den_constr(xDen,swarm_ids, 0, node_ids) # Only enforce initial solution to have sum dist = 1 

# 	# Add the MILP constraint on the density distribution
# 	listContr.extend(create_gtl_constr(xDen, milp_expr))
# 	mOpt.addConstrs((c for c in listContr))

# 	# Additional constraints on the swarm density evolution due to Mk
# 	chgConstraints = dict()
# 	for s_id in swarm_ids:
# 		for t in range(Kp):
# 			for n_id in node_ids:
# 				chgConstraints[(s_id,t,n_id)] = mOpt.addLConstr(
# 					gp.LinExpr([-1]+ [1.0/len(node_ids) for i in node_ids],\
# 						[xDen[(s_id,n_id,t+1)]] + [ xDen[(s_id,m_id,t)] for m_id in node_ids]
# 					), 
# 					gp.GRB.EQUAL, 
# 					0, 
# 					name='Mkv[{}][{}]({})'.format(s_id,n_id,t))
# 	# Set the cost function
# 	mOpt.setObjective(gp.quicksum([slackVar for _,slackVar in dictSlackSpec.items()]), gp.GRB.MINIMIZE)

# 	# Update the optimization problem
# 	mOpt.update()

# 	# # Display the problem
# 	# mOpt.display()
# 	return mOpt, xDen, dictSlackSpec, chgConstraints

# def gtl_proco_scp(milp_expr, lG, Kp, init_den_lb=None, init_den_ub=None,
# 					cost_fun=None, maxIter=20, epsTol=1e-6, mu_lin=1e1, mu_spec=1e1, 
# 					trust_init=0.5, trust_lim= 1e-6, trust_max=None, 
# 					trust_mult= 1.5, trust_div=1.5,
# 					timeLimit=5, n_thread=0, verbose=True):
# 	(newCoeffs, newVars, rhsVals, nVar), (lCoeffs, lVars, lRHS) = milp_expr
# 	print_constr(newCoeffs, newVars, rhsVals, nVar, lCoeffs, lVars, lRHS)
# 	# Construct the linearized problem
# 	linProb, xLin, Mlin, linConstr = create_linearized_problem_scp(milp_expr, lG, Kp, cost_fun, 
# 							init_den_lb=init_den_lb, init_den_ub=init_den_ub,
# 							timeLimit=timeLimit, n_thread=n_thread, verbose=verbose, 
# 							mu_lin=mu_lin, mu_spec=mu_spec)
# 	# Construct the verification step problem
# 	verProb, xVer, slackVer, verConstr = create_verif_problem_scp(milp_expr, lG, Kp, init_den_lb=init_den_lb, 
# 							init_den_ub=init_den_ub, verbose=verbose)

# 	# Get the node ids and swarm ids
# 	swarm_ids = lG.eSubswarm.keys()
# 	node_ids = lG.V.copy()

# 	# In case the trust region maxnimum value is not given, take the initial value as the undelrying value
# 	if trust_max is None:
# 		trust_max = trust_init

# 	# Initialization of the SCP with the uniform Markov matrix
# 	dictMk = find_initial_feasible_Markov(lG, Kp)
# 	initRange = {(s_id, n_id,m_id,t) : (Mlin[(s_id, n_id,m_id,t)].lb, Mlin[(s_id, n_id,m_id,t)].ub) \
# 					for s_id in swarm_ids for m_id in node_ids for t in range(Kp) for n_id in node_ids}

# 	# Solve the verification step to find the closest satisfiable xk by this Markov matrix
# 	dictXk, specValue = verification_step_scp(verProb, verConstr, xVer, dictMk,  Kp, swarm_ids, node_ids)
# 	# verProb.display()	

# 	curr_cost = get_cost_function(cost_fun, dictXk, dictMk, swarm_ids, Kp, node_ids) + mu_spec * specValue

# 	# Initialize the trust region
# 	rk = trust_init
# 	# Solving parameters and status
# 	solve_time = 0
# 	status = -1
# 	for i in range(maxIter):
# 		# Check if teh time limit is elapsed
# 		if solve_time >= timeLimit:
# 			status = -1
# 			break
# 		# Update the optimization problem and solve it
# 		update_linearized_problem_scp(linProb, xLin, Mlin, initRange, linConstr, rk, dictXk, dictMk, node_ids)
# 		# linProb.update()
# 		# linProb.display()
# 		# Measure optimization time
# 		cur_t = time.time()
# 		linProb.optimize()
# 		# linProb.display()
# 		solve_time += time.time() - cur_t
# 		if linProb.status == gp.GRB.OPTIMAL:
# 			status = 1
# 		elif linProb.status == gp.GRB.TIME_LIMIT:
# 			status = -1
# 			break
# 		else:
# 			status = 0
# 			break
# 		dictMk1 = {(s_id, n_id, m_id, t) : Mval.x for (s_id, n_id, m_id, t), Mval in Mlin.items()}
# 		dictXk1, specValue1 = verification_step_scp(verProb, verConstr, xVer, dictMk1,  Kp, swarm_ids, node_ids)
# 		# verProb.display()
# 		curr_cost_1 = get_cost_function(cost_fun, dictXk1, dictMk1, swarm_ids, Kp, node_ids) + mu_spec * specValue1
# 		# print('[Iteration {}] : Spec Opt. = {}'.format(i, xLin[(-1, BOOL_VAR-5)].x))
# 		linSpec = 0
# 		for (id_v, t2), sV in slackVer.items():
# 			print ('|Slack|[{}] = {}'.format(id_v, sV.x)) 
# 			linSpec += xLin[(id_v, t2)].x
# 		if verbose:
# 			print('Lin spec sat {}'.format(linSpec))
# 			print('[Iteration {}] : Old Opt. Cost = {}, Opt. Cost + spec sat = {}, Spec Sat = {}, Solve Time = {}'.format(i, curr_cost-mu_spec * specValue, curr_cost, specValue, solve_time))
# 			print('[Iteration {}] : Opt. Cost = {}, Opt. Cost + spec sat = {}, Spec Sat = {}, Solve Time = {}'.format(i, curr_cost_1-mu_spec * specValue1, curr_cost_1, specValue1, solve_time))
# 		if curr_cost_1 < curr_cost:
# 			dictMk = dictMk1
# 			dictXk = dictXk1
# 			curr_cost = curr_cost_1
# 			specValue = specValue1
# 			# rk = np.minimum(((rk - 1) * trust_mult + 1), trust_max)
# 			rk = np.minimum(rk * trust_mult, trust_max)
# 			if verbose:
# 				print('[Iteration {}] : Step Accepted, New Trust Region = {}'.format(i, rk))
# 		else:
# 			# rk = np.maximum(((rk - 1) / trust_div + 1), trust_lim)
# 			rk = np.maximum(rk / trust_div, trust_lim)
# 			if verbose:
# 				print('[Iteration {}] : Step Rejected, New Trust Region = {}'.format(i, rk))
# 		if rk <= trust_lim:
# 			if verbose:
# 				print('[Iteration {}] : Minimum trust region attiend, breaking.....'.format(i))
# 			break

# 	# linProb.display()
# 	# verProb.display()


def update_opt_step_k(mOpt, xDen, MarkovM, initRange, dictConstr, trustRegion, xk, Mk, swam_ids, Kp, node_ids):
	"""
		Update the optimization problem based on the optimal solution at the past iteration
		given by xk and Mk
	"""
	# Set the constraint limits on M imposed by the trust region
	for s_id in swam_ids:
		for n_id in node_ids:
			for t in range(Kp):
				xDen[(s_id, n_id, t+1)].lb = \
						np.maximum(xk[(s_id, n_id,t+1)]-trustRegion, 0)
				xDen[(s_id, n_id, t+1)].ub = \
						np.minimum(xk[(s_id, n_id,t+1)]+trustRegion, 1)
				for m_id in node_ids:
					MarkovM[(s_id, n_id, m_id, t)].lb = \
						np.maximum(Mk[(s_id, n_id, m_id, t)]-trustRegion, initRange[(s_id, n_id, m_id, t)][0])
					MarkovM[(s_id, n_id, m_id, t)].ub = \
						np.minimum(Mk[(s_id, n_id, m_id, t)]+trustRegion, initRange[(s_id, n_id, m_id, t)][1])

	for (s_id, n_id, t), c in dictConstr.items():
		c.RHS = -sum([Mk[(s_id, n_id, m_id, t)]*xk[(s_id, m_id, t)] for m_id in node_ids])
		for m_id in node_ids:
			mOpt.chgCoeff(c, MarkovM[(s_id, n_id, m_id, t)], -xk[(s_id, m_id, t)])
			mOpt.chgCoeff(c, xDen[(s_id, m_id, t)], - Mk[(s_id, n_id, m_id, t)])


def compute_actual_penalized_cost(cost_fun, lam, xk, Mk, swam_ids, Kp, node_ids):
	""" Compute the penalized cost function based on the constraint x(t+1) = M(t) x(t)
		In the paper, this penalized cost is referred to as J
	"""
	costVal = get_cost_function(cost_fun, xk, Mk, swam_ids, Kp, node_ids)
	penList = list()
	for s_id in swam_ids:
		for t in range(Kp):
			for n_id in node_ids:
				penList.append(np.abs(xk[(s_id,n_id,t+1)] \
								- sum([ Mk[(s_id,n_id,m_id,t)]* xk[(s_id, m_id, t)] \
										for m_id in node_ids])))
	return costVal + lam * sum(penList)


def sequential_optimization(mOpt, cost_fun, xDen, MarkovM, dictConstr, swam_ids, Kp, 
			node_ids, maxIter=20, epsTol=1e-5, lamda=10, devM=0.5, rho0=1e-5, 
			rho1=0.5, rho2=0.8, alpha=2.0, beta=1.5, 
			timeLimit=5, n_thread=0, verbose=True, autotune=True):
	""" Solve the sequential mixed integer linear program
	:param maxIter : The maximum number of iteration of the algorithm
	:param epsTol : The desired tolerance of the optimal solution
	:param lamda : Penalty cost for violating the state evolution constraint
	:param devM : Initial deviation between two consecutive iterates in control
	:param rho0 : rho0 < rho1 <<1, Threshold to consider the aproximation too inaccurate -> shrink
	:param rho1 : rho1 < rho2, Threshold to consider the approximation good (so accept current step) but need to shrink the trust region
	:param rho2 : 0 << rho2 < 1, Threshold to consider the approx good and increment the trust region
	:param alpha : alpha > 1, factor used to shrink the trust region
	:param beta : beta > 1, factor used to inflate the trust region
	"""
	dictXk = dict()
	dictMk = dict()
	initRange = dict()
	# Save the initial point that do not need to be feasible
	for s_id in swam_ids:
		for m_id in node_ids:
			for t in range(Kp+1):
				dictXk[(s_id, m_id, t)] = np.random.rand()
				# dictXk[(s_id, m_id, t)] = 0.5 *(xDen[(s_id, m_id, t)].lb + xDen[(s_id, m_id, t)].ub)
				if t == Kp:
					continue
				countEnforced = 0
				for n_id in node_ids:
					countEnforced += MarkovM[(s_id, n_id,m_id,t)].lb
					# dictMk[(s_id, n_id,m_id,t)] = MarkovM[(s_id, n_id,m_id,t)].lb
					dictMk[(s_id, n_id,m_id,t)] = np.random.rand()
					# dictMk[(s_id, n_id,m_id,t)] = 0.5*(MarkovM[(s_id, n_id,m_id,t)].lb + MarkovM[(s_id, n_id,m_id,t)].ub)
					initRange[((s_id, n_id,m_id,t))] = (MarkovM[(s_id, n_id,m_id,t)].lb, MarkovM[(s_id, n_id,m_id,t)].ub)
				# assert countEnforced < 1
				# residueV = 1 - countEnforced
				# for n_id in node_ids:
				# 	if dictMk[(s_id, n_id,m_id,t)] + residueV >= MarkovM[(s_id, n_id,m_id,t)].lb and \
				# 		dictMk[(s_id, n_id,m_id,t)] + residueV <= MarkovM[(s_id, n_id,m_id,t)].ub:
				# 		dictMk[(s_id, n_id,m_id,t)] = dictMk[(s_id, n_id,m_id,t)]  + residueV
				# 		break

	# INitialize the trust region
	rk = devM
	Jk = None
	solve_time = 0
	status = -1
	# Set the limit on the time limit
	mOpt.Params.OutputFlag = verbose
	mOpt.Params.Threads = n_thread
	mOpt.Params.TimeLimit = timeLimit
	# mOpt.Params.Presolve = 2 # More aggressive presolve step
	# # mOpt.Params.Crossover = 0
	# # mOpt.Params.CrossoverBasis = 0
	# mOpt.Params.NumericFocus = 3 # Maximum numerical focus
	# mOpt.Params.FeasibilityTol = 1e-6
	# mOpt.Params.OptimalityTol = 1e-6
	# mOpt.Params.BarConvTol = 1e-6

	for iterV in range(maxIter):
		# Check if teh time limit is elapsed
		if solve_time >= timeLimit:
			status = -1
			break
		# Update the optimization problem and solve it
		update_opt_step_k(mOpt, xDen, MarkovM, initRange, dictConstr, rk, dictXk, dictMk, swam_ids, Kp, node_ids)
		if iterV == 0 and autotune:
			mOpt.update()
			mOpt.tune()
		cur_t = time.time()
		mOpt.optimize()
		solve_time += time.time() - cur_t
		if mOpt.status == gp.GRB.OPTIMAL:
			status = 1
		elif mOpt.status == gp.GRB.TIME_LIMIT:
			status = -1
			break
		else:
			status = 0
			break
		dictXk1 = dict()
		dictMk1 = dict()
		for s_id in swam_ids:
			for n_id in node_ids:
				for t in range(Kp+1):
					dictXk1[(s_id, n_id, t)] = xDen[(s_id,n_id,t)].x
					if t == Kp:
						continue
					for m_id in node_ids:
						dictMk1[(s_id, n_id,m_id,t)] = MarkovM[(s_id,n_id,m_id,t)].x
		# for s_id in swam_ids:
		# 	for t in range(Kp):
		# 		for n_id in node_ids:
		# 			print (s_id, t, n_id, xDen[(s_id, n_id,t)].x)
		# Get the optimal cost
		Lk1 = mOpt.objVal
		if Jk is None:
			Jk = compute_actual_penalized_cost(cost_fun, lamda, dictXk, dictMk, swam_ids, Kp, node_ids)
		# Jk, _ = compute_actual_penalized_cost(cost_fun, lamda, dictXk, dictMk, swam_ids, Kp, node_ids)
		Jk1 = compute_actual_penalized_cost(cost_fun, lamda, dictXk1, dictMk1, swam_ids, Kp, node_ids)
		deltaJk = Jk - Jk1
		deltaLk = Jk - Lk1

		# If the solution is good enough
		if np.abs(deltaJk) < epsTol:
			dictXk = dictXk1
			dictMk = dictMk1
			break


		# If the approximation is too loose - contract the trust region and restart the otpimization
		rhok = deltaJk / deltaLk
		if verbose:
			print('Iteration : {}, Rhok : {}, rk : {}'.format(iterV, rhok, rk))
			print('Lk : {}, Lk+1 : {}'.format(Jk, Lk1))
			print('Jk : {}, Jk+1 : {}'.format(Jk, Jk1))
			print('deltaJk : ', deltaJk)
			print('deltaLk : ', deltaLk)
			# print('Jk+1 : ', Jk1)
		if rhok>0 and rhok < rho0:
			rk = rk / alpha
			continue

		# Accept this step
		# dictXk = trueXk
		dictXk = dictXk1
		dictMk = dictMk1
		Jk = Jk1

		# UPdate the trust region if needed
		rk = rk / alpha if rhok < rho1 else (beta*rk if rho2 < rhok else rk)
		rk = np.minimum(rk, 1) # rk should be less than 1 due to the Markov matrix values
		if rk < 1e-6:
			print(' Algorithm stopped : {}<1e-6, trust region too small'.format(rk))
			break
	# return dictXk, dictMk1
	return status, solve_time


def create_den_constr(xDen, swam_ids, Kp, node_ids):
	""" Given symbolic variables (gurobi, pyomo, etc..) representing the densities
		distribution at all times and all nodes, this function returns the 
		constraint 1^T x^s(t) = 1 for all s and t
	"""
	listConstr = list()
	for s_id in swam_ids:
		for t in range(Kp+1):
			listConstr.append(sum([ xDen[(s_id, n_id, t)] for n_id in node_ids]) == 1)
	return listConstr

def create_markov_constr(MarkovM, swam_ids, Kp, node_ids):
	""" Given symbolic variables (gurobi, pyomo, etc..) representing the Markov matrices
		at all times, this function returns the constraint 1^T M^s(t) = 1 for all s and t
		and M_ij^s(t) == 0 for (j,i) not in the edge list
	"""
	listConstr = list()
	# Stochastic natrix constraint
	for s_id in swam_ids:
		for t in range(Kp):
			for n_id in node_ids:
				listConstr.append(sum([MarkovM[(s_id,m_id,n_id,t)] for m_id in node_ids]) == 1)
	# Adjacency natrix constraints
	# for s_id, edges in edgeSwarm.items():
	# 	for t in range(Kp):
	# 		for (n_id, m_id) in edges:
	# 			listConstr.append(MarkovM[(s_id, m_id, n_id, t)] == 0)

	return listConstr

def create_bilinear_constr(xDen, MarkovM, swam_ids, Kp, node_ids):
	""" Return the bilinear constraint M(t) x(t) = x(t+1)
	"""
	listConstr = list()
	for s_id in swam_ids:
		for t in range(Kp):
			for n_id in node_ids:
				listConstr.append(sum([ MarkovM[(s_id,n_id,m_id,t)]* xDen[(s_id, m_id, t)] for m_id in node_ids])\
									== xDen[(s_id, n_id, t+1)])
	return listConstr


def get_cost_function(cost_fun, xDen, MarkovM, swam_ids, Kp, node_ids):
	""" Return the cost function over the graph trajectory length
	"""
	costVal = 0
	if cost_fun is not None:
		for t in range(Kp):
			xDict = dict()
			MDict = dict()
			for s_id in swam_ids:
				for n_id in node_ids:
					xDict[(s_id, n_id)] = xDen[(s_id, n_id, t)]
					for m_id in node_ids:
						MDict[(s_id, n_id, m_id)] = MarkovM[(s_id, n_id, m_id, t)]
			costVal += cost_fun(xDict, MDict, swam_ids, node_ids)
	return costVal


def create_minlp_model(util_funs, milp_expr, lG, Kp, init_den_lb=None, init_den_ub=None, 
							cost_fun=None, addBilinear=True, solve=False):
	# Obtain the milp encoding
	(newCoeffs, newVars, rhsVals, nVar), (lCoeffs, lVars, lRHS) = milp_expr

	# Get the boolean and continuous variables from the MILP expressions of the GTL
	bVars, rVars, lVars, denVars = getVars(newVars, lVars)

	# Swarm and graph configuration information
	swarm_ids = lG.eSubswarm.keys()
	node_ids = lG.V.copy()

	# Create the different density variables
	xDen = dict()
	for s_id in swarm_ids:
		for n_id in node_ids:
			for t in range(Kp+1):
				if t == 0 and init_den_lb is not None and init_den_ub is not None:
					xDen[(s_id, n_id, t)] = util_funs['r']('x[{}][{}]({})'.format(s_id, n_id, t),
												init_den_lb[s_id][n_id], init_den_ub[s_id][n_id])
				else:
					xDen[(s_id, n_id, t)] = util_funs['r']('x[{}][{}]({})'.format(s_id, n_id, t))
	# mOpt.addVar(lb=0, ub=1, name='x[{}][{}]({})'.format(s_id, n_id, t))
	
	# Create the different Markov matrices
	MarkovM = dict()
	for s_id in swarm_ids:
		nedges = lG.getSubswarmNonEdgeSet(s_id)
		for n_id in node_ids:
			for m_id in node_ids:
				for t in range(Kp):
					if (m_id, n_id) in nedges:
						MarkovM[(s_id, n_id, m_id, t)] = \
							util_funs['r']('M[{}][{}][{}]({})'.format(s_id, n_id, m_id, t),0,0)
					else:
						MarkovM[(s_id, n_id, m_id, t)] = \
							util_funs['r']('M[{}][{}][{}]({})'.format(s_id, n_id, m_id, t))
	# mOpt.addVar(lb=0, ub=1, name='M[{}][{}][{}]({})'.format(s_id, n_id, m_id, t))
	
	# Create the boolean variables from the GTL formula -> add them to the xDen dictionary
	for ind in bVars:
		xDen[ind] = util_funs['b']('b[{}]'.format(ind[0]))
		# mOpt.addVar(vtype=gp.GRB.BINARY, name='b[{}]({})'.format(ind0, tInd))

	# Create the tempory real variables from the GTL formula -> add them to the xDen dictionary
	for ind in rVars:
		xDen[ind] = util_funs['r']('r[{}]'.format(ind[0]))
		# mOpt.addVar(lb=0, ub=1.0, name='r[{}]({})'.format(ind0, tInd))

	# Create the binary variables from the loop constraint
	for ind in lVars:
		xDen[ind] = util_funs['b']('l[{}]'.format(ind[0]))
		# mOpt.addVar(vtype=gp.GRB.BINARY, name='l[{}]'.format(ind0))

	# Add the constraint on the density distribution
	listContr = create_den_constr(xDen,swarm_ids, Kp, node_ids)

	# Add the constraints on the Markov matrices
	listContr.extend(create_markov_constr(MarkovM, swarm_ids, Kp, node_ids))

	# Add the bilinear constraints
	if addBilinear:
		listContr.extend(create_bilinear_constr(xDen, MarkovM, swarm_ids, Kp, node_ids))

	# Add the mixed-integer constraints
	listContr.extend(create_gtl_constr(xDen, milp_expr))

	# Add all the constraints
	for constr in listContr:
		util_funs['constr'](constr)

	costVal = get_cost_function(cost_fun, xDen, MarkovM, swarm_ids, Kp, node_ids)
	# set_cost(cVal, xDen, mD, nids, sids)
	util_funs['cost'](costVal, xDen, MarkovM, swarm_ids, node_ids)

	# Solve the problem
	optCost, status, solveTime = -np.inf, -1, 0
	if solve:
		optCost, status, solveTime = util_funs['solve']()

	# Get the solution of the problem
	xDictRes = dict()
	MDictRes = dict()
	ljRes = dict()
	if solve and status == 1:
		for s_id in swarm_ids:
			for n_id in node_ids:
				for t in range(Kp+1):
					xDictRes[(s_id, n_id, t)] = \
						util_funs['opt'](xDen[(s_id, n_id, t)])
		# mOpt.addVar(lb=0, ub=1, name='x[{}][{}]({})'.format(s_id, n_id, t))
		
		for s_id in swarm_ids:
			for n_id in node_ids:
				for m_id in node_ids:
					for t in range(Kp):
						MDictRes[(s_id, n_id, m_id, t)] = \
								util_funs['opt'](MarkovM[(s_id, n_id, m_id, t)])

		for ind0 in lVars:
			ljRes[ind0] = util_funs['opt'](xDen[ind0])
	return optCost, status, solveTime, xDictRes, MDictRes, ljRes, swarm_ids, node_ids


def create_minlp_gurobi(milp_expr, lG, Kp, init_den_lb=None, init_den_ub=None, 
						cost_fun=None, solve=True, timeLimit=5, n_thread=0, verbose=False):
	# Create the Gurobi model
	mOpt = gp.Model('Bilinear MINLP formulation through GUROBI')
	mOpt.Params.OutputFlag = verbose
	mOpt.Params.NonConvex = 2
	mOpt.Params.Threads = n_thread
	mOpt.Params.TimeLimit = timeLimit
	def set_cost(cVal, xDen, mD, sids, nids):
		mOpt.setObjective(cVal, gp.GRB.MINIMIZE)
	def solve_f():
		c_time = time.time()
		mOpt.optimize()
		dur = time.time() - c_time
		if mOpt.status == gp.GRB.OPTIMAL:
			return mOpt.objVal, 1, dur
		elif mOpt.status == gp.GRB.TIME_LIMIT:
			return mOpt.objVal, -1, dur
		else:
			return -np.inf, 0, dur
	def ret_sol(solV):
		return solV.x

	util_funs = dict()
	util_funs['r'] = lambda name, lb=0, ub=1 : mOpt.addVar(lb=lb, ub=ub, name=name)
	util_funs['b'] = lambda name : mOpt.addVar(vtype=gp.GRB.BINARY, name=name)
	util_funs['constr'] = lambda constr : mOpt.addConstr(constr)
	util_funs['solve'] = solve_f
	util_funs['cost'] = set_cost
	util_funs['opt'] = ret_sol
	return create_minlp_model(util_funs, milp_expr, lG, Kp, init_den_lb, init_den_ub, 
								addBilinear=True, cost_fun=cost_fun, solve=solve)
	# mOpt.update()
	# mOpt.display()

def create_gtl_proco(milp_expr, lG, Kp, init_den_lb=None, init_den_ub=None,
					cost_fun=None, solve=True, 
					maxIter=20, epsTol=1e-5, lamda=10, devM=0.5, rho0=1e-5, 
					rho1=0.25, rho2=0.7, alpha=2.0, beta=3.5,
					timeLimit=5, n_thread=0, verbose=True, autotune=True):
	# Create the Gurobi model,
	mOpt = gp.Model('Sequential MILP solving through GUROBI')
	# mOpt.Params.OutputFlag = True

	# Actual cost 
	costVal = None
	dictConstr = dict()
	dictSlack = dict()
	xD = None
	MarM = None
	n_ids, s_ids = None, None
	def set_cost(cVal, xDen, mD, sids, nids):
		nonlocal costVal, dictSlack, dictConstr, xD, MarM, n_ids, s_ids
		n_ids = nids
		s_ids = sids
		costVal = cVal
		xD = xDen
		MarM = mD
		# Create the linearized constraints
		create_linearize_constr(mOpt, xD, MarM, dictSlack, dictConstr, s_ids, Kp, n_ids)
		# Add the penalized cost functions
		penList = list()
		for s_id in s_ids:
			for t in range(Kp):
				for n_id in n_ids:
					penList.append(dictSlack[(s_id, -n_id-1,t)])
					penList.append(dictSlack[(s_id, n_id,t)])
		pen_cost = costVal + lamda * sum(penList)
		mOpt.setObjective(pen_cost, gp.GRB.MINIMIZE)
		mOpt.update()
		# mOpt.display()

	def solve_f():
		status, dur = sequential_optimization(mOpt, cost_fun, xD, MarM, dictConstr, s_ids, Kp, n_ids,
						maxIter=maxIter, epsTol=epsTol, lamda=lamda, devM=devM, rho0=rho0, 
						rho1=rho1, rho2=rho2, alpha=alpha, beta=beta,
						timeLimit=timeLimit, n_thread=n_thread, verbose=verbose, autotune=autotune)
		if status == 0:
			return -np.inf, status, dur
		return costVal.getValue() if not (isinstance(costVal, int) or isinstance(costVal, float)) else costVal, status, dur

	def ret_sol(solV):
		return solV.x

	util_funs = dict()
	util_funs['r'] = lambda name, lb=0, ub=1 : mOpt.addVar(lb=lb, ub=ub, name=name)
	util_funs['b'] = lambda name : mOpt.addVar(vtype=gp.GRB.BINARY, name=name)
	util_funs['constr'] = lambda constr : mOpt.addConstr(constr)
	util_funs['solve'] = solve_f
	util_funs['cost'] = set_cost
	util_funs['opt'] = ret_sol
	return create_minlp_model(util_funs, milp_expr, lG, Kp, init_den_lb, init_den_ub, 
								addBilinear=False, cost_fun=cost_fun, solve=solve)
	# mOpt.update()
	# mOpt.display()

def create_minlp_scip(milp_expr, lG, Kp, init_den_lb=None, init_den_ub=None,
						cost_fun=None, solve=True,
						timeLimit=5, n_thread=0, verbose=False):
	import pyscipopt as pSCIP
	mOpt = pSCIP.Model('Bilinear MINLP formulation through SCIP')
	mOpt.hideOutput(quiet = (not verbose))
	mOpt.setParam('limits/time', timeLimit)
	# mOpt.setParam('')
	def set_cost(cVal, xDen, mD, sids, nids):
		mOpt.setObjective(cVal, "minimize")
	def solve_f():
		c_time = time.time()
		mOpt.optimize()
		dur = time.time() - c_time
		if mOpt.getStatus() == 'optimal':
			return mOpt.getObjVal(), 1, dur
		elif mOpt.getStatus() == 'timelimit':
			return mOpt.getObjVal(), -1, dur
		else:
			return -np.inf, 0, dur
	def ret_sol(solV):
		return mOpt.getVal(solV)
	util_funs = dict()
	util_funs['r'] = lambda name, lb=0, ub=1 : mOpt.addVar(lb=lb, ub=ub, name=name)
	util_funs['b'] = lambda name : mOpt.addVar(vtype="B", name=name)
	util_funs['constr'] = lambda constr : mOpt.addCons(constr)
	util_funs['solve'] = solve_f
	util_funs['cost'] = set_cost
	util_funs['opt'] = ret_sol
	return create_minlp_model(util_funs, milp_expr, lG, Kp, init_den_lb, init_den_ub,
								 addBilinear=True, cost_fun=cost_fun, solve=solve)
	# mOpt.writeProblem('model')

def create_minlp_pyomo(milp_expr, lG, Kp, init_den_lb=None, init_den_ub=None, 
			cost_fun=None, solve=True, 
			solver='couenne', solverPath='/home/fdjeumou/Documents/non_convex_solver/',
			timeLimit=5, n_thread=0, verbose=False):
	from pyomo.environ import Var, ConcreteModel, Constraint, NonNegativeReals, Binary, SolverFactory
	from pyomo.environ import Objective, minimize
	import pyutilib
	from pyomo.opt import SolverStatus, TerminationCondition

	mOpt = ConcreteModel('Bilinear MINLP formulation through PYOMO')

	# Function to return real values
	def r_values(name, lb=0, ub=1):
		setattr(mOpt, name, Var(bounds=(lb,ub), within=NonNegativeReals))
		return getattr(mOpt, name)

	def b_values(name):
		setattr(mOpt, name, Var(within=Binary))
		return getattr(mOpt, name)

	nbConstr = 0
	def constr(val):
		nonlocal nbConstr
		setattr(mOpt, 'C_{}'.format(nbConstr), Constraint(expr = val))
		nbConstr+=1

	def set_cost(cVal, xDen, mD, sids, nids):
		setattr(mOpt, 'objective', Objective(expr=cVal, sense=minimize))

	def solve_f():
		solverOpt = SolverFactory(solver, executable=solverPath+solver)
		try:
			c_time = time.time()
			results = solverOpt.solve(mOpt, tee=verbose, timelimit=timeLimit)
			dur = time.time() - c_time
			if results.solver.termination_condition == TerminationCondition.optimal:
				return mOpt.objective(), 1, dur
			else:
				return -np.inf, 0, dur
		except pyutilib.common.ApplicationError:
			return -np.inf, -1, timeLimit

	def ret_sol(solV):
		return solV.value

	# mOpt.hideOutput()
	util_funs = dict()
	util_funs['r'] = r_values
	util_funs['b'] = b_values
	util_funs['constr'] = constr
	util_funs['solve'] = solve_f
	util_funs['cost'] = set_cost
	util_funs['opt'] = ret_sol
	return create_minlp_model(util_funs, milp_expr, lG, Kp, init_den_lb, init_den_ub,
								addBilinear=True, cost_fun=cost_fun, solve=solve)
	# mOpt.pprint()

def create_reach_avoid_problem_lp(gtlFormulas, nodes, desDens, lG, cost_fun=None, cost_M = None, 
					solve=True, timeLimit=5, n_thread=0, verbose=False):
	""" Compute a solution of the LP problem (34) to (41). A solution of such a problem
		provides a Markov Matrix that ensures the satisfcation of reach-avoid specifications
	"""

	# Create the optimization problem
	mOpt = gp.Model('LP formulation for reach-avoid specs')
	mOpt.Params.OutputFlag = verbose
	mOpt.Params.Threads = n_thread
	mOpt.Params.TimeLimit = timeLimit

	# Swarm and graph configuration information
	swarm_ids = lG.eSubswarm.keys()
	node_ids = lG.V.copy()

	# Create the different Markov matrices
	MarkovM = dict()
	for s_id in swarm_ids:
		nedges = lG.getSubswarmNonEdgeSet(s_id)
		for n_id in node_ids:
			for m_id in node_ids:
				if (m_id, n_id) in nedges: # Encode the motion constraints
					MarkovM[(s_id, n_id, m_id)] = \
						mOpt.addVar(lb=0, ub=0, name='M[{}][{}][{}]'.format(s_id, n_id, m_id))
				else:
					MarkovM[(s_id, n_id, m_id)] = \
						mOpt.addVar(lb=0, name='M[{}][{}][{}]'.format(s_id, n_id, m_id))

	# Add the Markov matrix constraints
	for s_id in swarm_ids:
		for n_id in node_ids:
			mOpt.addConstr(sum([MarkovM[(s_id,m_id,n_id)] for m_id in node_ids]) == 1)

	# Add the desired density constraint
	for s_id, sDen in desDens.items():
		for n_id, val in sDen.items():
			mOpt.addConstr(sum([MarkovM[(s_id,n_id,m_id)]*sDen[m_id] for m_id in sDen]) == val)

	# Add the constraints imposed by the safety formulas
	A, b, indexShift = get_safe_encoding(lG, gtlFormulas, nodes, swarm_ids, node_ids)

	# First add the variables Y
	dictY = dict()
	for i in range(b.shape[0]):
		for j in range(b.shape[0]):
			dictY[(i,j)] = mOpt.addVar(lb=-gp.GRB.INFINITY, ub=0, name='Y[{}][{}]'.format(i,j))

	# Then add the S variable
	dictS = dict()
	for i in range(b.shape[0]):
		for j in range(len(swarm_ids)):
			dictS[(i,j)] = mOpt.addVar(lb=-gp.GRB.INFINITY, name='S[{}][{}]'.format(i,j))

	# Define the big O matrix
	dictO = dict()
	for i in range(len(swarm_ids)):
		for j in range(len(node_ids)*i, len(node_ids)*(i+1)):
			dictO[(i,j)] = 1.0

	# Define diagM
	diagMDict  = dict()
	for i in range(0, A.shape[1], len(node_ids)):
		(s_id, node_rge) = indexShift[i]
		for j, v1 in enumerate(node_rge):
			for k, v2 in enumerate(node_rge):
				diagMDict[(j+i, k+i )] = MarkovM[(s_id, v1, v2)]
	
	# Add the constraints
	for i in range(b.shape[0]):
		mOpt.addConstr(
			sum([dictY[(i,j)]*b[j] for j in range(b.shape[0])]) + \
			sum(dictS[(i,j)] for j in range(len(swarm_ids)))
			>= -b[i]
		)

	for i in range(b.shape[0]):
		for j in range(A.shape[1]):
			t1 = sum( dictY[(i,k)]*A[k,j] for k in range(b.shape[0]))
			t2 = sum( dictS[(i,k)]*dictO.get((k,j), 0) for k in range(len(swarm_ids)) )
			t3 = sum( A[i,k]*diagMDict.get((k,j), 0) for k in range(A.shape[1]))
			mOpt.addConstr(t1 + t2 <= - t3)

	# mOpt.update()
	# mOpt.display()

	# Build the cost function
	costVal = 0
	if cost_fun is not None:
		costVal += cost_fun(MarkovM, swarm_ids, node_ids)
	if cost_M is None:
		cost_M = dict()
		for s_id in swarm_ids:
			cost_M[s_id] = 1.0

	# Add the coefficient of ergocity
	dictMax = dict()
	for s_id in swarm_ids:
		dictMax[s_id] = list()
		for n_id in node_ids:
			for m_id in node_ids:
				tempV = [mOpt.addVar(lb=0) for p in node_ids]
				for p, tV in zip(node_ids, tempV):
					mOpt.addConstr(tV >= MarkovM[(s_id, p,n_id)]-MarkovM[(s_id, p, m_id)])
					mOpt.addConstr(-tV <= MarkovM[(s_id, p,n_id)]-MarkovM[(s_id, p, m_id)])
				expVal = gp.quicksum([ tV for tV in tempV])
				dictMax[s_id].append(expVal)
	for s_id, cs in cost_M.items():
		tV = mOpt.addVar(lb=0)
		for contrV in dictMax[s_id]:
			mOpt.addConstr(tV >= contrV)
		costVal += 0.5*cs * tV

	# Set the optimal cost
	mOpt.setObjective(costVal)

	optCost, status, solveTime = -np.inf, -1, 0

	# Optimize of required
	if solve:
		c_t = time.time()
		mOpt.optimize()
		solveTime = time.time() - c_t
		if mOpt.status == gp.GRB.OPTIMAL:
			optCost, status, solveTime =  mOpt.objVal, 1, solveTime
		elif mOpt.status == gp.GRB.TIME_LIMIT:
			optCost, status, solveTime =  mOpt.objVal, -1, solveTime
		else:
			optCost, status, solveTime =  -np.inf, 0, solveTime

	# Save the resulting Markov matrix
	MDictRes = dict()
	if solve and status == 1:
		for s_id in swarm_ids:
			for n_id in node_ids:
				for m_id in node_ids:
						MDictRes[(s_id, n_id, m_id)] = MarkovM[(s_id, n_id, m_id)].x

	return optCost, status, solveTime, MDictRes

def create_reach_avoid_problem_convex(gtlFormulas, nodes, desDens, lG, cost_fun=None, cost_M = None, 
					solve=True, timeLimit=5, n_thread=0, verbose=False, sdp_solver=True):
	""" Compute a solution of the LP problem (34) to (41). A solution of such a problem
		provides a Markov Matrix that ensures the satisfcation of reach-avoid specifications
	"""

	# Swarm and graph configuration information
	swarm_ids = lG.eSubswarm.keys()
	node_ids = lG.V.copy()

	# List of constraints
	pbConstr = list()

	# Create the different Markov matrices
	MarkovM = dict()
	for s_id in swarm_ids:
		MarkovM[s_id] = cp.Variable((len(node_ids), len(node_ids)), nonneg=True)
		nedges = lG.getSubswarmNonEdgeSet(s_id)
		for i, n_id in enumerate(node_ids):
			for j, m_id in enumerate(node_ids):
				if (m_id, n_id) in nedges: # Encode the motion constraints
					pbConstr.append(MarkovM[s_id][i, j] == 0)

	# Add the Markov matrix constraints
	for s_id in swarm_ids:
		pbConstr.append(np.ones(len(node_ids))@MarkovM[s_id] == np.ones(len(node_ids)))

	# Add the desired density constraint
	for s_id, sDen in desDens.items():
		vs = np.array([sDen[n_id] for i, n_id in enumerate(node_ids)])
		pbConstr.append(MarkovM[s_id] @ vs == vs)

	# Add the constraints imposed by the safety formulas
	A, b, indexShift = get_safe_encoding(lG, gtlFormulas, nodes, swarm_ids, node_ids)

	# First add the variables Y
	Ym = cp.Variable((b.shape[0], b.shape[0]), nonpos=True)

	# Then add the S variable
	Sm = cp.Variable((b.shape[0], len(swarm_ids)))

	# Define the big O matrix
	Om = np.zeros((len(swarm_ids), len(node_ids)*len(swarm_ids)))
	for i, s_id in enumerate(swarm_ids):
		Om[i,len(node_ids)*i:len(node_ids)*(i+1)] = 1.0

	# Define diagM
	diagMl  = list()
	for i in range(0, A.shape[1], len(node_ids)):
		(s_id, node_rge) = indexShift[i]
		rowList = list()
		for j, v1 in enumerate(node_rge):
			colList = [0 for _ in range(i)]
			for k, v2 in enumerate(node_rge):
				colList.append(MarkovM[s_id][j, k])
			colList.extend([0 for _ in range(i+len(node_ids), len(node_ids)*len(swarm_ids))])
			rowList.append(colList)
		diagMl.extend(rowList)
	diagM = cp.bmat(diagMl)
	
	# Add the constraints
	pbConstr.append(Ym @ b + Sm @ np.ones(len(swarm_ids)) >= -b)
	pbConstr.append(Ym @ A + Sm @ Om + A @ diagM <= 0)


	# Build the cost function
	costVal = 0
	if cost_fun is not None:
		costVal += cost_fun(MarkovM, swarm_ids, node_ids)

	if cost_M is None:
		cost_M = dict()
		for s_id in swarm_ids:
			cost_M[s_id] = 1.0

	if sdp_solver:
		for s_id, cs in cost_M.items():
			Ms = MarkovM[s_id]
			qinv = np.zeros((len(node_ids), len(node_ids)))
			qval = np.zeros((len(node_ids), len(node_ids)))
			rval =  np.zeros((1, len(node_ids)))
			for i, n_id in enumerate(node_ids):
				rval[0,i] = np.sqrt(desDens[s_id][n_id])
				qinv[i,i] = 0 if rval[0,i] < 1e-9 else (1.0/rval[0,i])
				qval[i,i] = rval[0,i]
			costVal += cs * cp.norm(qinv @ Ms @ qval - rval.T @ rval)
	else:
		for s_id, cs in cost_M.items():
			listMaxM = list()
			Ms = MarkovM[s_id]
			for i, n_id in enumerate(node_ids):
				colList = list()
				for j, m_id in enumerate(node_ids):
					colList.append(cp.norm(Ms[:,i]-Ms[:,j],1))
				listMaxM.append(colList)
			costVal += 0.5 * cs * cp.max(cp.bmat(listMaxM))

	# Set the optimal cost
	objVal = cp.Minimize(costVal)
	prob = cp.Problem(objVal, pbConstr)

	optCost, status, solveTime = -np.inf, -1, 0

	# Optimize of required
	if solve:
		if sdp_solver:
			import mosek
			prob.solve(solver=cp.MOSEK, verbose=verbose, 
						mosek_params={mosek.dparam.optimizer_max_time:timeLimit, 
								mosek.iparam.intpnt_solve_form:mosek.solveform.dual,
								mosek.iparam.num_threads:n_thread})
		else:
			opts = {'Threads' : n_thread, 'TimeLimit' : timeLimit}
			prob.solve(solver=cp.GUROBI, verbose=verbose, **opts)
		if prob.status == cp.OPTIMAL:
			optCost, status, solveTime =  prob.value, 1, prob.solver_stats.solve_time
		elif prob.status == cp.SOLVER_ERROR:
			optCost, status, solveTime =  prob.value, -1, prob.solver_stats.solve_time
		else:
			optCost, status, solveTime =  -np.inf, 0, prob.solver_stats.solve_time

	# Save the resulting Markov matrix
	MDictRes = dict()
	if solve and status == 1:
		for s_id in swarm_ids:
			for i, n_id in enumerate(node_ids):
				for j, m_id in enumerate(node_ids):
						MDictRes[(s_id, n_id, m_id)] = MarkovM[s_id][i, j].value

	return optCost, status, solveTime, MDictRes

if __name__ == "__main__":
    """
    Example of utilization of this class
    """
    V = set({1, 2, 3})
    lG = LabelledGraph(V)

    # Add the subswarm with id 0
    lG.addSubswarm(0, [(1,2), (2,3), (2,1), (3,2)])

    # Add the subswarm with id 1
    lG.addSubswarm(1, [(1,2), (2,3), (2,1), (3,2)])

    # Get the density symbolic representation for defining the node labels
    x = lG.getRprDensity()

    # Now add some node label on the graph
    lG.addNodeLabel(1, x[(0,1)] + x[(0,2)])
    lG.addNodeLabel(2, x[(0,1)] + x[(1,1)])
    lG.addNodeLabel(3, [x[(0,1)] + x[(0,2)], x[(0,3)] - x[(1,2)]])

    Kp = 3
    piV3 = AtomicGTL([0.5, 0.25])
    piV1 = AtomicGTL([0.5])
    formula1 = AlwaysEventuallyGTL(piV3)
    formula2 = AlwaysGTL(piV1)
    m_milp = create_milp_constraints([formula1, formula2], [3, 1], Kp, lG, initTime=0)

    def cost_fun(xDict, MDict, swarm_ids, node_ids):
    	costValue = 0
    	for s_id in swarm_ids:
    		for n_id in node_ids:
    			for m_id in node_ids:
    				costValue += MDict[(s_id, n_id, m_id)]
    	return costValue
    cost_fun = None
    # print_milp_repr([piV3], [3], Kp, lG, initTime=0)

    optCost, status, solveTime, xDictRes, MDictRes, ljRes, swarm_ids, node_ids = \
    						create_minlp_gurobi(m_milp, lG, Kp, cost_fun=cost_fun,
    						timeLimit=5, n_thread=1, verbose=False)
    print(ljRes)
    print(optCost, status, solveTime)
    maxDiff = 0
    for s_id in swarm_ids:
    	for n_id in node_ids:
    		for t in range(Kp):
    			s = sum(MDictRes[(s_id, n_id, m_id, t)]*xDictRes[(s_id, m_id,t)] for m_id in node_ids)
    			# print(np.abs(xDictRes[(s_id, n_id,t+1)]-s))
    			maxDiff = np.maximum(maxDiff, np.abs(xDictRes[(s_id, n_id,t+1)]-s))
    print(maxDiff)

    # create_minlp_scip(m_milp, lG, Kp)
    optCost, status, solveTime, xDictRes, MDictRes, ljRes, swarm_ids, node_ids = \
    								create_minlp_scip(m_milp, lG, Kp, cost_fun=cost_fun,
    									timeLimit=5, n_thread=0, verbose=False)
    print(ljRes)
    print(optCost, status, solveTime)
    maxDiff = 0
    for s_id in swarm_ids:
    	for n_id in node_ids:
    		for t in range(Kp):
    			s = sum(MDictRes[(s_id, n_id, m_id, t)]*xDictRes[(s_id, m_id,t)] for m_id in node_ids)
    			# print(np.abs(xDictRes[(s_id, n_id,t+1)]-s))
    			maxDiff = np.maximum(maxDiff, np.abs(xDictRes[(s_id, n_id,t+1)]-s))
    print(maxDiff)
    # print(xDictRes)

    optCost, status, solveTime, xDictRes, MDictRes, ljRes, swarm_ids, node_ids = gtl_proco_scp(m_milp, lG, Kp, 
    	cost_fun=cost_fun, verbose=True, maxIter=30, epsTol=1e-6, mu_lin=1)
    print(ljRes)
    print(optCost, status, solveTime)
    maxDiff = 0
    for s_id in swarm_ids:
    	for n_id in node_ids:
    		for t in range(Kp):
    			s = sum(MDictRes[(s_id, n_id, m_id, t)]*xDictRes[(s_id, m_id,t)] for m_id in node_ids)
    			# print(np.abs(xDictRes[(s_id, n_id,t+1)]-s))
    			maxDiff = np.maximum(maxDiff, np.abs(xDictRes[(s_id, n_id,t+1)]-s))
    print(maxDiff)

    # # create_minlp_pyomo(m_milp, lG, Kp)
    # optCost, status, solveTime, xDictRes, MDictRes, ljRes, swarm_ids, node_ids = \
    # 								create_minlp_pyomo(m_milp, lG, Kp,cost_fun=cost_fun,
    # 								timeLimit=5, n_thread=0, verbose=False)
    # print(ljRes)
    # print(optCost, status, solveTime)
    # print(xDictRes)
    # print(optCost)

    
    # optCost, status, solveTime, xDictRes, MDictRes, ljRes, swarm_ids, node_ids = \
    # 		create_gtl_proco(m_milp, lG, Kp, cost_fun=cost_fun, solve=True, 
				# 	maxIter=20, epsTol=1e-5, lamda=100, devM=0.5, rho0=1e-2, 
				# 	rho1=0.75, rho2=0.95, alpha=2.0, beta=1.5,
				# 	timeLimit=5, n_thread=1, verbose=True, autotune=False)
    # print(ljRes)
    # print(optCost, status, solveTime)
    # maxDiff = 0
    # for s_id in swarm_ids:
    # 	for n_id in node_ids:
    # 		for t in range(Kp):
    # 			s = sum(MDictRes[(s_id, n_id, m_id, t)]*xDictRes[(s_id, m_id,t)] for m_id in node_ids)
    # 			# print(np.abs(xDictRes[(s_id, n_id,t+1)]-s))
    # 			maxDiff = np.maximum(maxDiff, np.abs(xDictRes[(s_id, n_id,t+1)]-s))
    # print(maxDiff)