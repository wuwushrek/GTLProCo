from .LabelledGraph import LabelledGraph
from .GTLformula import *

# Import the optimizer tools
import gurobipy as gp
import cvxpy as cp
import time

def create_linearize_constr(mOpt, xDen, MarkovM, dictSlack, dictConstr, swam_ids, Kp, node_ids):
	""" Create a gurobi prototype of linearized nonconvex constraint
		around the solution of the previous iteration.
		Additionally, create the slack variables for the constraints
		:param mOpt : the Gurobi instance problem to solve
		:param xDen : the density distribution variable of the swarm of the gurobi problem
		:param MarkovM : the Markov matrix variable of the gurobi problem
		:param dictSlack : A dictionary to store the slack variables added to the linearized constraints
		:param dictConstr : A dictionary to stire the parameterized and linearized constraints
		:param swam_ids :A set containing the identifier for all the swarms
		:param Kp : the loop time horizon
		:param node_ids : A set containing the identifier of all the nodes of the graph
	"""
	# Create the slack variables to add to the linearized constraints
	for s_id in swam_ids:
		for t in range(Kp):
			for n_id in node_ids:
				dictSlack[(s_id, n_id, t)] = \
					mOpt.addVar(name='Z[{}][{}]({})'.format(s_id,n_id,t))
				dictSlack[(s_id, -n_id-1, t)] = \
					mOpt.addVar(name='nZ[{}][{}]({})'.format(s_id,n_id,t))

	# Create and add thelinearized constraints
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
	""" Create a L1-norm trust region type of constraints
		:param mopt :  the Gurobi instance problem to solve
		:param xDen : the density distribution variable of the swarm of the gurobi problem
		:param swam_ids :A set containing the identifier for all the swarms
		:param Kp : the loop time horizon
		:param node_ids : A set containing the identifier of all the nodes of the graph
	"""
	# Store the extra variable w_i^s(t) for all i in node_ids, s in swarm_ids and time t
	extraVar = {(s_id,n_id,t) : mOpt.addVar(lb=0, name='t[{}][{}][{}]'.format(s_id,n_id,t)) for s_id in swam_ids for t in range(Kp+1) for n_id in node_ids}
	# Store the constraints on the new variable |x_i^s(t) - xk_i^s(t)| <= w_i^s(t) for all t and swarm s, and index i
	dictConstrVar = {(s_id,n_id,t) : (mOpt.addLConstr( gp.LinExpr([1,-1], [xDen[s_id,n_id,t],extraVar[(s_id,n_id,t)]]), gp.GRB.LESS_EQUAL, 0),\
										mOpt.addLConstr( gp.LinExpr([-1,-1], [xDen[s_id,n_id,t],extraVar[(s_id,n_id,t)]]), gp.GRB.LESS_EQUAL, 0))\
						for s_id in swam_ids for t in range(Kp+1) for n_id in node_ids 
					}
	# Store the constraint sum_i w_i^s(t) <= trust region for all swarm s and time t
	dictConstrL1 = {(s_id, t) : mOpt.addLConstr(gp.LinExpr([1 for i in node_ids], [extraVar[(s_id,n_id,t)] for n_id in node_ids]), gp.GRB.LESS_EQUAL, 0) for s_id in swam_ids for t in range(Kp+1)}
	return extraVar, dictConstrVar, dictConstrL1


def find_initial_feasible_Markov(lG, Kp, xk, verbose=True):
	""" Given  the time-varying density distribution xk, this function computes the time-varying Markov matrix
		that satisfies as close as possible x(t+1) = M(t) x(t).
		That is, find M(t) solution of the optimization problem minimize lambda such that xk(t+1) =  M(t) xk(t) + lambda
		:param lG : The Labelled graph representation
		:param Kp : the loop time horizon
		:param xk : the current value of the density distribution
		:param verbose : specify if verbose when solving the optimization problem
	"""
	# Create the Gurobi model
	mOpt = gp.Model('Initial Markov matrix feasibility problem')
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
					# Motion constraints been enforced
					if (m_id, n_id) in nedges:
						MarkovM[(s_id, n_id, m_id, t)] = \
							mOpt.addVar(lb=0, ub=0, name='M[{}][{}][{}][{}]'.format(s_id, n_id, m_id,t))
					else:
						MarkovM[(s_id, n_id, m_id, t)] = \
							mOpt.addVar(lb=0, ub=1, name='M[{}][{}][{}][{}]'.format(s_id, n_id, m_id,t))

	# Add the constraints on the stochasticity of the Markov matrices
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

	# Set the objective function as the quantifier if the bilinear constraint
	mOpt.setObjective(gp.quicksum([slackVar for _,slackVar in slVar.items()]), gp.GRB.MINIMIZE)
	# Find the optimal solution of the problem
	curr_time = time.time()
	mOpt.optimize()
	solve_time = time.time() - curr_time
	# Collect the solution of the problem
	dictM = {(s_id, n_id, m_id, t) : Mval.x for (s_id, n_id, m_id,t), Mval in MarkovM.items()}
	return dictM, solve_time

def find_initial_feasible_density(milp_expr, lG, Kp, init_den_lb=None, init_den_ub=None, verbose=True):
	""" Compute an initial feasible time-varying density distribution to the GTL Constraints.
		:param milp_expr : A mixed integer representation of the GTL constraints
		:param lG : The labelled graph
		:param Kp : The loop time horizon
		:param init_den_lb : The lower bound on the density distribution at time 0
		:param init_den_ub : The upper bound on the density distribution at time 0
		:param verbose : Specify if verbose when solving the optimization problem
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
				# Initial distribution range being enforced
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

	# Trivial constraints from the Markov matrix x(t+1) <= A^t x(t) and x(t) <= A^t x(t+1)
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
	curr_time = time.time()
	mOpt.optimize()
	solve_time = time.time() - curr_time
	# mOpt.display()
	resxk = {kVal : xVal.x for kVal, xVal in xDen.items()}
	return resxk, solve_time

def create_linearized_problem_scp(milp_expr, lG, Kp, cost_fun, init_den_lb=None, init_den_ub=None,
									timeLimit=5, n_thread=0, verbose=True, mu_lin=10, mu_period=1):
	""" Create the linearized problem to solve at each iteration of the sequential convex programming scheme
		:param milp_expr :  A mixed integer representation of the GTL constraints
		:param lG : The labelled graph
		:param Kp : The loop time horizon 
		:param cost_fun : The cost function to optimize as a fujnction of the Markov martix and the density distribution
		:param init_den_lb : The lower bound on the density distribution at time 0
		:param init_den_ub : The upper bound on the density distribution at time 0
		:param timeLimit : The time limit at each time step f the solver
		:param n_thread : The maximum number of thread to use
		:param verbose : Specify if verbose when solving the optimization problem
		:param mu_lin : A penalization term for the linearized constraints
		:param mu_period : A penalization term for looping early than Kp
	"""
	# Create the Gurobi model
	mOpt = gp.Model('Linearized problem at each iteration')
	mOpt.Params.OutputFlag = verbose
	mOpt.Params.Presolve = 2
	mOpt.Params.NumericFocus = 3
	# mOpt.Params.MIPFocus = 1
	# mOpt.Params.Crossover = 0
	# mOpt.Params.CrossoverBasis = 0
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

	# Create the time-varying Markov matrices
	MarkovM = dict()
	for s_id in swarm_ids:
		nedges = lG.getSubswarmNonEdgeSet(s_id)
		for n_id in node_ids:
			for m_id in node_ids:
				for t in range(Kp):
					# Motion constraints been enforced
					if (m_id, n_id) in nedges:
						MarkovM[(s_id, n_id, m_id, t)] = \
							mOpt.addVar(lb=0, ub=0, name='M[{}][{}][{}]({})'.format(s_id, n_id, m_id, t))
					else:
						MarkovM[(s_id, n_id, m_id, t)] = \
							mOpt.addVar(lb=0, ub=1, name='M[{}][{}][{}]({})'.format(s_id, n_id, m_id, t))

	# Trivial constraints from the Markov matrix x(t+1) <= A^t x(t) and x(t) <= A^t x(t+1)
	for s_id in swarm_ids:
		nedges = lG.getSubswarmNonEdgeSet(s_id)
		for t in range(Kp):
			for n_id in node_ids:
				mOpt.addConstr(xDen[(s_id, n_id, t+1)] <= gp.quicksum([ 0 if (m_id, n_id) in nedges else xDen[(s_id,m_id,t)] for m_id in node_ids]))
				mOpt.addConstr(xDen[(s_id, n_id, t)] <= gp.quicksum([ 0 if (n_id, m_id) in nedges else xDen[(s_id,m_id,t+1)] for m_id in node_ids]))


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
	mOpt.setObjective(gp.quicksum([*[mu_lin*sVar for _, sVar in dictSlack.items()], costVal, *[mu_period*(ind[0]+1)*xDen[ind] for ind in lVars]]), gp.GRB.MINIMIZE)

	mOpt.update()

	return mOpt, xDen, MarkovM, lVars, dictConstr, dictConstrVar, dictConstrL1

def update_linearized_problem_scp(mOpt, xDen, MarkovM, dictConstr, trustRegion, dictConstrVar, dictConstrL1, xk, Mk, node_ids):
	""" Given the linearized bilinear constraints, update such a constraint with respect to the previous solution xk and Mk.
		:param mOpt : The gurobi instance of the linearized problem
		:param xDen : The density distribution envolution of the problem instance
		:param MarkovM : The Markov matrix of the problem instance
		:param dictConstr : A dictionary to stire the parameterized and linearized constraints
		:param trustRegion : The trust region value
		:param dictConstrVar : # Store the constraints on the new variable |x_i^s(t) - xk_i^s(t)| <= w_i^s(t) for all t and swarm s, and index i
		:param dictConstrL1 : Store the L1-norm constraint sum_i w_i^s(t) <= trust region for all swarm s and time t
		:param xk : Solution of the linearized problem at the last iteration
		:param Mk : Solution of the linearized problem at the last iteration
		:param node_ids : The se of node identifers
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
	""" Compute some metrics value used for the contratcion or expansion of trust region
		:param xk : past density solution iterate
		:param Mk : past Markov solution iterate
		:param xk1 : current density solution iterate
		:param Mk1 : current Markov solution iterate
		:param swam_ids : the set of swarm identifier
		:param Kp : the loop time horizon
		:param node_ids : the set of node identifiers
	"""
	# Compute the linearization cost attained by Xk1 and Mk1
	fklist = np.array([sum( [xk[(s_id,m_id,t)]*Mk1[(s_id,n_id,m_id,t)]+Mk[(s_id,n_id,m_id,t)]*(xk1[(s_id,m_id,t)]-xk[(s_id,m_id,t)]) for m_id in node_ids])\
							 for s_id in swam_ids for t in range(Kp) for n_id in node_ids])
	# Compute the actual relaized cost attained by Xk1 and Mk1
	flist = np.array([sum([xk1[(s_id,m_id,t)]*Mk1[(s_id,n_id,m_id,t)] for m_id in node_ids]) \
						for s_id in swam_ids for t in range(Kp) for n_id in node_ids])
	# Collect Xk1 and Xk as an array
	x1nextlist = np.array([xk1[(s_id,n_id,t+1)] for s_id in swam_ids for t in range(Kp) for n_id in node_ids])
	xnextlist = np.array([xk[(s_id,n_id,t+1)] for s_id in swam_ids for t in range(Kp) for n_id in node_ids])
	# Compute the actual realized cost by xk and Mk
	prodxlist = np.array([sum([xk[(s_id,m_id,t)]*Mk[(s_id,n_id,m_id,t)] for m_id in node_ids]) \
						for s_id in swam_ids for t in range(Kp) for n_id in node_ids])
	return flist, fklist, np.linalg.norm(x1nextlist-flist,1), np.linalg.norm(xnextlist-prodxlist,1) 

def gtlproco_scp(milp_expr, lG, Kp, init_den_lb=None, init_den_ub=None,
					cost_fun=None, maxIter=10, costTol=1e-6, bilTol=1e-7,
					mu_lin=1e1, mu_period=1, trust_lim= 1e-4, 
					timeLimit=5, n_thread=0, verbose=True, verbose_solver=True):
	""" Solve the probabilistic swarm control problem under GTL specifications
		:param milp_expr : A mixed integer representation of the GTL constraints
		:param lG : The labelled graph
		:param Kp : The loop time horizon
		:param init_den_lb : The lower bound on the density distribution at time 0
		:param init_den_ub : The upper bound on the density distribution at time 0
		:param cost_fun : The cost function to minimize
		:param maxIter : The maximum number of iteration of the SCP scheme
		:param costTol : The tolerance to decide the cost is optimal (variation between two iterations) 
		:param bilTol : Accuracy tolerance for the bilinear constraint
		:param mu_lin : A penalization term for the linearized constraints
		:param mu_period : A penalization term for looping early than Kp
		:param trust_lim : The trust region limit as lower bound under stopping the sequential scheme
		:param timeLimit : The time limit at each time step f the solver
		:param n_thread : The maximum number of thread to use
		:param verbose : Specify if verbose when solving the optimization problem
		:param verbose_solver : Specify if gurobi verbose
	"""
	
	# Get the node ids and swarm ids
	swarm_ids = lG.eSubswarm.keys()
	node_ids = lG.V.copy()

	# Initialize the SCP with a feasible solution to the GTL constraints
	dictXk, sTime1 = find_initial_feasible_density(milp_expr, lG, Kp, init_den_lb=init_den_lb, init_den_ub=init_den_ub, verbose=verbose_solver)

	# Initialize the SCP with the closes Markov Matrix
	dictMk, sTime2 = find_initial_feasible_Markov(lG, Kp, dictXk, verbose=verbose_solver)

	# Construct the linearized problem
	linProb, xLin, Mlin, lVars, linConstr, dictConstrVar, dictConstrL1 = \
		create_linearized_problem_scp(milp_expr, lG, Kp, cost_fun, init_den_lb=init_den_lb, init_den_ub=init_den_ub,
						timeLimit=timeLimit, n_thread=n_thread, verbose=verbose_solver, mu_lin=mu_lin, mu_period=mu_period)

	# Initialize the periodicy coefficient
	ljRes = {ind0 : dictXk[ind0] for ind0 in lVars}

	# Set the maximum trust region and intial trust region
	trust_max = 2.0
	trust_init = 1.0 # len(node_ids)

	# Initialize the trust region
	rk = trust_init

	# Solving parameters and status
	solve_time = sTime1 + sTime2
	status = -1
	optCost = get_cost_function(cost_fun, dictXk, dictMk, swarm_ids, Kp, node_ids) + sum([mu_period*(ind[0]+1)*dictXk[ind] for ind in lVars])
	attainedCost = optCost

	for i in range(maxIter):

		# Check if the time limit is elapsed
		if solve_time >= timeLimit:
			status = -1
			break

		# Update the optimization problem and solve it
		update_linearized_problem_scp(linProb, xLin, Mlin, linConstr, rk, dictConstrVar, dictConstrL1, dictXk, dictMk, node_ids)

		# Update the starting point of the linearized problem
		for kVal, xVal in xLin.items():
			xVal.start = dictXk[kVal]
		for kVal, Mval in Mlin.items():
			Mval.start = dictMk[kVal]

		# Measure optimization time
		cur_t = time.time()
		linProb.optimize()
		solve_time += time.time() - cur_t

		# Save the status of the problem
		if linProb.status == gp.GRB.OPTIMAL:
			status = 1
		elif linProb.status == gp.GRB.TIME_LIMIT:
			status = -1
			break
		else:
			status = 0

		# Collect the solution of the current iteration
		dictMk1 = {kVal : Mval.x for kVal, Mval in Mlin.items()}
		dictXk1 = {kVal : xVal.x for kVal, xVal in xLin.items()}

		# Compute the linearized and realized cost
		truef, linf, x1diffbil, xdiffbil = compute_true_linearized_fun(dictXk, dictMk, dictXk1, dictMk1, swarm_ids, Kp, node_ids)

		# Check if the bilinear constraints are satisified
		finish_bil = x1diffbil <= bilTol*len(node_ids)*Kp*len(swarm_ids) or xdiffbil <= bilTol*len(node_ids)*Kp*len(swarm_ids)

		diffLin = np.linalg.norm(truef-linf, 1)
		noImprov = (i >= 0) and (np.abs(linProb.objVal-attainedCost) < costTol) and finish_bil
		actualCost  = get_cost_function(cost_fun, dictXk1, dictMk1, swarm_ids, Kp, node_ids) + sum([mu_period*(ind[0]+1)*dictXk1[ind] for ind in lVars])
		attainedCost = linProb.objVal

		# The trust region expander and contractor coefficient are dependent on the accuracy of the bilinear constraint
		rho_bil_k  =  1 if (xdiffbil == 0 and x1diffbil ==0) else (np.inf if xdiffbil == 0 else x1diffbil / xdiffbil)

		# Check if the accuracy of the bilinear constraint of the new solution is better than the past solution
		if rho_bil_k > 1: # If the obtained accuracy is worst the reject the step
			rk  =  rk / (1e-1+rho_bil_k) if rho_bil_k < np.inf else 0
			if verbose:
				print('[Iteration {}] : Reject current solution -> Coarse Linearization'.format(i))
		else: # If the accuracy has improved accept the step and change the trust region
			dictXk = dictXk1
			dictMk = dictMk1
			optCost = actualCost
			ljRes = {ind0 : xLin[ind0].x for ind0 in lVars}
			rk = np.minimum(rk / (1e-3+rho_bil_k) if rho_bil_k < np.inf else 0, trust_max)
			if verbose:
				print('[Iteration {}] : Accept current solution'.format(i))

		# Some printing
		if verbose:
			print('[Iteration {}] : Rho bil = {}, bil diff xk = {}, bil diff xk1 = {}'.format(i, rho_bil_k, xdiffbil, x1diffbil))
			print('[Iteration {}] : Trust region = {}, actual Cost = {}, attained Cost = {}, Lin Error = {}, Solve Time = {}'.format(
					i, rk, actualCost, attainedCost, diffLin, solve_time))

		# If the solution does not improve
		if noImprov:
			if verbose:
				print('[Iteration {}] : No improvement in the solution'.format(i))
			break

		# If the trust region value is below the threshold
		if rk < trust_lim:
			if verbose:
				print('[Iteration {}] : Minimum trust region reached'.format(i))
			break
			
	return optCost, status, solve_time, dictXk, dictMk, ljRes, swarm_ids, node_ids


def create_den_constr(xDen, swam_ids, Kp, node_ids):
	""" Given symbolic variables (gurobi, pyomo, etc..) representing the densities
		distribution at all times and all nodes, this function returns the 
		constraint 1^T x^s(t) = 1 for all s and t
		:param xDen : The symbolic variable representing the density distribution of the swarm
		:param swam_ids : The set of swarm identifiers
		:param Kp : The loop time horizon
		:param node_ids : The set of node identifiers
	"""
	listConstr = list()
	for s_id in swam_ids:
		for t in range(Kp+1):
			listConstr.append(sum([ xDen[(s_id, n_id, t)] for n_id in node_ids]) == 1)
	return listConstr

def create_markov_constr(MarkovM, swam_ids, Kp, node_ids):
	""" Given symbolic variables (gurobi, pyomo, etc..) representing the Markov matrices
		at all times, this function returns the stochasticity constraint 1^T M^s(t) = 1 for all s and t
		:param MarkovM : Symbolic variable representing the density distribution of the swarm
		:param swam_ids : The set of swarm identifiers
		:param Kp : The loop time horizon
		:param node_ids : The set of node identifiers
	"""
	listConstr = list()
	# Stochastic natrix constraint
	for s_id in swam_ids:
		for t in range(Kp):
			for n_id in node_ids:
				listConstr.append(sum([MarkovM[(s_id,m_id,n_id,t)] for m_id in node_ids]) == 1)
	return listConstr

def create_bilinear_constr(xDen, MarkovM, swam_ids, Kp, node_ids):
	""" Return the bilinear constraint M(t) x(t) = x(t+1)
		:param xDen :  The symbolic variable representing the density distribution of the swarm
		:param MarkovM : Symbolic variable representing the density distribution of the swarm
		:param swam_ids : The set of swarm identifiers
		:param Kp : The loop time horizon
		:param node_ids : The set of node identifiers
	"""
	listConstr = list()
	for s_id in swam_ids:
		for t in range(Kp):
			for n_id in node_ids:
				listConstr.append(sum([ MarkovM[(s_id,n_id,m_id,t)]* xDen[(s_id, m_id, t)] for m_id in node_ids])\
									== xDen[(s_id, n_id, t+1)])
	return listConstr


def get_cost_function(cost_fun, xDen, MarkovM, swam_ids, Kp, node_ids):
	""" Compute the cost function given the density evolution and Markvo matrix evolution
		:param cost_fun : A python function taking as input xDict (emtire swarm density at a fixed time), 
							MDict (Markov matrices at a fixed time), swam_ids, node_ids
		:param xDen :  The symbolic variable representing the density distribution of the swarm
		:param MarkovM : Symbolic variable representing the density distribution of the swarm
		:param swam_ids : The set of swarm identifiers
		:param Kp : The loop time horizon
		:param node_ids : The set of node identifiers
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
							cost_fun=None, solve=False, mu_period=1):
	""" Generic function to create and solve the bilinear optimization problem.
		This function can be used to solve the MINLP using Gurobi, SCIP, Pyomo, Bonmin etc...
		:param util_funs : Solver specific function to create continuous/binary variables, create constraints,
							solve the optimization problem, etc...
		:param milp_expr : A mixed integer representation of the GTL constraints
		:param lG : The labelled graph
		:param Kp : The loop time horizon
		:param init_den_lb : The lower bound on the density distribution at time 0
		:param init_den_ub : The upper bound on the density distribution at time 0
		:param cost_fun : The cost function to minimize
		:param solve : Specify if the problem must be solved or not
		:param mu_period : A penalization term for looping early than Kp
	"""
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
				# Enforce the initial swarm distribution
				if t == 0 and init_den_lb is not None and init_den_ub is not None:
					xDen[(s_id, n_id, t)] = util_funs['r']('x[{}][{}]({})'.format(s_id, n_id, t),
												init_den_lb[s_id][n_id], init_den_ub[s_id][n_id])
				else:
					xDen[(s_id, n_id, t)] = util_funs['r']('x[{}][{}]({})'.format(s_id, n_id, t))

	# Create the different Markov matrices
	MarkovM = dict()
	for s_id in swarm_ids:
		nedges = lG.getSubswarmNonEdgeSet(s_id)
		for n_id in node_ids:
			for m_id in node_ids:
				for t in range(Kp):
					# Enforce the motion constraints by the adjacency matrix
					if (m_id, n_id) in nedges:
						MarkovM[(s_id, n_id, m_id, t)] = \
							util_funs['r']('M[{}][{}][{}]({})'.format(s_id, n_id, m_id, t),0,0)
					else:
						MarkovM[(s_id, n_id, m_id, t)] = \
							util_funs['r']('M[{}][{}][{}]({})'.format(s_id, n_id, m_id, t))
	
	# Create the boolean variables from the GTL formula -> add them to the xDen dictionary
	for ind in bVars:
		xDen[ind] = util_funs['b']('b[{}]'.format(ind[0]))

	# Create the tempory real variables from the GTL formula -> add them to the xDen dictionary
	for ind in rVars:
		xDen[ind] = util_funs['r']('r[{}]'.format(ind[0]))

	# Create the binary variables from the loop constraint
	for ind in lVars:
		xDen[ind] = util_funs['b']('l[{}]'.format(ind[0]))

	# Add the constraint on the density distribution
	listContr = create_den_constr(xDen,swarm_ids, Kp, node_ids)

	# Add the constraints on the Markov matrices
	listContr.extend(create_markov_constr(MarkovM, swarm_ids, Kp, node_ids))

	# Add the bilinear constraints
	listContr.extend(create_bilinear_constr(xDen, MarkovM, swarm_ids, Kp, node_ids))

	# Add the mixed-integer constraints from the GTL specifications
	listContr.extend(create_gtl_constr(xDen, milp_expr))

	# Add all the constraints yo the problem
	for constr in listContr:
		util_funs['constr'](constr)

	# Compute the cost function if given by the user
	costVal = get_cost_function(cost_fun, xDen, MarkovM, swarm_ids, Kp, node_ids) + sum([mu_period*(ind[0]+1)*xDen[ind] for ind in lVars])

	# Set the cost function in the optimization problem
	util_funs['cost'](costVal, xDen, MarkovM, swarm_ids, node_ids)

	# Solve the problem
	optCost, status, solveTime = -np.inf, -1, 0
	if solve:
		optCost, status, solveTime = util_funs['solve']()

	# Collect the solution of the problem
	xDictRes = dict()
	MDictRes = dict()
	ljRes = dict()
	if solve and status == 1:
		for s_id in swarm_ids:
			for n_id in node_ids:
				for t in range(Kp+1):
					xDictRes[(s_id, n_id, t)] = \
						util_funs['opt'](xDen[(s_id, n_id, t)])
		
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
						cost_fun=None, solve=True, timeLimit=5, n_thread=0, verbose=False, mu_period=1):
	""" Solve the probabilistic swarm control under GTL constraints using the nonconvex solver GUROBI.
		:param milp_expr : A mixed integer representation of the GTL constraints
		:param lG : The labelled graph
		:param Kp : The loop time horizon
		:param init_den_lb : The lower bound on the density distribution at time 0
		:param init_den_ub : The upper bound on the density distribution at time 0
		:param cost_fun : The cost function to minimize
		:param solve : Specify if the problem must be solve or not
		:param timeLimit : Specify the time limit of the solver
		:param n_thread : SPecify the maximum number of thread to be used by the solver
		:param verbose : Output specificities when solving
		:param mu_period : A penalization term for looping early than Kp
	"""
	# Create the Gurobi model
	mOpt = gp.Model('Bilinear MINLP formulation through GUROBI')
	mOpt.Params.OutputFlag = verbose
	mOpt.Params.NonConvex = 2
	mOpt.Params.Threads = n_thread
	mOpt.Params.TimeLimit = timeLimit

	# Define the function to set the cost function of the optimization problem
	def set_cost(cVal, xDen, mD, sids, nids):
		mOpt.setObjective(cVal, gp.GRB.MINIMIZE)

	# Define a function to solve the problem and return the correct status
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

	# Function to return the value of an optimization variable
	def ret_sol(solV):
		return solV.x

	# Save all these function to be used when creating the problem and solving it
	util_funs = dict()
	util_funs['r'] = lambda name, lb=0, ub=1 : mOpt.addVar(lb=lb, ub=ub, name=name)
	util_funs['b'] = lambda name : mOpt.addVar(vtype=gp.GRB.BINARY, name=name)
	util_funs['constr'] = lambda constr : mOpt.addConstr(constr)
	util_funs['solve'] = solve_f
	util_funs['cost'] = set_cost
	util_funs['opt'] = ret_sol

	return create_minlp_model(util_funs, milp_expr, lG, Kp, init_den_lb, init_den_ub, 
								cost_fun=cost_fun, solve=solve, mu_period=mu_period)


def create_minlp_scip(milp_expr, lG, Kp, init_den_lb=None, init_den_ub=None,
						cost_fun=None, solve=True,
						timeLimit=5, n_thread=0, verbose=False, mu_period=1):
	""" Solve the probabilistic swarm control under GTL constraints using the nonconvex solver SCIP
		:param milp_expr : A mixed integer representation of the GTL constraints
		:param lG : The labelled graph
		:param Kp : The loop time horizon
		:param init_den_lb : The lower bound on the density distribution at time 0
		:param init_den_ub : The upper bound on the density distribution at time 0
		:param cost_fun : The cost function to minimize
		:param solve : Specify if the problem must be solve or not
		:param timeLimit : Specify the time limit of the solver
		:param n_thread : SPecify the maximum number of thread to be used by the solver
		:param verbose : Output specificities when solving
		:param mu_period : A penalization term for looping early than Kp
	"""
	# Import and set up the solver
	import pyscipopt as pSCIP
	# Create the optimization problem instance
	mOpt = pSCIP.Model('Bilinear MINLP formulation through SCIP')
	mOpt.hideOutput(quiet = (not verbose))
	mOpt.setParam('limits/time', timeLimit)

	# Define the cost function
	def set_cost(cVal, xDen, mD, sids, nids):
		mOpt.setObjective(cVal, "minimize")

	# Define the function to solve the problem
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

	# Define the function to return the value of each variable
	def ret_sol(solV):
		return mOpt.getVal(solV)

	# Set up the parameters
	util_funs = dict()
	util_funs['r'] = lambda name, lb=0, ub=1 : mOpt.addVar(lb=lb, ub=ub, name=name)
	util_funs['b'] = lambda name : mOpt.addVar(vtype="B", name=name)
	util_funs['constr'] = lambda constr : mOpt.addCons(constr)
	util_funs['solve'] = solve_f
	util_funs['cost'] = set_cost
	util_funs['opt'] = ret_sol

	return create_minlp_model(util_funs, milp_expr, lG, Kp, init_den_lb, init_den_ub,
								cost_fun=cost_fun, solve=solve, mu_period=1)

def create_minlp_pyomo(milp_expr, lG, Kp, init_den_lb=None, init_den_ub=None, 
			cost_fun=None, solve=True, 
			solver='couenne', solverPath='/home/fdjeumou/Documents/non_convex_solver/',
			timeLimit=5, n_thread=0, verbose=False, mu_period=1):
	""" Solve the probabilistic swarm control under GTL constraints using the python interface pyomo to bonmin, couenne, ipopt
		:param milp_expr : A mixed integer representation of the GTL constraints
		:param lG : The labelled graph
		:param Kp : The loop time horizon
		:param init_den_lb : The lower bound on the density distribution at time 0
		:param init_den_ub : The upper bound on the density distribution at time 0
		:param cost_fun : The cost function to minimize
		:param solve : Specify if the problem must be solve or not
		:param solver : Specify the solver to use 
		:param solverPath : Specify the path of the solver to use
		:param timeLimit : Specify the time limit of the solver
		:param n_thread : SPecify the maximum number of thread to be used by the solver
		:param verbose : Output specificities when solving
		:param mu_period : A penalization term for looping early than Kp
	"""
	from pyomo.environ import Var, ConcreteModel, Constraint, NonNegativeReals, Binary, SolverFactory
	from pyomo.environ import Objective, minimize
	import pyutilib
	from pyomo.opt import SolverStatus, TerminationCondition

	mOpt = ConcreteModel('Bilinear MINLP formulation through PYOMO')

	# Function to return real values variable
	def r_values(name, lb=0, ub=1):
		setattr(mOpt, name, Var(bounds=(lb,ub), within=NonNegativeReals))
		return getattr(mOpt, name)

	# Function to return binary variables
	def b_values(name):
		setattr(mOpt, name, Var(within=Binary))
		return getattr(mOpt, name)

	# Function to add constraints
	nbConstr = 0
	def constr(val):
		nonlocal nbConstr
		setattr(mOpt, 'C_{}'.format(nbConstr), Constraint(expr = val))
		nbConstr+=1

	# FUnction to set the cost to optimize
	def set_cost(cVal, xDen, mD, sids, nids):
		setattr(mOpt, 'objective', Objective(expr=cVal, sense=minimize))

	# Function to solve the optimization problem
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

	# Function to return the value of an optimization variable
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
							cost_fun=cost_fun, solve=solve, mu_period=1)

def create_reach_avoid_problem_lp(gtlFormulas, nodes, desDens, lG, cost_fun=None, cost_M = None, 
					solve=True, timeLimit=5, n_thread=0, verbose=False):
	""" Compute a solution of the reach-avoid LP problem using GUROBI. A solution of such a problem
		provides a Markov Matrix that ensures the satisfcation of reach-avoid specifications
		:param gtlFormulas : The safe-avoid GTL formulas to satisfy
		:param nodes : The set of nodes corresponding to each GTL formula to satisfy
		:param desDens : The desired density to reach (reach specifications)
		:param lG : The labelled graph
		:param cost_fun : The cost function to optimize
		:param cost_M : A dictionary given the weight for each ergocity coefficient for each swarm
		:param solve : Specify if the problem has to be solved
		:param timeLimit : Specify the time limit of the solver
		:param n_thread : SPecify the maximum number of thread to be used by the solver
		:param verbose : Output specificities when solving
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
	# Add the second dual constraint
	for i in range(b.shape[0]):
		for j in range(A.shape[1]):
			t1 = sum( dictY[(i,k)]*A[k,j] for k in range(b.shape[0]))
			t2 = sum( dictS[(i,k)]*dictO.get((k,j), 0) for k in range(len(swarm_ids)) )
			t3 = sum( A[i,k]*diagMDict.get((k,j), 0) for k in range(A.shape[1]))
			mOpt.addConstr(t1 + t2 <= - t3)

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

	# Optimize if it is required
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
	""" Compute a solution of the LP problem when there's scrambling patter and the more general SDP problem when there's no scrambling pattern. 
		A solution of such a problem provides a Markov Matrix that ensures the satisfcation of reach-avoid specifications
		:param gtlFormulas : The safe-avoid GTL formulas to satisfy
		:param nodes : The set of nodes corresponding to each GTL formula to satisfy
		:param desDens : The desired density to reach (reach specifications)
		:param lG : The labelled graph
		:param cost_fun : The cost function to optimize
		:param cost_M : A dictionary given the weight for each ergocity coefficient for each swarm
		:param solve : Specify if the problem has to be solved
		:param timeLimit : Specify the time limit of the solver
		:param n_thread : SPecify the maximum number of thread to be used by the solver
		:param verbose : Output specificities when solving
		:param sdp_solver : Use the SDP solver instead of relying on the ergocity coefficient
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
							timeLimit=5, n_thread=0, verbose=True, mu_period=1)
	print('-------------------------------------------')
	print(ljRes)
	print(optCost, status, solveTime)
	# Compute and print the accuracy of the bilinear constraint
	maxDiff = 0
	for s_id in swarm_ids:
		for n_id in node_ids:
			for t in range(Kp):
				s = sum(MDictRes[(s_id, n_id, m_id, t)]*xDictRes[(s_id, m_id,t)] for m_id in node_ids)
				# print(np.abs(xDictRes[(s_id, n_id,t+1)]-s))
				maxDiff = np.maximum(maxDiff, np.abs(xDictRes[(s_id, n_id,t+1)]-s))
	print('Accuracy bilinear constraint : ', maxDiff)
	print('-------------------------------------------')

	# create_minlp_scip(m_milp, lG, Kp)
	optCost, status, solveTime, xDictRes, MDictRes, ljRes, swarm_ids, node_ids = \
									create_minlp_scip(m_milp, lG, Kp, cost_fun=cost_fun,
										timeLimit=5, n_thread=0, verbose=True, mu_period=1)
	print('-------------------------------------------')
	print(ljRes)
	print(optCost, status, solveTime)
	# Compute and print the accuracy of the bilinear constraint
	maxDiff = 0
	for s_id in swarm_ids:
		for n_id in node_ids:
			for t in range(Kp):
				s = sum(MDictRes[(s_id, n_id, m_id, t)]*xDictRes[(s_id, m_id,t)] for m_id in node_ids)
				# print(np.abs(xDictRes[(s_id, n_id,t+1)]-s))
				maxDiff = np.maximum(maxDiff, np.abs(xDictRes[(s_id, n_id,t+1)]-s))
	print('Accuracy bilinear constraint : ', maxDiff)
	print('-------------------------------------------')

	optCost, status, solveTime, xDictRes, MDictRes, ljRes, swarm_ids, node_ids = gtlproco_scp(m_milp, lG, Kp, 
		cost_fun=cost_fun, maxIter=100, verbose=True, verbose_solver=False, costTol=1e-4, bilTol=1e-6, mu_lin=1e1, mu_period=1)
	print('-------------------------------------------')
	print(ljRes)
	print(optCost, status, solveTime)
	# Compute and print the accuracy of the bilinear constraint
	maxDiff = 0
	for s_id in swarm_ids:
		for n_id in node_ids:
			for t in range(Kp):
				s = sum(MDictRes[(s_id, n_id, m_id, t)]*xDictRes[(s_id, m_id,t)] for m_id in node_ids)
				# print(np.abs(xDictRes[(s_id, n_id,t+1)]-s))
				maxDiff = np.maximum(maxDiff, np.abs(xDictRes[(s_id, n_id,t+1)]-s))
	print('Accuracy bilinear constraint : ', maxDiff)
	print('-------------------------------------------')

	# create_minlp_pyomo(m_milp, lG, Kp)
	optCost, status, solveTime, xDictRes, MDictRes, ljRes, swarm_ids, node_ids = \
									create_minlp_pyomo(m_milp, lG, Kp,cost_fun=cost_fun,
									timeLimit=5, n_thread=0, verbose=True, mu_period=1)
	print('-------------------------------------------')
	print(ljRes)
	print(optCost, status, solveTime)
	# Compute and print the accuracy of the bilinear constraint
	maxDiff = 0
	for s_id in swarm_ids:
		for n_id in node_ids:
			for t in range(Kp):
				s = sum(MDictRes[(s_id, n_id, m_id, t)]*xDictRes[(s_id, m_id,t)] for m_id in node_ids)
				# print(np.abs(xDictRes[(s_id, n_id,t+1)]-s))
				maxDiff = np.maximum(maxDiff, np.abs(xDictRes[(s_id, n_id,t+1)]-s))
	print('Accuracy bilinear constraint : ', maxDiff)
	print('-------------------------------------------')
