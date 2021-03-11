from .LabelledGraph import LabelledGraph
from .GTLformula import *

# Import the optimizer tools
import gurobipy as gp

def create_gurobi_minlp(milp_expr, ):

	# Create the Gurobi model
	mOpt = gp.Model('Bilinear MINLP formulation through GUROBI')

	# Create the different density variables
	xDen = dict()
	for s_id in swarm_ids:
		for n_id in node_ids:
			for t in range(Kp+1):
				xDen[(s_id, n_id, t)] = \
					mOpt.addVar(lb=0, ub=1, name='x[{}][{}]({})'.format(s_id, n_id, t))
	
	# Create the different Markov matrices
	MarkovM = dict()
	for s_id in swarm_ids:
		for n_id in node_ids:
			for m_id in node_ids:
				for t in range(Kp+1):
					MarkovM[(s_id, n_id, m_id, t)] = \
						mOpt.addVar(lb=0, ub=1, name='M[{}][{}][{}]({})'.format(s_id, n_id, m_id, t))
	
	# Create the boolean variables from the GTL formula -> add them to the xDen dictionary
	for (ind0, ind1, tInd) in bVars:
		xDen[(ind0, ind1, tInd)] = mOpt.addVar(vtype=gp.GRB.BINARY, name='b[{}][{}]({})'.format(ind0, ind1, tInd))

	# Create the tempory real variables from the GTL formula -> add them to the xDen dictionary
	for (ind0, ind1, tInd) in rVars:
		xDen[(ind0, ind1, tInd)] = mOpt.addVar(lb=0, ub=1.0, name='r[{}][{}]({})'.format(ind0, ind1, tInd))

	# Create the binary variables from the loop constraint
	for (ind0, ind1) in lVars:
		xDen[(ind0, ind1)] = mOpt.addVar(vtype=gp.GRB.BINARY, name='l[{}]'.format(ind0))

	# Add the constraint on the density distribution
	listContr = create_den_constr(xDen)

	# Add the constraints on the Markov matrices
	listContr.extend(create_markov_constr(MarkovM))

	# Add the bilinear constraints
	listContr.extend(create_bilinear_constr(xDen, MarkovM))

	# Add the mixed-integer constraints
	listContr.extend(create_gtl_constr(xDen, milp_expr))

	# Add cost function