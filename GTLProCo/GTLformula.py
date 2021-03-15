import numpy as np
from .LabelledGraph import LabelledGraph
from abc import ABC, abstractmethod
import itertools

STRICT_INEQ = 1e-8
BIG_M = 50
N_VARS = 0
BOOL_VAR = -1

def create_gtl_constr(dictVar, milp_expr):
	""" Return the constraints encoded by this milp_expression
		This function is usually called by the optimization solver
	"""
	(newCoeffs, newVars, rhsVals, nVar), (lCoeffs, lVars, lRHS) = milp_expr
	# GTL formula constraints
	resContr = list()
	for nCoeffs, vs, rhsV in zip(newCoeffs, newVars, rhsVals):
		# (v[0], v[1], ts) if v[1] > BOOL_VAR else (v[0] if v[1] == BOOL_VAR-2 else (v[0], ts))
		contrRepr = [c * dictVar[v] for c, v in zip(nCoeffs, vs)]
		resContr.append(sum(contrRepr) <= rhsV)
	# Loop constraints
	for nCoeffs, vs, rhsV in zip(lCoeffs, lVars, lRHS):
		contrRepr = [c*dictVar[v] for c, v in zip(nCoeffs, vs)]
		resContr.append(sum(contrRepr) <= rhsV)
	return resContr


def create_milp_constraints(listFormula, listNode, Kp, lG, initTime=0):
	"""
		Create the MILP constraints associated to the list of formula
		and the corresponding nodes.
		The MILP should be an equivalent formulation for satisfaction
		of each formula at each node at time 0
	"""
	assert isinstance(listFormula, list) and isinstance(listNode, list), "Argument should be a list"
	assert len(listFormula) == len(listNode), "Different list length between formulas and nodes"
	
	newCoeffs = list()
	newVars = list()
	rhsVals = list()

	varsAnd = list()
	for gtl, node in zip(listFormula, listNode):
		nCoeffs, nVars, rhsVs, fEval = gtl.milp_repr(lG, node, initTime, Kp)
		newCoeffs.extend(nCoeffs)
		newVars.extend(nVars)
		rhsVals.extend(rhsVs)
		varsAnd.append(fEval)

	# Add the last constraint AND constraint between each formula
	if len(listFormula) > 1:
		nCoeffs, nVars, rVals, nVar = and_op(varsAnd)
		newCoeffs.extend(nCoeffs)
		newVars.extend(nVars)
		rhsVals.extend(rVals)
	else:
		nVar = varsAnd[0]

	# The formula must be True nVar == 1
	newCoeffs.append([1])
	newVars.append([nVar])
	rhsVals.append(1)

	newCoeffs.append([-1])
	newVars.append([nVar])
	rhsVals.append(-1)

	# Add the constraint due to the loop
	lCoeffs = list()
	lVars = list()
	lRHS = list()

	lCoeffs.append([1 for i in range(1, Kp+1)])
	lVars.append([ (i, BOOL_VAR-2) for i in range(1,Kp+1)])
	lRHS.append(1)
	lCoeffs.append([-1 for i in range(1, Kp+1)])
	lVars.append([ (i, BOOL_VAR-2) for i in range(1,Kp+1)])
	lRHS.append(-1)

	for node in listNode:
		(coeffs, var) = lG.getLabelMatRepr(node)
		for j in range(1, Kp+1):
			for cV, indexV in zip(coeffs, var):
				t1 = [c for c in cV]
				t2 = [-c for c in cV]
				v1 = [(v[0],v[1], Kp) for v in indexV]
				v2 = [(v[0],v[1], j-1) for v in indexV]
				lCoeffs.append(t1 + t2 + [BIG_M])
				lVars.append(v1 + v2 + [(j,BOOL_VAR-2)])
				lRHS.append(BIG_M)
				lCoeffs.append(t2 + t1 + [BIG_M])
				lVars.append(v1 + v2 + [(j,BOOL_VAR-2)])
				lRHS.append(BIG_M)
	return (newCoeffs, newVars, rhsVals, nVar), (lCoeffs, lVars, lRHS)

def getVars(newVars, lVars=list()):
	"""
	Return the boolean and continuous variables provided by the inputs
	"""
	varBool = set()
	varReal = set()
	varL = set()
	densDistr = set()
	for vs in newVars:
		for v in vs:
			if v[1] == BOOL_VAR:
				varBool.add(v)
			elif v[1] == BOOL_VAR-1:
				varReal.add(v)
			elif v[1] == BOOL_VAR-2:
				varL.add(v)
			else:
				densDistr.add(v)
	for vs in lVars:
		for ind in vs:
			if ind[1] == BOOL_VAR-2:
				varL.add(ind)
			else:
				densDistr.add(ind)
	return varBool, varReal, varL, densDistr


def print_constr(newCoeffs, newVars, rhsVals, nVar,
							lCoeffs=list(), lVars=list(), lRHS=list()):
	"""
		Provide a string representation of the MILP constraints
		encoded by the given aruguments
	"""
	# Sanity check
	assert len(newVars) == len(newCoeffs) and len(newVars) == len(rhsVals)
	assert len(lCoeffs) == len(lVars) and len(lVars) == len(lRHS)

	# Store the set of all variables of this MILP representation
	varBool, varReal, varL, densDistr = getVars(newVars, lVars)
	dictVar = dict()
	for ind in varBool:
		dictVar[ind] = 'b_{}'.format(ind[0])
	for ind in varReal:
		dictVar[ind] = 'r_{}'.format(ind[0])
	for ind in varL:
		dictVar[ind] = 'l_{}'.format(ind[0])
	for (ind1, ind2, ts) in densDistr:
		dictVar[(ind1, ind2, ts)] = 'x_{}_{}({})'.format(ind1, ind2, ts)

	# GTL formula constraints
	resContr = list()
	for nCoeffs, vs, rhsV in zip(newCoeffs, newVars, rhsVals):
		contrRepr = ['{}*{}'.format(c, dictVar[v]) for c, v in zip(nCoeffs, vs)]
		sumRes = '{} <= {}'.format(' + '.join(contrRepr), rhsV)
		resContr.append(sumRes)
	# Loop constraints
	lConstr = list()
	for nCoeffs, vs, rhsV in zip(lCoeffs, lVars, lRHS):
		contrRepr = ['{}*{}'.format(c, dictVar[v]) for c, v in zip(nCoeffs, vs)]
		sumRes = '{} <= {}'.format(' + '.join(contrRepr), rhsV)
		lConstr.append(sumRes)

	# Start the printing formula
	print('-------------------------')
	print('Result variables: {}'.format(dictVar[nVar]))
	print('-------------------------')
	print('Boolean variables in {0,1}')
	print(', '.join([dictVar[v] for v in varBool]))
	print('-------------------------')
	print('Real variables in [0,1]')
	print(', '.join([dictVar[v] for v in varReal]))
	print('-------------------------')
	print('Density variables in [0,1]')
	print(', '.join([dictVar[v] for v in densDistr]))
	print('-------------------------')
	print('Loop variables in {0,1}')
	print(', '.join([dictVar[v] for v in varL]))
	print('-------------------------')
	print('Constraints GTL formula')
	print('\n'.join(resContr))
	print('-------------------------')
	print('Constraints Loop')
	print('\n'.join(lConstr))

def print_milp_repr(listFormula, listNode, Kp, lG, initTime=0):
	"""
		Return a string of the MILP representation of this GTL formula
	"""
	(newCoeffs, newVars, rhsVals, nVar), (lCoeffs, lVars, lRHS) = \
					create_milp_constraints(listFormula, listNode, Kp, lG, initTime)
	# Start the printing formula
	# print('--------------------------------------------------')
	print(lG)
	print('------ GTL Formula -------')
	reprNode = ', '.join(['{}'.format(n) for n in listNode])
	reprGTL = ', '.join(['{}'.format(gtl) for gtl in listFormula])
	print ('At nodes {} and time index {} -- Length trajectory {}'.format(reprNode, initTime, Kp))
	print(reprGTL)
	print_constr(newCoeffs, newVars, rhsVals, nVar, lCoeffs, lVars, lRHS)

def and_op(bVars):
	""" 
		Perform and AND operation between multiple variables
	"""
	global BOOL_VAR, N_VARS

	# Create the variable storing the result of the feasibility of this formula
	nVar = (N_VARS, BOOL_VAR-1) # -1 means it is a boolean var and -2 means it is not
	N_VARS += 1	# Increment the number of new variables

	# Get the number of propositions in the AND operation
	nProp = len(bVars)

	# Save the new constrainrs
	newCoeffs = list()
	newVars = list()
	rhsVals = list()

	# Get the MILP representation using the array repr of the linear label
	tContr = list()
	tCoeff =  list()
	for varT in bVars:
		newCoeffs.append([1, -1])
		newVars.append([nVar, varT])
		rhsVals.append(0)
		# Add the binding constraints of the result of each prop
		tContr.append(varT)
		tCoeff.append(1)
	tContr.append(nVar)
	tCoeff.append(-1)
	# Add the last constraint
	newCoeffs.append(tCoeff)
	newVars.append(tContr)
	rhsVals.append(nProp-1)
	return newCoeffs, newVars, rhsVals, nVar

def or_op(bVars):
	""" 
		Perform and OR operation between multiple variables
	"""
	global BOOL_VAR, N_VARS

	# Create the variable storing the result of the feasibility of this formula
	nVar = (N_VARS, BOOL_VAR-1) # -1 means it is a boolean var and -2 means it is not
	N_VARS += 1	# Increment the number of new variables

	# Get the number of propositions in the AND operation
	nProp = len(bVars)

	# Save the new constrainrs
	newCoeffs = list()
	newVars = list()
	rhsVals = list()

	# Get the MILP representation using the array repr of the linear label
	tContr = list()
	tCoeff =  list()
	for varT in bVars:
		newCoeffs.append([-1, 1])
		newVars.append([nVar, varT])
		rhsVals.append(0)
		# Add the binding constraints of the result of each prop
		tContr.append(varT)
		tCoeff.append(-1)
	tContr.append(nVar)
	tCoeff.append(1)
	# Add the last constraint
	newCoeffs.append(tCoeff)
	newVars.append(tContr)
	rhsVals.append(0)
	return newCoeffs, newVars, rhsVals, nVar

def eval_formula(self, graphTraj, formula, lG, node):
	"""
		Eval the formula at the node given by node
		:param lG : labelled graph
		:param node : the node at which to evaluate the GTL formula
		:param graphTraj : a graph trajectory --> Value of all densities
		:param formula : A GTL formula
	"""
	pass
		

def get_neighbors(lG, node, k):
	"""
		Get the set of subset of nodes at distance k from current node
		:param lG : labelled graph
		:param node : the node at which to evaluate the GTL formula
		:param k : Distance to the node in term of edges
	"""
	Edge = lG.getGlobalEdgeSet()
	if k == 1:
		resNode = set()
		for (s1, s2) in Edge:
			if s1 == node:
				resNode.add(s2)
			if s2 == node:
				resNode.add(s1)
		return resNode
	res = get_neighbors(lG, node, k-1)
	nRes = set()
	for s in res:
		for (s1, s2) in Edge:
			if s1 == s:
				nRes.append(s2)
			if s2 == s:
				nRes.append(s1)
	return nRes

class GTLFormula(ABC):
	"""
		Abstract base  class that represents an infinite horizon GTL
	"""
	@abstractmethod
	def milp_repr(self, lG, node, t, Kp):
		"""
			Get the Mixed-integer linear representation of this formula.
			This function will be used by the MILP solver to find the adequate
			Markov chain to probabilistically control the swarm
			:param lG : the labelled graph
			:param node : the node at which to evaluate the formula
			:param t : the time index at which the formula must be true
			:param Kp :  the graph trajectory length cycle
		"""
		pass


class AtomicGTL(GTLFormula):
	"""
		This GTL formula represents an atomic node proposition.
		Specifically, given the label f(x) at a node, where x is the density 
		distribution of all the subswarms. This specification enforces the 
		label at the node to satisfy f(x) <= c, where c parameterized the formula.
	"""
	def __init__(self, cVal):
		self.c = np.array(cVal)


	def __repr__(self):
		return "(x <= {})".format(self.c)

	def milp_repr(self, lG, node, t, Kp):
		"""
			Get the Mixed-integer linear representation of this formula.
			This function will be used by the MILP solver to find the adequate
			Markov chain to probabilistically control the swarm
			:param lG : the labelled graph
			:param node : the node at which to evaluate the formula
			:param t : the time index at which the formula must be true
			:param Kp :  the graph trajectory length cycle
		"""
		global N_VARS, BIG_M, STRICT_INEQ, BOOL_VAR

		# Sanity check if the constraint is the same size of the node label
		assert len(lG.getLabelMatRepr(node)[0]) ==  self.c.shape[0]

		# Create the variable storing the result of the feasibility of this formula
		nVar = (N_VARS, BOOL_VAR-1) # -1 means it is a boolean var and -2 means it is not
		N_VARS += 1	# Increment the number of new variables

		# Get the MILP representation using the array repr of the linear label
		(coeffs, var) = lG.getLabelMatRepr(node)

		# Save the new constrainrs
		newCoeffs = list()
		newVars = list()
		rhsVal = list()

		and_f = list()
		# Add the new constraints
		for cV, indexV, rhs in zip(coeffs, var, self.c):
			tList1, iList1 = list(), list()
			tList2, iList2 = list(), list()
			for cvalue, (sId, nId) in zip(cV, indexV):
				# Add the current coefficient and the variable that it multiplies
				tList1.append(cvalue)
				iList1.append((sId, nId, t))
				# Add the negation of the past constraint A x > b -> -Ax < -b
				tList2.append(-cvalue)
				iList2.append((sId, nId, t))
			nVarT = (N_VARS, BOOL_VAR) # -1 means it is a boolean var and -2 means it is not
			N_VARS += 1	# Increment the number of new variables
			# Add the term (1-P) phi
			tList1.append(BIG_M)
			iList1.append(nVarT)
			# Add the term -P phi
			tList2.append(-BIG_M)
			iList2.append(nVarT)
			# The right right side
			rhsVal.append(rhs + BIG_M)
			rhsVal.append(-rhs - STRICT_INEQ)
			# Add the new coefficients and variables
			newCoeffs.append(tList1)
			newCoeffs.append(tList2)
			newVars.append(iList1)
			newVars.append(iList2)
			and_f.append(nVarT)
		# Perform the and operation to ensure satisfaction of this atomic prop
		nCoeffs, nVars, rVals, nVar = and_op(and_f)
		# Add the last constraint
		newCoeffs.extend(nCoeffs)
		newVars.extend(nVars)
		rhsVal.extend(rVals)
		return newCoeffs, newVars, rhsVal, nVar

class AndGTL(GTLFormula):
	"""
		This GTL formula represents an AND operation between multiples GTL 
		formulas.
	"""
	def __init__(self, listFormula):
		self.mFormula = list()
		for gtl in listFormula:
			self.mFormula.append(gtl)

	def __repr__(self):
		return '({})'.format(' & '.join([v.__repr__() for v in self.mFormula]))

	def milp_repr(self, lG, node, t, Kp):
		"""
			Get the Mixed-integer linear representation of this formula.
			This function will be used by the MILP solver to find the adequate
			Markov chain to probabilistically control the swarm
			:param lG : the labelled graph
			:param node : the node at which to evaluate the formula
			:param t : the time index at which the formula must be true
			:param Kp :  the graph trajectory length cycle
		"""
		# Save the new constrainrs
		newCoeffs = list()
		newVars = list()
		rhsVals = list()

		# Get the MILP representation using the array repr of the linear label
		varsAnd = list()
		for gtl in self.mFormula:
			nCoeffs, nVars, rhsVs, fEval = gtl.milp_repr(lG, node, t, Kp)
			newCoeffs.extend(nCoeffs)
			newVars.extend(nVars)
			rhsVals.extend(rhsVs)
			varsAnd.append(fEval)
		nCoeffs, nVars, rVals, nVar = and_op(varsAnd)
		# Add the last constraint
		newCoeffs.extend(nCoeffs)
		newVars.extend(nVars)
		rhsVals.extend(rVals)
		return newCoeffs, newVars, rhsVals, nVar

class OrGTL(GTLFormula):
	"""
		This GTL formula represents an OR operation between multiples GTL 
		formulas.
	"""
	def __init__(self, listFormula):
		self.mFormula = list()
		for gtl in listFormula:
			self.mFormula.append(gtl)

	def __repr__(self):
		return '({})'.format(' | '.join([v.__repr__() for v in self.mFormula]))

	def milp_repr(self, lG, node, t, Kp):
		"""
			Get the Mixed-integer linear representation of this formula.
			This function will be used by the MILP solver to find the adequate
			Markov chain to probabilistically control the swarm
			:param lG : the labelled graph
			:param node : the node at which to evaluate the formula
			:param t : the time index at which the formula must be true
			:param Kp :  the graph trajectory length cycle
		"""
		# Save the new constrainrs
		newCoeffs = list()
		newVars = list()
		rhsVals = list()

		# Get the MILP representation using the array repr of the linear label
		varsAnd = list()
		for gtl in self.mFormula:
			nCoeffs, nVars, rhsVs, fEval = gtl.milp_repr(lG, node, t, Kp)
			newCoeffs.extend(nCoeffs)
			newVars.extend(nVars)
			rhsVals.extend(rhsVs)
			varsAnd.append(fEval)
		nCoeffs, nVars, rVals, nVar = or_op(varsAnd)
		# Add the last constraint
		newCoeffs.extend(nCoeffs)
		newVars.extend(nVars)
		rhsVals.extend(rVals)
		return newCoeffs, newVars, rhsVals, nVar

class NotGTL(GTLFormula):
	"""
		This GTL formula represents the negation of a proposition
	"""
	def __init__(self, rFormula):
		self.rFormula = rFormula

	def __repr__(self):
		return '!({})'.format(self.rFormula)

	def milp_repr(self, lG, node, t, Kp):
		"""
			Get the Mixed-integer linear representation of this formula.
			This function will be used by the MILP solver to find the adequate
			Markov chain to probabilistically control the swarm
			:param lG : the labelled graph
			:param node : the node at which to evaluate the formula
			:param t : the time index at which the formula must be true
			:param Kp :  the graph trajectory length cycle
		"""
		global N_VARS, BIG_M, STRICT_INEQ, BOOL_VAR
		# Create the variable storing the result of the feasibility of this formula
		nVar = (N_VARS, BOOL_VAR-1) # -1 means it is a boolean var and -2 means it is not
		N_VARS += 1	# Increment the number of new variables

		# Get the MILP representation of the formula after not operator
		nCoeffs, nVars, rhsVs, fEval = self.rFormula.milp_repr(lG, node, t, Kp)

		# Add the equality constraint by the NOT operator
		nCoeffs.append([1,1])
		nVars.append([nVar, fEval])
		rhsVs.append(1)

		nCoeffs.append([-1,-1])
		nVars.append([nVar, fEval])
		rhsVs.append(-1)
		return nCoeffs, nVars, rhsVs, nVar

class NextGTL(GTLFormula):
	"""
		This GTL formula represents the Next operator in the GTL language
	"""
	def __init__(self, rFormula):
		self.rFormula = rFormula

	def __repr__(self):
		return 'X({})'.format(self.rFormula)

	def milp_repr(self, lG, node, t, Kp):
		"""
			Get the Mixed-integer linear representation of this formula.
			This function will be used by the MILP solver to find the adequate
			Markov chain to probabilistically control the swarm
			:param lG : the labelled graph
			:param node : the node at which to evaluate the formula
			:param t : the time index at which the formula must be true
			:param Kp :  the graph trajectory length cycle
		"""
		if t < Kp:
			# Get the MILP representation of the formula after not operator
			return self.rFormula.milp_repr(lG, node, t+1, Kp)

		# In case t == Kp, the next time step is implied by the position of the loop
		newCoeffs = list()
		newVars = list()
		rhsVals = list()

		varsAnd = list()
		for i in range(1, Kp+1):
			nCoeffs, nVars, rhsVs, fEval = self.rFormula.milp_repr(lG, node, i, Kp)
			newCoeffs.extend(nCoeffs)
			newVars.extend(nVars)
			rhsVals.extend(rhsVs)

			ljVar = (i, BOOL_VAR-2)
			nCs, nVs, rs, nV = and_op([fEval, ljVar])
			newCoeffs.extend(nCs)
			newVars.extend(nVs)
			rhsVals.extend(rs)
			varsAnd.append(nV)

		nCoeffs, nVars, rVals, nVar = or_op(varsAnd)

		# Add the last constraint
		newCoeffs.extend(nCoeffs)
		newVars.extend(nVars)
		rhsVals.extend(rVals)
		return newCoeffs, newVars, rhsVals, nVar

class AlwaysGTL(GTLFormula):
	def __init__(self, rFormula):
		self.rFormula = rFormula

	def __repr__(self):
		return 'G({})'.format(self.rFormula)

	def milp_repr(self, lG, node, t, Kp):
		"""
			Get the Mixed-integer linear representation of this formula.
			This function will be used by the MILP solver to find the adequate
			Markov chain to probabilistically control the swarm
			:param lG : the labelled graph
			:param node : the node at which to evaluate the formula
			:param t : the time index at which the formula must be true
			:param Kp :  the graph trajectory length cycle
		"""
		if t == Kp:
			# Get the MILP representation of the formula after Always operator
			return self.rFormula.milp_repr(lG, node, t, Kp)

		nC1, nV1, rhsVs1, fEval1 = self.rFormula.milp_repr(lG, node, t, Kp)
		nC2, nV2, rhsVs2, fEval2 = self.milp_repr(lG, node, t+1, Kp)
		nC1.extend(nC2)
		nV1.extend(nV2)
		rhsVs1.extend(rhsVs2)
		nCoeffs, nVars, rVals, nVar = and_op([fEval1, fEval2])
		nC1.extend(nCoeffs)
		nV1.extend(nVars)
		rhsVs1.extend(rVals)
		return nC1, nV1, rhsVs1, nVar

class EventuallyAlwaysGTL(GTLFormula):
	"""
		This formula represents the persistence property Eventually always
	"""
	def __init__(self, rFormula):
		self.rFormula = rFormula

	def __repr__(self):
		return 'FG({})'.format(self.rFormula)

	def milp_repr(self, lG, node, t, Kp):
		"""
			Get the Mixed-integer linear representation of this formula.
			This function will be used by the MILP solver to find the adequate
			Markov chain to probabilistically control the swarm
			:param lG : the labelled graph
			:param node : the node at which to evaluate the formula
			:param t : the time index at which the formula must be true
			:param Kp :  the graph trajectory length cycle
		"""
		newCoeffs = list()
		newVars = list()
		rhsVals = list()

		orList = list()
		for i in range(1,Kp+1):
			lVar = [(i, BOOL_VAR-2)]
			for j in range(i,Kp+1):
				nCoeffs, nVars, rhsVs, fEval = \
					self.rFormula.milp_repr(lG, node, j, Kp)
				newCoeffs.extend(nCoeffs)
				newVars.extend(nVars)
				rhsVals.extend(rhsVs)
				lVar.append(fEval)
			nC1, nV1, rhsV1, nVar1 = and_op(lVar)
			newCoeffs.extend(nC1)
			newVars.extend(nV1)
			rhsVals.extend(rhsV1)
			orList.append(nVar1)
		nC1, nV1, rhsV1, nVar1 = or_op(orList)
		newCoeffs.extend(nC1)
		newVars.extend(nV1)
		rhsVals.extend(rhsV1)
		return newCoeffs, newVars, rhsVals, nVar1

class AlwaysEventuallyGTL(GTLFormula):
	"""
		This formula represents the liveness property Always eventually.
	"""
	def __init__(self, rFormula):
		self.rFormula = rFormula

	def __repr__(self):
		return 'GF({})'.format(self.rFormula)

	def milp_repr(self, lG, node, t, Kp):
		"""
			Get the Mixed-integer linear representation of this formula.
			This function will be used by the MILP solver to find the adequate
			Markov chain to probabilistically control the swarm
			:param lG : the labelled graph
			:param node : the node at which to evaluate the formula
			:param t : the time index at which the formula must be true
			:param Kp :  the graph trajectory length cycle
		"""
		newCoeffs = list()
		newVars = list()
		rhsVals = list()

		orList = list()
		for i in range(1,Kp+1):
			ljVar = (i, BOOL_VAR-2)
			lVar = []
			for j in range(i,Kp+1):
				nCoeffs, nVars, rhsVs, fEval = \
					self.rFormula.milp_repr(lG, node, j, Kp)
				newCoeffs.extend(nCoeffs)
				newVars.extend(nVars)
				rhsVals.extend(rhsVs)
				lVar.append(fEval)
			nC1, nV1, rhsV1, nVar1 = or_op(lVar)
			newCoeffs.extend(nC1)
			newVars.extend(nV1)
			rhsVals.extend(rhsV1)

			nC1, nV1, rhsV1, nVar1 = and_op([ljVar, nVar1])
			newCoeffs.extend(nC1)
			newVars.extend(nV1)
			rhsVals.extend(rhsV1)

			orList.append(nVar1)

		nC1, nV1, rhsV1, nVar1 = or_op(orList)
		newCoeffs.extend(nC1)
		newVars.extend(nV1)
		rhsVals.extend(rhsV1)
		return newCoeffs, newVars, rhsVals, nVar1

class NeighborGTL(GTLFormula):
	"""
		This formula represents an the liveness property Always eventually.
	"""
	def __init__(self, rFormula, D, N):
		self.rFormula = rFormula
		self.N = N
		self.D = D

	def __repr__(self):
		return 'O({})'.format(self.rFormula)

	def milp_repr(self, lG, node, t, Kp):
		"""
			Get the Mixed-integer linear representation of this formula.
			This function will be used by the MILP solver to find the adequate
			Markov chain to probabilistically control the swarm
			:param lG : the labelled graph
			:param node : the node at which to evaluate the formula
			:param t : the time index at which the formula must be true
			:param Kp :  the graph trajectory length cycle
		"""
		Sv = get_neighbors(lG, node, self.D)
		assert len(Sv) >= self.N
		allSubset = list(itertools.combinations(Sv, self.N))
		newCoeffs = list()
		newVars = list()
		rhsVals = list()

		or_list = []
		for lSubset in allSubset:
			and_list = []
			for v in lSubset:
				nCoeffs, nVars, rhsVs, fEval = \
							self.rFormula.milp_repr(lG, v, t, Kp)
				newCoeffs.extend(nCoeffs)
				newVars.extend(nVars)
				rhsVals.extend(rhsVs)
				and_list.append(fEval)
			nC1, nV1, rhsV1, nVar1 = and_op(and_list)
			newCoeffs.extend(nC1)
			newVars.extend(nV1)
			rhsVals.extend(rhsV1)
			or_list.append(nVar1)
		nC1, nV1, rhsV1, nVar1 = or_op(or_list)
		newCoeffs.extend(nC1)
		newVars.extend(nV1)
		rhsVals.extend(rhsV1)
		return newCoeffs, newVars, rhsVals, nVar1


if __name__ == "__main__":
    """
    Example of utilization of this class
    """
    V = set({1, 2, 3})
    lG = LabelledGraph(V)

    # Add the subswarm with id 1
    lG.addSubswarm(0, [(1,2), (2,3), (2,1), (3,2)])

    # Add the subswarm with id 1
    lG.addSubswarm(1, [(1,2), (2,3), (2,1), (3,2)])

    # Get the density symbolic representation for defining the node labels
    x = lG.getRprDensity()

    # Now add some node label on the graph
    lG.addNodeLabel(1, x[(0,1)] + x[(0,2)])
    lG.addNodeLabel(2, x[(0,1)] + x[(1,1)])
    lG.addNodeLabel(3, [x[(0,1)] + x[(0,2)], x[(0,3)] - x[(1,2)]])
    # Set the BIG M coeff to be 1 as for easy interpretation
    BIG_M = 0

    # Create an atomic GTL specification
    piV1 = AtomicGTL([0.5])
    piV2 = AtomicGTL([0.7])
    piV21 = AtomicGTL([0.8])
    piV22 = AtomicGTL([0.4])
    print_milp_repr([piV1], [1], 2, lG, initTime=0)

    piV3 = AtomicGTL([0.5, 0.25])
    piV31 = AtomicGTL([-0.5, 0.7])
    piV32 = AtomicGTL([-0.2, np.inf])
    print_milp_repr([piV3], [3], 2, lG, initTime=1)

    # Create an AND constraint
    and1 = AndGTL([piV1, piV2, piV21, piV22])
    print_milp_repr([and1], [1], 2, lG, initTime=1)
    and2 = AndGTL([piV3, piV31, piV32])
    print_milp_repr([and2], [3], 2, lG, initTime=1)

    # Create an OR constraint
    or1 = OrGTL([piV1, piV2, piV21, piV22])
    print_milp_repr([or1], [1], 2, lG, initTime=1)
    or2 = OrGTL([piV3, piV31, piV32])
    print_milp_repr([or2], [3], 2, lG, initTime=1)

    # # # Mix constraint
    # # or3 = OrGTL([and1, or1])
    # # print_milp_repr(or3, lG, 1, 0, 5)

    # Create Not constraint
    not1 = NotGTL(and1)
    print_milp_repr([not1], [1], 2, lG, initTime=1)

    # Create Next constraint
    next1 = NextGTL(and1)
    print_milp_repr([next1], [1], 2, lG, initTime=0)

    # Create Next constraint
    next1 = NextGTL(piV1)
    print_milp_repr([next1], [1], 5, lG, initTime=5)

    # Create always variable
    always1 = AlwaysGTL(piV1)
    print_milp_repr([always1], [1], 3, lG, initTime=0)

    # Create always eventually
    alwaysEven1 = AlwaysEventuallyGTL(piV1)
    print_milp_repr([alwaysEven1, always1], [1, 1], 3, lG, initTime=0)

    # Create always eventually
    evenAlways1 = EventuallyAlwaysGTL(piV1)
    print_milp_repr([evenAlways1], [1], 3, lG, initTime=0)

    # Create neighbor constraints
    neigh1 = NeighborGTL(piV1, 1, 1)
    print_milp_repr([neigh1], [1], 3, lG, initTime=0)