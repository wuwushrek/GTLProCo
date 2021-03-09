import numpy as np
from .LabelledGraph import LabelledGraph
from abc import ABC, abstractmethod

STRICT_INEQ = 1e-8
BIG_M = 1e3
N_VARS = 0
BOOL_VAR = -1

# def and_op(bVars):
# 	global BOOL_VAR, N_VARS

# 	# Create the variable storing the result of the feasibility of this formula
# 	nVar = (N_VARS, BOOL_VAR-1) # -1 means it is a boolean var and -2 means it is not
# 	N_VARS += 1	# Increment the number of new variables

# 	# Get the number of propositions in the AND operation
# 	nProp = len(bVars)

# 	# Save the new constrainrs
# 	newCoeffs = list()
# 	newVars = list()
# 	rhsVal = list()

# 	# Get the MILP representation using the array repr of the linear label
# 	tContr = list()
# 	tCoeff =  list()
# 	for varT in bVars:
# 		newCoeffs.extend(nCoeffs)
# 		newVars.extend(nVars)
# 		rhsVals.extend(rhsVs)
# 		timeVal.extend(tVal)
# 		# Add the constraint by the and operator
# 		newCoeffs.append([1, -1])
# 		newVars.append([nVar, fEval])
# 		rhsVals.append(0)
# 		timeVal.append(t)
# 		# Add the binding constraints of the result of each prop
# 		tContr.append(fEval)
# 		tCoeff.append(1)
# 	tContr.append(nVar)
# 	tCoeff.append(-1)
# 	# Add the last constraint
# 	newCoeffs.append(tCoeff)
# 	newVars.append(tContr)
# 	rhsVals.append(nProp-1)
# 	timeVal.append(t)
# 	return newCoeffs, newVars, rhsVals, nVar, timeVal

def eval_formula(self, graphTraj, formula, lG, node):
	"""
		Eval the formula at the node given by node
		:param lG : labelled graph
		:param node : the node at which to evaluate the GTL formula
		:param graphTraj : a graph trajectory --> Value of all densities
		:param formula : A GTL formula
	"""
	pass

def print_milp_repr(gtl, lG, node, t, Kp):
	"""
		Return a string of the MILP representation of this GTL formula
	"""
	newCoeffs, newVars, rhsVal, nVar, timeVal = gtl.milp_repr(lG, node, t, Kp)
	# Sanity check
	assert len(newVars) == len(newCoeffs) and len(newVars) == len(timeVal) and \
				len(newVars) == len(rhsVal)
	# Store the set of all variables of this MILP representation
	varBool = set()
	varReal = set()
	densDistr = set()
	ljVar = set()
	dictVar = dict()
	for vs, ts in zip(newVars, timeVal):
		for (ind1, ind2) in vs:
			if ind2 == BOOL_VAR:
				varRepr = 'b_{}({})'.format(ind1, ts)
				varBool.add(varRepr)
			elif ind2 == BOOL_VAR-1:
				varRepr = 'r_{}({})'.format(ind1, ts)
				varReal.add(varRepr)
			elif ind2 == BOOL_VAR-2:
				varRepr = 'l_{}'.format(ind1)
				ljVar.add(varRepr)
			else:
				varRepr = 'x_{}_{}({})'.format(ind1, ind2, ts)
				densDistr.add(varRepr)
			dictVar[(ind1, ind2, ts)] = varRepr
	resContr = list()
	for nCoeffs, vs, rhsV, ts in zip(newCoeffs, newVars, rhsVal, timeVal):
		contrRepr = ['{}*{}'.format(c,dictVar[(v[0], v[1], ts)]) \
												for c, v in zip(nCoeffs, vs)]
		sumRes = '{} <= {}'.format(' + '.join(contrRepr), rhsV)
		resContr.append(sumRes)

	# Start the printing formula
	# print('--------------------------------------------------')
	print(lG)
	print('------ GTL Formula -------')
	print ('At node {} and time index {} -- Length trajectory {}'.format(node, t, Kp))
	print(gtl)
	print('-------------------------')
	print('Result variables: {}'.format(dictVar[(nVar[0],nVar[1],timeVal[-1])]))
	print('-------------------------')
	print('Boolean variables in {0,1}')
	print(', '.join(varBool))
	print('-------------------------')
	print('Real variables in [0,1]')
	print(', '.join(varReal))
	print('-------------------------')
	print('Density variables in [0,1]')
	print(', '.join(densDistr))
	print('-------------------------')
	print('Loop variables in {0,1}')
	print(', '.join(ljVar))
	print('-------------------------')
	print('Constraints')
	print('\n'.join(resContr))
		

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
		nVar = (N_VARS, BOOL_VAR) # -1 means it is a boolean var and -2 means it is not
		N_VARS += 1	# Increment the number of new variables

		# Get the MILP representation using the array repr of the linear label
		(coeffs, var) = lG.getLabelMatRepr(node)

		# Save the new constrainrs
		newCoeffs = list()
		newVars = list()
		rhsVal = list()
		timeVal = list()

		# Add the new constraints
		for cV, indexV, rhs in zip(coeffs, var, self.c):
			tList1, iList1 = list(), list()
			tList2, iList2 = list(), list()
			for cvalue, (sId, nId) in zip(cV, indexV):
				# Add the current coefficient and the variable that it multiplies
				tList1.append(cvalue)
				iList1.append((sId, nId))
				# Add the negation of the past constraint A x > b -> -Ax < -b
				tList2.append(-cvalue)
				iList2.append((sId, nId))
			# Add the term (1-P) phi
			tList1.append(BIG_M)
			iList1.append(nVar)
			# Add the term -P phi
			tList2.append(-BIG_M)
			iList2.append(nVar)
			# The right right side
			rhsVal.append(rhs + BIG_M)
			timeVal.append(t)
			rhsVal.append(-rhs - STRICT_INEQ)
			timeVal.append(t)
			# Add the new coefficients and variables
			newCoeffs.append(tList1)
			newCoeffs.append(tList2)
			newVars.append(iList1)
			newVars.append(iList2)
		return newCoeffs, newVars, rhsVal, nVar, timeVal

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
		global N_VARS, BIG_M, STRICT_INEQ, BOOL_VAR

		# Create the variable storing the result of the feasibility of this formula
		nVar = (N_VARS, BOOL_VAR-1) # -1 means it is a boolean var and -2 means it is not
		N_VARS += 1	# Increment the number of new variables

		# Get the number of propositions in the AND operation
		nProp = len(self.mFormula)

		# Save the new constrainrs
		newCoeffs = list()
		newVars = list()
		rhsVals = list()
		timeVal = list()

		# Get the MILP representation using the array repr of the linear label
		tContr = list()
		tCoeff =  list()
		for gtl in self.mFormula:
			nCoeffs, nVars, rhsVs, fEval, tVal = gtl.milp_repr(lG, node, t, Kp)
			newCoeffs.extend(nCoeffs)
			newVars.extend(nVars)
			rhsVals.extend(rhsVs)
			timeVal.extend(tVal)
			# Add the constraint by the and operator
			newCoeffs.append([1, -1])
			newVars.append([nVar, fEval])
			rhsVals.append(0)
			timeVal.append(t)
			# Add the binding constraints of the result of each prop
			tContr.append(fEval)
			tCoeff.append(1)
		tContr.append(nVar)
		tCoeff.append(-1)
		# Add the last constraint
		newCoeffs.append(tCoeff)
		newVars.append(tContr)
		rhsVals.append(nProp-1)
		timeVal.append(t)
		return newCoeffs, newVars, rhsVals, nVar, timeVal

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
		global N_VARS, BIG_M, STRICT_INEQ, BOOL_VAR

		# Create the variable storing the result of the feasibility of this formula
		nVar = (N_VARS, BOOL_VAR-1) # -1 means it is a boolean var and -2 means it is not
		N_VARS += 1	# Increment the number of new variables

		# Get the number of propositions in the AND operation
		nProp = len(self.mFormula)

		# Save the new constrainrs
		newCoeffs = list()
		newVars = list()
		rhsVals = list()
		timeVal = list()

		# Get the MILP representation using the array repr of the linear label
		tContr = list()
		tCoeff =  list()
		for gtl in self.mFormula:
			nCoeffs, nVars, rhsVs, fEval, tVal = gtl.milp_repr(lG, node, t, Kp)
			newCoeffs.extend(nCoeffs)
			newVars.extend(nVars)
			rhsVals.extend(rhsVs)
			timeVal.extend(tVal)
			# Add the constraint by the or operator
			newCoeffs.append([-1, 1])
			newVars.append([nVar, fEval])
			rhsVals.append(0)
			timeVal.append(t)
			# Save the sum of results of past formula
			tContr.append(fEval)
			tCoeff.append(-1)
		tContr.append(nVar)
		tCoeff.append(1)
		# Add the last constraint
		newCoeffs.append(tCoeff)
		newVars.append(tContr)
		rhsVals.append(0)
		timeVal.append(t)
		return newCoeffs, newVars, rhsVals, nVar, timeVal

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
		nCoeffs, nVars, rhsVs, fEval, tVal = self.rFormula.milp_repr(lG, node, t, Kp)

		# Add the equality constraint by the NOT operator
		nCoeffs.append([1,1])
		nVars.append([nVar, fEval])
		rhsVs.append(1)
		tVal.append(t)

		nCoeffs.append([-1,-1])
		nVars.append([nVar, fEval])
		rhsVs.append(-1)
		tVal.append(t)
		return nCoeffs, nVars, rhsVs, nVar, tVal

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
		global N_VARS, BIG_M, STRICT_INEQ, BOOL_VAR
		if t < Kp:
			# Get the MILP representation of the formula after not operator
			return self.rFormula.milp_repr(lG, node, t+1, Kp)

		nVar = (N_VARS, BOOL_VAR-1) # -1 means it is a boolean var and -2 means it is not
		N_VARS += 1	# Increment the number of new variables

		# In case t == Kp, the next time step is implied by the position of the loop
		newCoeffs = list()
		newVars = list()
		rhsVals = list()
		timeVal = list()
		tContr = list()
		tCoeff =  list()
		for i in range(1, Kp+1):
			nCoeffs, nVars, rhsVs, fEval, tVal = self.rFormula.milp_repr(lG, node, i, Kp)
			ljVar = (i, BOOL_VAR-2)
			newCoeffs.extend(nCoeffs)
			newVars.extend(nVars)
			rhsVals.extend(rhsVs)
			timeVal.extend(tVal)
			# Intersection lj nVar constraint
			ntVar = (N_VARS, BOOL_VAR-1) # -1 means it is a boolean var and -2 means it is not
			N_VARS += 1	# Increment the number of new variables
			
			newCoeffs.append([1, -1])
			newVars.append([ntVar, fEval])
			rhsVals.append(0)
			timeVal.append(i)

			newCoeffs.append([1, -1])
			newVars.append([ntVar, ljVar])
			rhsVals.append(0)
			timeVal.append(i)

			newCoeffs.append([1, 1, -1])
			newVars.append([fEval, ljVar, ntVar])
			rhsVals.append(1)
			timeVal.append(i)

			# Add the binding constraints of the result of each prop
			newCoeffs.append([-1, 1])
			newVars.append([nVar, ntVar])
			rhsVals.append(0)
			timeVal.append(i)

			tContr.append(ntVar)
			tCoeff.append(-1)
		tContr.append(nVar)
		tCoeff.append(1)
		# Add the last constraint
		newCoeffs.append(tCoeff)
		newVars.append(tContr)
		rhsVals.append(0)
		timeVal.append(Kp)
		return newCoeffs, newVars, rhsVals, nVar, timeVal

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
    print_milp_repr(piV1, lG, 1, 0, 5)

    piV3 = AtomicGTL([0.5, 0.25])
    piV31 = AtomicGTL([-0.5, 0.7])
    piV32 = AtomicGTL([-0.2, np.inf])
    print_milp_repr(piV3, lG, 3, 1, 5)

    # Create an AND constraint
    and1 = AndGTL([piV1, piV2, piV21, piV22])
    print_milp_repr(and1, lG, 1, 0, 5)
    and2 = AndGTL([piV3, piV31, piV32])
    print_milp_repr(and2, lG, 3, 1, 5)

    # Create an OR constraint
    or1 = OrGTL([piV1, piV2, piV21, piV22])
    print_milp_repr(or1, lG, 1, 0, 5)
    or2 = OrGTL([piV3, piV31, piV32])
    print_milp_repr(or2, lG, 3, 1, 5)

    # # Mix constraint
    # or3 = OrGTL([and1, or1])
    # print_milp_repr(or3, lG, 1, 0, 5)

    # Create Not constraint
    not1 = NotGTL(and1)
    print_milp_repr(not1, lG, 1, 0, 5)

    # Create Next constraint
    next1 = NextGTL(and1)
    print_milp_repr(next1, lG, 1, 0, 5)

    # Create Next constraint
    next1 = NextGTL(piV1)
    print_milp_repr(next1, lG, 1, 5, 5)