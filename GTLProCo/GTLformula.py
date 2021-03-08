import numpy as 
from .LabelledGraph import LabelledGraph
from abc import ABC, abstractmethod

STRICT_INEQ = 1e-8
BIG_M = 1e3
N_VARS = 0
BOOL_VAR = -1

def eval_formula(self, graphTraj, formula, lG, node):
	"""
		Eval the formula at the node given by node
		:param lG : labelled graph
		:param node : the node at which to evaluate the GTL formula
		:param graphTraj : a graph trajectory --> Value of all densities
		:param formula : A GTL formula
	"""
	pass

class GTLFormula(ABC):
	"""
		Abstract base  class that represents an infinite horizon GTL
	"""
	@abstractmethod
	def milp_repr(self, lG, node):
		"""
			Get the Mixed-integer linear representation of this formula.
			This function will be used by the MILP solver to find the adequate
			Markov chain to probabilistically control the swarm
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

	def milp_repr(self, lG, node):
		"""
			Get the Mixed-integer linear representation of this formula.
			This function will be used by the MILP solver to find the adequate
			Markov chain to probabilistically control the swarm
		"""
		# Sanity check if the constraint is the same size of the node label
		assert len(lG.getNodeLabel(node)) ==  self.c.shape[0]

		# Create the variable storing the result of the feasibility of this formula
		nVar = (N_VARS, BOOL_VAR) # -1 means it is a boolean var and -2 means it is not
		N_VARS += 1	# Increment the number of new variables

		# Get the MILP representation using the array repr of the linear label
		(coeffs, var) = lG.getLabelMatRepr(node)

		# Save the new constrainrs
		newCoeffs = list()
		newVars = list()
		rhsVal = list()

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
			rhsVal.append(-rhs - STRICT_INEQ)
			# Add the new coefficients and variables
			newCoeffs.append(tList1)
			newCoeffs.append(tList2)
			newVars.append(iList1)
			newVars.append(iList2)
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

	def milp_repr(self, lG, node):
		"""
			Get the Mixed-integer linear representation of this formula.
			This function will be used by the MILP solver to find the adequate
			Markov chain to probabilistically control the swarm
		"""
		# Create the variable storing the result of the feasibility of this formula
		nVar = (N_VARS, BOOL_VAR-1) # -1 means it is a boolean var and -2 means it is not
		N_VARS += 1	# Increment the number of new variables

		# Get the number of propositions in the AND operation
		nProp = len(self.mFormula)

		# Save the new constrainrs
		newCoeffs = list()
		newVars = list()
		rhsVals = list()

		# Get the MILP representation using the array repr of the linear label
		tContr = list()
		tCoeff =  list()
		for gtl in self.mFormula:
			nCoeffs, nVars, rhsVs, fEval = gtl.milp_repr(lG, node)
			newCoeffs.extend(nCoeffs)
			newVars.extend(nVars)
			rhsVals.extend(rhsVs)
			# Add the constraint by the and operator
			newCoeffs.append([1, -1])
			newVars.append([nVar, fEval])
			rhsVals.append(0)
			tContr.append(fEval)
			tCoeff.append(1)
		tContr.append(nVar)
		tCoeff.append(-1)
		# Add the last constraint
		newCoeffs.append(tCoeff)
		newVars.append(tContr)
		rhsVals.append(nProp-1)
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

	def milp_repr(self, lG, node):
		"""
			Get the Mixed-integer linear representation of this formula.
			This function will be used by the MILP solver to find the adequate
			Markov chain to probabilistically control the swarm
		"""
		# Create the variable storing the result of the feasibility of this formula
		nVar = (N_VARS, BOOL_VAR-1) # -1 means it is a boolean var and -2 means it is not
		N_VARS += 1	# Increment the number of new variables

		# Get the number of propositions in the AND operation
		nProp = len(self.mFormula)

		# Save the new constrainrs
		newCoeffs = list()
		newVars = list()
		rhsVals = list()

		# Get the MILP representation using the array repr of the linear label
		tContr = list()
		tCoeff =  list()
		for gtl in self.mFormula:
			nCoeffs, nVars, rhsVs, fEval = gtl.milp_repr(lG, node)
			newCoeffs.extend(nCoeffs)
			newVars.extend(nVars)
			rhsVals.extend(rhsVs)
			# Add the constraint by the or operator
			newCoeffs.append([-1, 1])
			newVars.append([nVar, fEval])
			rhsVals.append(0)
			# Save the sum of results of past formula
			tContr.append(fEval)
			tCoeff.append(-1)
		tContr.append(nVar)
		tCoeff.append(1)
		# Add the last constraint
		newCoeffs.append(tCoeff)
		newVars.append(tContr)
		rhsVals.append(0)
		return newCoeffs, newVars, rhsVals, nVar

class NotGTL(GTLFormula):
	"""
		This GTL formula represents the negation of a proposition
	"""
	def __init__(self, rFormula):
		self.rFormula = rFormula

	def milp_repr(self, lG, node):
		"""
			Get the Mixed-integer linear representation of this formula.
			This function will be used by the MILP solver to find the adequate
			Markov chain to probabilistically control the swarm
		"""
		# Create the variable storing the result of the feasibility of this formula
		nVar = (N_VARS, BOOL_VAR-1) # -1 means it is a boolean var and -2 means it is not
		N_VARS += 1	# Increment the number of new variables

		# Get the MILP representation of the formula after not operator
		nCoeffs, nVars, rhsVs, fEval = self.rFormula.milp_repr(lG, node)
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

	def milp_repr(self, lG, node):
		"""
			Get the Mixed-integer linear representation of this formula.
			This function will be used by the MILP solver to find the adequate
			Markov chain to probabilistically control the swarm
		"""
		