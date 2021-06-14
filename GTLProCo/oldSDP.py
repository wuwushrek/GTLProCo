#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cvxpy as cp
import numpy as np
import numpy.linalg as la

USE_ERG_COEFF = 0
USE_REV_APPROX = 1
USE_NREV_APPROX = 2
USE_OTHER = 3

# Assumptions and input :
# @v  : target (desired) swarm distribution is known positive and fixed
# @Aa : Adjacency matrix representing the physical transition between bins
# @L  : Matrix specifying the constraints on the evolution of x(t)
# @p  : vector specifying the constraints on the evolution of x(t)
# Basically L [x(t) ,x(t+1), ... , x(t+h)] <= p
# safe: Argument to actually include safety constraints or not
#
# @eps: small number of enforcing reversibility of the synthesized markov matrix
# Return :
# @M  : markov matrix s.t x(t+1) = M x(t)
# @mu : upper bound for the abs(eigen value)
def computeMforFixedV(Aa, v, K,d, L, p, method=USE_ERG_COEFF, safe=False, eps=1e-8):
    #---------------------------------------------------
    # Dimensions useful in the problem
    nbins = Aa.shape[0] # Number of bins

    #---------------------------------------------------
    # check for zero entry in v
    max_zero_id = 0
    for i in range(v.shape[0]-1):
        if v[i] <= eps:
            max_zero_id += 1
        else:
            break

    # Create problem variables
    if max_zero_id == 0:
        M = cp.Variable(shape=(nbins,nbins), nonneg=True) # markov matrix
    else:
        M1 = cp.Variable(shape=(max_zero_id,max_zero_id), nonneg=True)
        M2 = cp.Variable(shape=(nbins-max_zero_id,max_zero_id), nonneg=True)
        M4 = cp.Variable(shape=(nbins-max_zero_id,nbins-max_zero_id), nonneg=True)
        M = cp.bmat([[M1 , np.zeros((max_zero_id,nbins-max_zero_id))],[M2,M4]])
    # Add constraints variables
    if safe :
        S = cp.Variable(shape=(p.shape[0], K.shape[0]), nonpos=True)
        y = cp.Variable(shape=(p.shape[0],1))

    #----------------------------------------------------
    # Create constraints for the problem
    pbConstr = list()

    # Ergodic and Motion constraints + Stochasticity of M
    pbConstr.append(np.ones(nbins) * M == np.ones(nbins)) # Constrainst 1^T M = 1^T
    pbConstr.append(cp.multiply((np.ones((nbins,nbins)) - Aa.T), M) == 0) # motions constr
   
    # Ensures ergocity of M when Aa is connected
    if max_zero_id == 0:
        pbConstr.append(M >= eps* Aa.T)
    else:
        pbConstr.append(M1 >= eps * Aa.T[:max_zero_id,:max_zero_id])
        pbConstr.append(M2 >= eps * Aa.T[max_zero_id:,:max_zero_id])
        pbConstr.append(M4 >= eps * Aa.T[max_zero_id:,max_zero_id:])

    # Checking for reversible Markov matrices or not
    if method == USE_REV_APPROX:
        pbConstr.append(M * cp.diag(v) == cp.diag(v) * M.T) # Reversible condition
    else:
        pbConstr.append(M * v == v) # stationnary distribution constraints

    # Density evolution constraints
    if safe:
        pbConstr.append(S*d + y + p >= 0)
        pbConstr.append(S*K + y * np.ones((1,nbins)) + L * cp.bmat([[np.identity(nbins)],[M]]) <= 0)

    #-----------------------------------------------------
    # Create the objective function <-> goal to minimize ergocity coefficient
    if method == USE_ERG_COEFF:
        idMatrix = np.identity(nbins-max_zero_id)
        objSubList = []
        for i in range(nbins-max_zero_id):
            for j in range(nbins-max_zero_id):
                objSubList.append([cp.norm(M[max_zero_id:,max_zero_id:] *(idMatrix[:,i] - idMatrix[:,j]),1)])
        objMatrix = cp.bmat(objSubList)
        if  max_zero_id == 0:
            costFun = 0.5 * cp.max(objMatrix)
        else:
            auxCostFun = cp.max(np.ones((1,max_zero_id)) * M[:max_zero_id,:max_zero_id])
            costFun = cp.maximum(0.5 * cp.max(objMatrix) , auxCostFun)
        problem = cp.Problem(cp.Minimize(costFun), pbConstr)
        problem.solve(solver=cp.GUROBI,verbose=True) # Gurobi ? MOSEK slower for LP

    # For reversible M, the second eigen value of M can be compute :
    if method == USE_REV_APPROX:
        q_vec = np.sqrt(v[max_zero_id:,0])
        q_vec.shape = (q_vec.shape[0], 1)
        Q_mat = cp.diag(q_vec)
        Q_mat_inv = cp.diag(cp.inv_pos(q_vec))
        # SECOND EIGENVALUE OF M has to be minimize  -> thanks to reversibility <->
        if max_zero_id == 0:
            costFun = cp.lambda_max(Q_mat_inv * M * Q_mat - q_vec*q_vec.T)
        else:
            costFun1 = cp.lambda_max(Q_mat_inv * M4 * Q_mat - q_vec*q_vec.T)
            auxCostFun = cp.max(np.ones((1,max_zero_id)) * M[:max_zero_id,:max_zero_id])
            costFun = cp.maximum(costFun1 , auxCostFun)
        problem = cp.Problem(cp.Minimize(costFun), pbConstr)
        problem.solve(solver=cp.MOSEK,verbose=True)
        # return M.value , problem.value
    # For non Rev M, an upper bound of the optimal second value of M can be found
    # For General P row stochastic, We define M(P) = P * Prev with
    # Prev = inv(diag(v)) * P.T * diag(v) hence M(P) is reversible and we can
    # prove that the mixing rate of P is less than mixing rate of M(P)
    # also more algebra prove that :
    # lambda2(M(P)) = norm2_square(Diag(q)*P*inv(diag(q)) - q*q.T) for q = sqrt(v)
    if method == USE_NREV_APPROX:
        q_vec = np.sqrt(v[max_zero_id:,0])
        q_vec.shape = (q_vec.shape[0], 1)
        Q_mat = cp.diag(q_vec)
        Q_mat_inv = cp.diag(cp.inv_pos(q_vec))
        if max_zero_id == 0:
            costFun = cp.norm(Q_mat_inv * M * Q_mat - q_vec*q_vec.T , 2)
        else:
            costFun1 = cp.norm(Q_mat_inv * M4 * Q_mat - q_vec*q_vec.T , 2)
            auxCostFun = cp.max(np.ones((1,max_zero_id)) * M[:max_zero_id,:max_zero_id])
            costFun = cp.maximum(costFun1 , auxCostFun)
        problem = cp.Problem(cp.Minimize(costFun), pbConstr)
        problem.solve(solver=cp.MOSEK,verbose=True)
        # return M.value , problem.value
    if method == USE_OTHER:
        costFun = cp.norm(M.T - np.ones((nbins,nbins)) * M * (1.0 / nbins))
        problem = cp.Problem(cp.Minimize(costFun), pbConstr)
        problem.solve(solver=cp.MOSEK, verbose=True)
    if max_zero_id == 0:
        return M.value , problem.value
    else:
        res = np.zeros((nbins,nbins))
        res[:max_zero_id,:max_zero_id] = M1.value[:,:]
        res[max_zero_id:,:max_zero_id] = M2.value[:,:]
        res[max_zero_id:,max_zero_id:] = M4.value[:,:]
        return res, problem.value