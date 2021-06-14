import numpy as np
import matplotlib
import matplotlib.pyplot as plt

solverList = ['GTLProco', 'GTLProco_SDP', 'GTLProco_LP', 'Gurobi_MINLP', 'SCIP', 'couenne', 'bonmin']
data_dir = '/home/fdjeumou/Documents/GTLProCo/'
fileList =  ['final_results.npz']
# Load the data
mData = dict()
for fName in fileList:
	mDataT = np.load(data_dir+fName)
	for solver in solverList:
		if solver not in mData:
			mData[solver] = mDataT[solver] # [mDataT[solver][:,1] >= 0]
		else:
			mData[solver] = np.concatenate((mData[solver], mDataT[solver]))
	
size_pb = np.array([])
for solver in solverList:
	uni_length = np.unique(mData[solver][:,0])
	if size_pb.shape[0] <= uni_length.shape[0]:
		size_pb = uni_length.astype(dtype=np.int64)

# Maximum duration
MAX_DURATION = 60*5

# Find the meantime, standard deviation, maximum and minimum for each algo
res_solver = dict()
for solver in solverList:
	m_solv = mData[solver]
	res_array = np.zeros((size_pb.shape[0],6))
	curr_size = 0
	for i in range(size_pb.shape[0]):
		tVal = m_solv[m_solv[:,0] == size_pb[i], :]
		for j in range(tVal.shape[0]):
			tVal[j,-2] = np.minimum(tVal[j,-2], MAX_DURATION)
		if tVal.shape[0] == 0:
			res_array[i,0] = MAX_DURATION
			res_array[i,1] = MAX_DURATION
			res_array[i,2] = MAX_DURATION
			continue
		meanVal = np.mean(tVal[:,-2])
		stdVal = np.std(tVal[:,-2])
		maxVal = np.minimum(np.max(tVal[:,-2]), meanVal+stdVal)
		minVal = np.maximum(np.min(tVal[:,-2]), meanVal-stdVal)
		res_array[i,0] = meanVal
		res_array[i,1] = minVal
		res_array[i,2] = maxVal
		ind = tVal[:,-1] >= 0
		if sum(ind) == 0:
			curr_size = i
			continue
		meanVal = np.mean(tVal[ind,-1])
		stdVal = np.std(tVal[ind,-1])
		maxVal = np.minimum(np.max(tVal[ind,-1]), meanVal+stdVal)
		minVal = np.maximum(np.min(tVal[ind,-1]), meanVal-stdVal)
		res_array[i,3] = meanVal
		res_array[i,4] = minVal
		res_array[i,5] = maxVal
		curr_size = i+1
	# res_array = res_array[:curr_size,:]
	res_solver[solver] = (res_array, curr_size)

# Clean the GTLProco data by merging the LP and SDP solver 
datasdp, csizesdp = res_solver['GTLProco_SDP']
datalp, csizelp = res_solver['GTLProco_LP']
datagtl, csize = res_solver['GTLProco']
datagtl[:csizesdp,:] = np.minimum(datasdp[:csizesdp,:], datagtl[:csizesdp,:])
datagtl[:csizelp,:] = np.minimum(datalp[:csizelp,:], datagtl[:csizelp,:])
datagtl[datagtl == 0] =1e-15
res_solver['couenne'][0][res_solver['couenne'][0] == 0] =1e-15
res_solver['bonmin'][0][res_solver['bonmin'][0] == 0] =1e-15
res_solver['SCIP'][0][res_solver['SCIP'][0] == 0] =1e-15
res_solver.pop('GTLProco_SDP')
res_solver.pop('GTLProco_LP')

# Plot the solving time for each of the solver
colorSet = {'GTLProco':'blue', 'Gurobi_MINLP':'green', 'SCIP':'cyan', 'couenne':'gray', 'bonmin':'magenta'}
markerSet = {'GTLProco':'o', 'Gurobi_MINLP':'*', 'SCIP':'D', 'couenne':'^', 'bonmin':'x'}
linewidth = 2
markerSize = 5
lineWidthTargetDuration = 6

plt.figure()
plt.plot(size_pb, [MAX_DURATION for i in range(size_pb.shape[0])],
				linewidth=lineWidthTargetDuration, color='orange', label='Time limit')
for solver, (dataInfo, currSize) in res_solver.items():
	plt.plot(size_pb, dataInfo[:,0], linewidth=linewidth, marker=markerSet[solver], 
				markersize=markerSize, color=colorSet[solver], label=solver)
	plt.fill_between(size_pb, dataInfo[:,1], dataInfo[:,2], color=colorSet[solver], alpha=0.5)
plt.ylabel('Compute time (s)')
plt.xlabel('Number of bins')
plt.yscale('log')
plt.grid(True)
plt.legend(ncol=3, bbox_to_anchor=(0,1), loc='lower left', columnspacing=1.0)
plt.tight_layout()

# Save the image
import tikzplotlib
tikzplotlib.clean_figure()
tikzplotlib.save('compute_time.tex')

# # Plot the attained accuracy for each of the solver
# plt.figure()
# for solver, (dataInfo, currSize) in res_solver.items():
# 	plt.plot(size_pb[:currSize], dataInfo[:currSize,3], linewidth=linewidth, marker=markerSet[solver], 
# 				markersize=markerSize, color=colorSet[solver], label=solver)
# 	plt.fill_between(size_pb[:currSize], dataInfo[:currSize,4], dataInfo[:currSize,5], color=colorSet[solver], alpha=0.5)
# plt.ylabel('Accuracy of bilinear constraint')
# plt.xlabel('Number of bins')
# plt.yscale('log')
# plt.grid(True)
# plt.legend(ncol=3, bbox_to_anchor=(0,1), loc='lower left', columnspacing=1.0)
# plt.tight_layout()

plt.show()