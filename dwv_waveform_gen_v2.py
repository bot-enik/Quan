# Second version with working 2x2 case, but binary. Now need to translate to +1 or -1

from dimod import ConstrainedQuadraticModel, CQM, Binary, quicksum
from dwave.system import LeapHybridCQMSampler
from dimod import ExactSolver

import random 
import numpy as np
from scipy.io import loadmat as loadmat

R = [[0, 0.2250, 0.3500],
     [0, 0, 0.2500],
     [0, 0, 0]]

#R = [[0, 0.2250],
#     [0, 0]]

R = loadmat('R.mat')['R']

(M,M) = np.shape(R)

num_waveforms = M
num_parts = int((M**2-M)/2)
len_waveforms = 40

num_items = num_waveforms*len_waveforms

SM = np.zeros((num_parts, 2), 'int')
cntr = 0
for ii in range(0,M):
    for jj in range(ii+1,M):
        SM[cntr, 0] = ii
        SM[cntr, 1] = jj
        cntr += 1

##------------------------------------- Building objective function ------------------------------------------##

cqm = ConstrainedQuadraticModel()

waveforms = [[Binary(jj*len_waveforms + ii) for ii in range(len_waveforms)] for jj in range(num_waveforms)]
slack = [Binary(num_waveforms*len_waveforms + ii) for ii in range(num_parts)]

objective = sum(slack[ii] for ii in range(num_parts))
cqm.set_objective(objective)

for rr in range(num_parts):
    idx0 = SM[rr][0]
    idx1 = SM[rr][1]
    cqm.add_constraint(-1*slack[rr] - sum(waveforms[idx0][ii]*waveforms[idx1][ii] for ii in range(len_waveforms))/len_waveforms <= -1*R[idx0][idx1], label='Less or equal_'+str(rr))

for rr in range(num_parts):
    idx0 = SM[rr][0]
    idx1 = SM[rr][1]
    cqm.add_constraint(-1*slack[rr] + sum(waveforms[idx0][ii]*waveforms[idx1][ii] for ii in range(len_waveforms))/len_waveforms <= R[idx0][idx1], label='Greater or equal_'+str(rr))

for rr in range(num_waveforms):
    cqm.add_constraint(sum(waveforms[rr][ii]*waveforms[rr][ii] for ii in range(len_waveforms))/len_waveforms == R[rr][rr], label='Equality_'+str(rr))


cqm_sampler = LeapHybridCQMSampler()
sampleset = cqm_sampler.sample_cqm(cqm, label='Waveform generation')
print(sampleset.info)
print(sampleset)

feasible_sols = np.where(sampleset.record.is_feasible == True)
first_feasible_sol = np.where(sampleset.record[feasible_sols[0][0]][0] == 1)
print(first_feasible_sol)

np.save('sampleset.npy',sampleset.record, allow_pickle=True)