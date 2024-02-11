# First version with naive absolute value approximation 


from dimod import ConstrainedQuadraticModel, CQM, Binary, quicksum
from dwave.system import LeapHybridCQMSampler
from dimod import ExactSolver

import random 
import numpy as np

R = [[0, 0.2250, 0.3500],
     [0, 0, 0.2500],
     [0, 0, 0]]

(M,M) = np.shape(R)

num_waveforms = M
len_waveforms = 40

num_items = num_waveforms*len_waveforms


SM = np.zeros((int((M**2-M)/2), 2), 'int')
cntr = 0
for ii in range(0,M):
    for jj in range(ii+1,M):
        SM[cntr, 0] = ii
        SM[cntr, 1] = jj
        cntr += 1

##------------------------------------- Building objective function ------------------------------------------##

cqm = ConstrainedQuadraticModel()

waveforms = [[Binary(jj*len_waveforms + ii) for ii in range(len_waveforms)] for jj in range(num_waveforms)]

objective = 0
for rr in range(num_waveforms-1):
    idx0 = SM[rr][0]
    idx1 = SM[rr][1]
    objective += sum(waveforms[idx0][ii]*waveforms[idx1][ii] for ii in range(len_waveforms))/len_waveforms - R[idx0][idx1]

cqm.set_objective(objective)

for rr in range(num_waveforms-1):
    idx0 = SM[rr][0]
    idx1 = SM[rr][1]
    cqm.add_constraint(-1*sum(waveforms[idx0][ii]*waveforms[idx1][ii] for ii in range(len_waveforms))/len_waveforms <= -1*R[idx0][idx1])


cqm_sampler = LeapHybridCQMSampler()
sampleset = cqm_sampler.sample_cqm(cqm, label='waveform generation')
print(sampleset.info)
print(sampleset)

feasible_sols = np.where(sampleset.record.is_feasible == True)
first_feasible_sol = np.where(sampleset.record[feasible_sols[0][0]][0] == 1)
print(first_feasible_sol)



np.save('sampleset.npy',sampleset.record, allow_pickle=True)