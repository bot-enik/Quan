# Second version with working 2x2 case, but binary. Now need to translate to +1 or -1

from dimod import ConstrainedQuadraticModel, CQM, Binary, Spin, Real, quicksum
from dwave.system import LeapHybridCQMSampler
from dimod import ExactSolver

import random 
import numpy as np
from scipy.io import loadmat as loadmat
from scipy.io import savemat as savemat

R = loadmat('Rrepr.mat')['R_binary_representable']

(M,M) = np.shape(R)

num_waveforms = M
num_parts = int((M**2-M)/2)
len_waveforms = 100

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

waveforms = [[Spin(jj*len_waveforms + ii) for ii in range(len_waveforms)] for jj in range(num_waveforms)]

slack = [Real(num_waveforms*len_waveforms + ii) for ii in range(num_parts)]

objective = sum(slack[ii] for ii in range(num_parts))
cqm.set_objective(objective)

 # Method for {-1, +1} variables:
for rr in range(num_parts):
    idx0 = SM[rr][0]
    idx1 = SM[rr][1]
    cqm.add_constraint(-1*slack[rr] - sum((waveforms[idx0][ii])*(waveforms[idx1][ii]) for ii in range(len_waveforms))/(len_waveforms) <= -1*R[idx0][idx1], label='Less or equal_'+str(rr))

for rr in range(num_parts):
    idx0 = SM[rr][0]
    idx1 = SM[rr][1]
    cqm.add_constraint(-1*slack[rr] + sum((waveforms[idx0][ii])*(waveforms[idx1][ii]) for ii in range(len_waveforms))/(len_waveforms) <= R[idx0][idx1], label='Greater or equal_'+str(rr))


cqm_sampler = LeapHybridCQMSampler()
sampleset = cqm_sampler.sample_cqm(cqm, label='Waveform generation')
print(sampleset.info)
print(sampleset)

np.save('sampleset.npy',sampleset.record, allow_pickle=True)

# %%%%%%%%%%%%% Second part
sampleset = np.load('sampleset.npy', allow_pickle=True)
sample = sampleset['sample']
ind_feasible_sols = np.reshape(np.where(sampleset['is_feasible'] == True), (-1,))
energy = sampleset['energy']

sample_feasible = sample[ind_feasible_sols, :]
energy_feasible = energy[ind_feasible_sols]
minenergy_feasible = sample_feasible[np.argmin(energy_feasible)]

chosen_sample = minenergy_feasible
#chosen_sample = sample[1,:]

S = chosen_sample[0:len_waveforms*M]
num_parts = int((M**2-M)/2)
X = np.reshape(S, (M,len_waveforms))

R = X @ np.transpose(X) / (len_waveforms)

savemat('Rgen.mat', {'Rgen':R})