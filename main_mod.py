from dimod import ConstrainedQuadraticModel, CQM, Binary, Spin, Real, quicksum
from dwave.system import LeapHybridCQMSampler
from dimod import ExactSolver

import random 
import numpy as np
from scipy.io import loadmat as loadmat
from scipy.io import savemat as savemat
from funcs import nth_upper_diagonal_indices as nth_upper_diagonal_indices

R = loadmat('Rquan.mat')['Rquan']

(M,M) = np.shape(R)

num_waveforms = M
num_parts = int((M**2-M)/2)
num_diags = M-2
len_waveforms = 100

num_items = num_waveforms*len_waveforms

SM = np.zeros((num_parts, 2), 'int')
cntr = 0
for ii in range(0,M):
    for jj in range(ii+1,M):
        SM[cntr, 0] = ii
        SM[cntr, 1] = jj
        cntr += 1

diag_sums = []
for dd in range(1,num_diags+1):
    diag_sums.append(np.sum(R[nth_upper_diagonal_indices(M, dd)]))


##------------------------------------- Building objective function ------------------------------------------##

cqm = ConstrainedQuadraticModel()

waveforms = [[Spin(jj*len_waveforms + ii) for ii in range(len_waveforms)] for jj in range(num_waveforms)]

slack = [Real(num_waveforms*len_waveforms + ii) for ii in range(num_diags)]

objective = sum(slack[ii] for ii in range(num_diags))
cqm.set_objective(objective)

for rr in range(1, num_diags+1):
    idx0, idx1 = nth_upper_diagonal_indices(M, rr)
    diag_len = len(idx0)
    cqm.add_constraint(-1*slack[rr-1] - sum(sum((waveforms[idx0[dd]][ii])*(waveforms[idx1[dd]][ii]) for ii in range(len_waveforms)) for dd in range(diag_len))/(len_waveforms) <= -1*diag_sums[rr-1], label='Less or equal_'+str(rr))

for rr in range(1, num_diags+1):
    idx0, idx1 = nth_upper_diagonal_indices(M, rr)
    diag_len = len(idx0)
    cqm.add_constraint(-1*slack[rr-1] + sum(sum((waveforms[idx0[dd]][ii])*(waveforms[idx1[dd]][ii]) for ii in range(len_waveforms)) for dd in range(diag_len))/(len_waveforms) <= diag_sums[rr-1], label='Greater or equal_'+str(rr))


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