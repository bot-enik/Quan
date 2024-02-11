
from dimod import ConstrainedQuadraticModel, CQM, Binary, quicksum, Spin, Real
from dwave.system import LeapHybridCQMSampler
from dimod import ExactSolver

import random 
import numpy as np
from scipy.io import loadmat as loadmat


#----------- First part: reduce the correlation matrix to smaller ------------#

R = loadmat('Rquan.mat')['Rquan']
(M,M) = np.shape(R)

Mnew = 7; # Dimensions of the lower right corner
len_waveforms = 100

R = R[M-Mnew:, M-Mnew:]
(M,M) = np.shape(R)

num_waveforms = M
num_parts = int((M**2-M)/2)


num_items = num_waveforms*len_waveforms

SM = np.zeros((num_parts, 2), 'int')
cntr = 0
for ii in range(0,M):
    for jj in range(ii+1,M):
        SM[cntr, 0] = ii
        SM[cntr, 1] = jj
        cntr += 1

##------ Building objective function ------##

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



#---------------------------------- Second part ------------------------------#

from dimod import ConstrainedQuadraticModel, CQM, Binary, quicksum, Spin, Real
from dwave.system import LeapHybridCQMSampler
from dimod import ExactSolver

import random 
import numpy as np
from scipy.io import loadmat as loadmat

X = loadmat('Xwaveforms.mat')['X']
R = loadmat('Rquan.mat')['Rquan']
(M,M) = np.shape(R)

Mprev = 7;
len_waveforms = 100
num_waveforms = M-Mprev
num_parts_var = int(((M-Mprev)*(Mprev+M-1))/2)
num_parts_fxd = int((Mprev*(Mprev-1))/2)
num_parts = num_parts_var + num_parts_fxd

num_items = num_waveforms*len_waveforms

SM = np.zeros((num_parts_var, 2), 'int')
cntr = 0
for ii in range(0,M-Mprev):
    for jj in range(ii+1,M):
        SM[cntr, 0] = ii
        SM[cntr, 1] = jj
        cntr += 1

##------ Building objective function ------##

cqm = ConstrainedQuadraticModel()

waveforms = [[Spin(jj*len_waveforms + ii) for ii in range(len_waveforms)] for jj in range(num_waveforms)]

slack = [Real(num_waveforms*len_waveforms + ii) for ii in range(num_parts_var)]

objective = sum(slack[ii] for ii in range(num_parts_var))
cqm.set_objective(objective)

for rr in range(num_parts_var):
    idx0 = SM[rr][0]
    idx1 = SM[rr][1]
    if idx1 < M-Mprev: # Constraints for variables with themselves
        cqm.add_constraint(-1*slack[rr] - sum((waveforms[idx0][ii])*(waveforms[idx1][ii]) for ii in range(len_waveforms))/(len_waveforms) <= -1*R[idx0][idx1], label='Less or equal_'+str(rr))
    else: # Constraints for variables with fixed waveforms
        cqm.add_constraint(-1*slack[rr] - sum((waveforms[idx0][ii])*(X[idx1-M+Mprev][ii]) for ii in range(len_waveforms))/(len_waveforms) <= -1*R[idx0][idx1], label='Less or equal_'+str(rr))


for rr in range(num_parts_var):
    idx0 = SM[rr][0]
    idx1 = SM[rr][1]
    if idx1 < M-Mprev: # Constraints for variables with themselves
        cqm.add_constraint(-1*slack[rr] + sum((waveforms[idx0][ii])*(waveforms[idx1][ii]) for ii in range(len_waveforms))/(len_waveforms) <= R[idx0][idx1], label='Greater or equal_'+str(rr))
    else: # Constraints for variables with fixed waveforms
        cqm.add_constraint(-1*slack[rr] + sum((waveforms[idx0][ii])*(X[idx1-M+Mprev][ii]) for ii in range(len_waveforms))/(len_waveforms) <= R[idx0][idx1], label='Greater or equal_'+str(rr))


cqm_sampler = LeapHybridCQMSampler()
sampleset = cqm_sampler.sample_cqm(cqm, label='Waveform generation')
print(sampleset.info)
print(sampleset)

np.save('sampleset.npy',sampleset.record, allow_pickle=True)














