import numpy as np

import pyemma.coordinates as pco

import sys
fundamental_path = "/storage/mi/pycon/"
sys.path.append(fundamental_path + "TensorTrain/")
import TensorTrain2.TTtensors as TT
import TensorTrain2.ALSClass as ALS
import TensorTrain2.ALSAlgo as ALM
import TensorTrain2.Util as UT

''' 1. General definitions:'''
# Dimension:
d = 258
 
''' 2. Basis functions and directories:'''
print "Preparing data:"
# Path of basis evaluations:
basispath = fundamental_path + "TTApplications/BPTI_Contacts/Evaluations/"
# List for basis readers:
basis = []
for i in range(d):
    # Create list of evaluation files for this coordinate:
    file_list = [basispath+"Basis%d.npy"%(i)]
    # Create a reader for this basis:
    ireader = pco.source(file_list,chunk_size=100000)
    # Append it:
    basis.append(ireader)

# Define a directory for intermediate files, interfaces, and results:
ifacedir = fundamental_path + "TTApplications/BPTI_Contacts/Interfaces/"
ifilename = fundamental_path + "TTApplications/BPTI_Contacts/Intermediate/"
resdir = fundamental_path + "TTApplications/BPTI_Contacts/Results_eps995/"


''' 3. Computational Settings:'''
# Lag time (5 microseconds):
tau = 200
# Physical time step:
dt = 0.025
# Number of eigenfunctions:
M = 2
# Maximum rank:
rmax = 20
# Tolerance:
tol = 0.995
# Gradient tolerance:
gtol = 1e-4

    
''' 4. Define TT and ALS objects:'''
# Initialise Components:
U = []
for i in range(d):
    if i == 0:
        iU = np.zeros((1,basis[i].dimension(),1,M))
    else:
        iU = np.zeros((1,basis[i].dimension(),1))
        iU[0,0,0] = 1
    U.append(iU)
print "Create TT-tensor."
# Create TT-object:        
T = TT.BlockTTtensor(U,basis,M,ifacedir)
# Create ALS object:
A = ALS.ALS(tau,dt,M,ifilename,rmax,tol,gtol=gtol)
 
''' 5. Run Optimization:'''
T,A = ALM.RunALS(T,A)
 
''' 6. Save the results: '''
print "Save results."
# Extract the ranks:
ranks = T.R
np.savetxt(resdir+"Ranks.dat",ranks)
# Extract the objective values and timescales:
J_arr = A.J
ts = A.ts
np.savetxt(resdir+"Objectives.dat",J_arr)
np.savetxt(resdir+"Timescales.dat",ts)
# Extract the least-squares errors and save them:
np.savetxt(resdir+"LSQErrors.dat",T.lse)
# Compute the eigenfunctions:
Ef = UT.EvalEigenfunctions(T,tau,resdir+"Eigenfunction")
