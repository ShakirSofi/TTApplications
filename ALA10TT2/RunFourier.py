import numpy as np

import pyemma.coordinates as pco

import sys
fundamental_path = "/Users/fnueske/Documents/Uni/"
sys.path.append(fundamental_path + "TensorTrain/")
import TensorTrain2.TTtensors as TT
import TensorTrain2.ALSClass as ALS
import TensorTrain2.ALSAlgo as ALM
import TensorTrain2.Util as UT

''' 1. General definitions:'''
# Dimension:
d = 16

''' 2. Basis functions and directories:'''
print "Preparing data:"
# Path of basis evaluations:
basispath = fundamental_path + "TTApplications/ALA10TT2/Evaluations/"
# Number of trajectories:
ntraj = 6
# List for basis readers:
basis = []
for i in range(d):
    # Create list of evaluation files for this coordinate:
    file_list = [basispath+"Traj%d/Basis%d.npy"%(j,i) for j in range(ntraj)]
    # Create a reader for this basis:
    ireader = pco.source(file_list,chunk_size=50000)
    # Append it:
    basis.append(ireader)
    
# Define a directory for intermediate files, interfaces, and results:
ifacedir = fundamental_path + "TTApplications/ALA10TT2/Interfaces/"
ifilename = fundamental_path + "TTApplications/ALA10TT2/Intermediate/Intermediate"
resdir = fundamental_path + "TTApplications/ALA10TT2/ResultsCG/"

''' 3. Computational Settings:'''
# Lag time:
tau = 40
# Physical time step:
dt = 0.05
# Number of eigenfunctions:
M = 2
# Maximum rank:
rmax = 10
# Tolerance:
tol = 0.995
# Set the CG-tolerance:
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