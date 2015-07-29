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
resdir = fundamental_path + "TTApplications/BPTI_Contacts/ResultsReduced/"


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

''' Settings for experiments with least-squares error and contact difference:'''
# Different cut-offs based on least-squares:
cut_lsq = np.array([1e-6,1e-5,1e-4,1e-3,1e-2,1e-1])
# Different cut-offs based on contact difference:
cut_diff = np.array([0.2,0.3,0.4,0.5,0.6,0.7,0.8])
# Load the least-squares errors:
lsq = np.loadtxt(fundamental_path + "TTApplications/BPTI_Contacts/Results_eps995/LSQErrorNormalized.dat")
# Load the contact differences:
diff = np.loadtxt(fundamental_path + "TTApplications/BPTI_Contacts/Results_eps995/DifferenceSwitchingFunction.dat")
    
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
    
''' Run different reduced models based on least squares error:'''
# Get the number of cut-offs and prepare output:
ncut = cut_lsq.shape[0]
res_lsq = np.zeros((3,ncut))
print "Attempting reduced models for %d different cut-offs."%ncut

# Perform the optimizations for reduced models:
for j in range(ncut):
    jcut = cut_lsq[j]
    # Get the relevant coordinates:
    jd = np.where(lsq >= jcut)[0]
    # Get their number:
    njd = jd.shape[0]
    print "Using cut-off = %f, with %d coordinates:"%(jcut,njd)
    print jd
    # Reduce basis and U:
    jbasis = [basis[i] for i in jd]
    jU = U[:njd]
    # Create TT object and ALS object:       
    T = TT.BlockTTtensor(jU,jbasis,M,ifacedir)
    # Define ALS object:
    A = ALS.ALS(tau,dt,M,ifilename,rmax,tol,gtol=gtol)
    # Run Optimization
    T,A = ALM.RunALS(T,A)
    # Extract the results:
    jts = A.ts[0,-1]
    res_lsq[0,j] = jts
    res_lsq[1,j] = njd
    res_lsq[2,j] = jcut
    print "Results: t2 = %.5f"%jts
    print ""
     
print "Finished least-squares analysis, saving data."
np.savetxt(resdir + "ResultsLSQ.dat",res_lsq)

''' Run different reduced models based on contact difference:'''
# Get the number of cut-offs and prepare output:
ncut = cut_diff.shape[0]
res_diff = np.zeros((3,ncut))
print "Attempting reduced models for %d different cut-offs."%ncut

# Perform the optimizations for reduced models:
for j in range(ncut):
    jcut = cut_diff[j]
    # Get the relevant coordinates:
    jd = np.where(diff >= jcut)[0]
    # Get their number:
    njd = jd.shape[0]
    print "Using cut-off = %f, with %d coordinates:"%(jcut,njd)
    print jd
    # Reduce basis and U:
    jbasis = [basis[i] for i in jd]
    jU = U[:njd]
    # Create TT object and ALS object:       
    T = TT.BlockTTtensor(jU,jbasis,M,ifacedir)
    # Define ALS object:
    A = ALS.ALS(tau,dt,M,ifilename,rmax,tol,gtol=gtol)
    # Run Optimization
    T,A = ALM.RunALS(T,A)
    # Extract the results:
    jts = A.ts[0,-1]
    res_diff[0,j] = jts
    res_diff[1,j] = njd
    res_diff[2,j] = jcut
    print "Results: t2 = %.5f"%jts
    print ""
    
print "Finished contact difference analysis, saving data."
np.savetxt(resdir + "ResultsContactDiff.dat",res_diff)

