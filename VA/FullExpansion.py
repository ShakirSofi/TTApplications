import numpy as np
import scipy.linalg as scl


import pyemma.coordinates as pco

# Pathname:
fundamental_path = "/storage/mi/pycon/"
import sys
sys.path.append(fundamental_path + "TensorTrain/")
import TensorTrain2.Util as UT

''' Settings:'''
# Number of coordinates:
d = 4
# Path to store everything:
resdir = fundamental_path + "TTApplications/VA/ResultsFull/"

''' Computational Settings:'''
# Lag time:
tau = 40
# Physical time step:
dt = 0.05
# Number of eigenfunctions:
M = 3
# List of lag times for timescale test:
tau_list = np.array([5,10,20,30,40,50,60,70,80])
#tau_list = np.array([40])


''' Load the complete data'''
print "Preparing data:"
# Path of basis evaluations:
basispath = fundamental_path + "TTApplications/VA/Evaluations/"
# Number of trajectories:
ntraj = 4
# List for basis readers:
basis = []
for i in range(d):
    # Create list of evaluation files for this coordinate:
    file_list = [basispath+"Traj%d/Basis%d.npy"%(j,i) for j in range(ntraj)]
    # Create a reader for this basis:
    ireader = pco.source(file_list,chunk_size=1000000)
    # Append it:
    basis.append(ireader)
    

''' Compute full products:'''
# Filename for intermediate storage:
productdir = fundamental_path + "TTApplications/VA/FullProducts/"
productreader = basis[0]
# Evaluate the products:
for i in range(1,d):
    productreader = UT.DoubleProducts(productreader,basis[i],productdir+"Product")
print "Evaluated the products."

''' Solve the optimization problem:'''
ntau = np.shape(tau_list)[0]
ts = np.zeros((ntau,M))
ts[:,0] = dt*tau_list
for i in range(ntau):
    # Solve the problem for the next tau:
    itau = tau_list[i]
    eigv = UT.Diagonalize(productreader,itau,M)
    # Get the eigenvalues:
    D = eigv.eigenvalues
    # Compute implied timescales:
    ts[i,1:] = -dt*itau/np.log(D[1:])
    print "Finished lag time %f"%itau
    # Check if this is the lag time for further analysis:
    if itau == tau:
        # Get the C0-matrix:
        C0 = eigv.cov
        # Diagonalize it:
        S,U = scl.eigh(C0)
        ind = np.where(S>=1e-12)[0]
        S = S[ind]
        U = U[:,ind]
        # Compute orthogonal transformation:
        Sq = np.diag(np.sqrt(1.0/S))
        Uorth = np.dot(U,Sq)
        # Transform the eigenvectors:
        V = eigv.eigenvectors
        pU = scl.pinv(Uorth)
        V = np.dot(pU,V)
        np.savetxt(resdir + "Eigenvectors_Transformed.dat",V)

''' Save Results:'''
print "Saving data."
np.savetxt(resdir + "Timescales.dat",ts)