import numpy as np

import sys
fundamental_path = "/storage/mi/pycon/"
sys.path.append(fundamental_path + "TensorTrain/")
import TensorTrain2.Util as UT

''' This script reads the torsion angle data for VA peptide, and evaluates
the Fourier basis functions for all data points.'''

''' 1. Set paths:'''
# Path where trajectory data is stored:
trajpath = "/home/mi/vitalini/Systems/Dimers/ff_AMBER99SB_ILDN/Ac_VA_NHMe/analysis/"
# Path where function evaluations are stored:
datapath = fundamental_path + "TTApplications/VA/Evaluations/"


''' 2. Settings:'''
print "Loading trajectories."
# Number of trajectories:
ntraj = 4
# Highest frequency of Fourier basis for each dihedral:
nf = 2
# Set the number of coordinates:
d = 4

''' 3. Load the trajectories and evaluate the basis:'''
# Loop over the trajectories:
for m in range(ntraj):
    # Load the data for this trajectory:
    mdata1 = np.loadtxt(trajpath + "rep%d/torsion_V1.dat"%(m+1))
    mdata2 = np.loadtxt(trajpath + "rep%d/torsion_A2.dat"%(m+1))
    mdata = np.hstack((mdata1,mdata2))
    # Downsample the data:
    mdata = np.pi*(mdata[::50,:]/180.0)
    # Evaluate the functions for each coordinate:
    for n in range(d):
        fdata = UT.EvalFourier(mdata[:,n],nf)
        # Save to disc:
        np.save(datapath+"Traj%d/Basis%d.npy"%(m,n),fdata)
    print "Finished trajectory %d."%(m)