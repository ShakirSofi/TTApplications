import numpy as np

import sys
sys.path.append("/storage/mi/pycon/TensorTrain/")
import TensorTrain2.Util as UT

''' This script reads the torsion angle data for VA peptide, and evaluates
the Fourier basis functions for all data points.'''

''' 1. Set paths:'''
# Path where trajectory data is stored:
trajpath = "/storage/mi/pycon/DecaAlanine/Amber03/"
# Path where function evaluations are stored:
datapath = "/storage/mi/pycon/TTApplications/ALA10TT2/Evaluations/"


''' 2. Settings:'''
print "Loading trajectories."
# Number of trajectories:
ntraj = 6
# Highest frequency of Fourier basis for each dihedral:
nf = 3
# Set the number of coordinates:
d = 16

''' 3. Load the trajectories and evaluate the basis:'''
# Loop over the trajectories:
for m in range(ntraj):
    # Load the data for this trajectory:
    mdata = np.load(trajpath + "DihedralTimeSeries_%d.npy"%m)
    # Downsample the data:
    mdata = mdata[::50,:]
    # Evaluate the functions for each coordinate:
    for n in range(d):
        fdata = UT.EvalFourier(mdata[:,n],nf)
        # Save to disc:
        np.save(datapath+"Traj%d/Basis%d.npy"%(m,n),fdata)
    print "Finished trajectory %d."%(m)