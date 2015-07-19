import numpy as np

import sys
sys.path.append("/storage/mi/pycon/TensorTrain/")
import TensorTrain2.Util as TTU

# Result directory:
resdir = "/storage/mi/pycon/TTApplications/ALA10TT2/Results/"

# Physical time step:
dt = 50

# Load the eigenfunctions:
ntraj = 6
ev_traj = [resdir + "Eigenfunction_%d.npy"%m for m in range(ntraj)]

# Create a histogram:
bins = 80
H = TTU.CreateEVHistogram(ev_traj,bins)

''' Write out frames to a trajectory file:'''
#Define centers:
c = np.array([[-1.1],[-0.2],[0.9],[1.5]])
# Define allowed distance to the centers:
d = np.array([0.4,0.1,0.1,0.1])
# Define trajectories:
trajpath = "/group/ag_cmb/ppxasjsm/deca-alanine-ffamber03/production/"
trajfiles = [trajpath + "run%d/noPBC.xtc"%m for m in range(ntraj)]
# Topology:
topfile = trajpath + "run0/md_production0_noWater.gro"
# Filename:
filename = resdir + "EigenfunctionFrames"
# Number of frames per traj and center:
nframes = 100
# Write out trajs:
TTU.SaveEVFrames(trajfiles,dt,ev_traj,c,d,filename,topfile,nframes)