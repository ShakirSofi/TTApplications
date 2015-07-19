import numpy as np

import sys
sys.path.append("/storage/mi/pycon/TensorTrain/")
import TensorTrain2.Util as TTU

# Result directory:
resdir = "/storage/mi/pycon/TTApplications/VA/Results/"

# Load the eigenfunctions:
ntraj = 4
ev_traj = [resdir + "Eigenfunction_%d.npy"%m for m in range(ntraj)]

# Create a histogram:
bins = 80
m = np.array([1,2])
H = TTU.CreateEVHistogram(ev_traj,bins,m=m)

# ''' Write out frames to a trajectory file:'''
# Define centers:
# c = np.array([[-0.9],[1.2]])
# # Define allowed distance to the centers:
# d = np.array([0.2,0.2])
# # Define trajectories:
# trajpath = "/storage/mi/pycon/MR121_GS2W/Trajectories/"
# trajfiles = [trajpath + "MR121-GSGS-W_dt10ps_traj%d.xtc"%(m+1) for m in range(ntraj)]
# # Topology:
# topfile = trajpath + "structure.pdb"
# # Filename:
# filename = resdir + "EigenfunctionFrames"
# # Number of frames per traj and center:
# nframes = 250
# # Write out trajs:
# TTU.SaveEVFrames(trajfiles,ev_traj,c,d,filename,topfile,nframes)