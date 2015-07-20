import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("/storage/mi/pycon/TensorTrain/")
import TensorTrain2.Util as TTU

# Result directory:
resdir = "/storage/mi/pycon/TTApplications/BPTI_Contacts/Results_eps99CG/"

''' Implied Timescales:'''
ts = np.loadtxt(resdir+"Timescales.dat")
d = 258
nsweeps = ts.shape[0]/(2*(d-2))
xp1 = np.arange(1,d-1)
xp2 = np.arange(d-2,0,-1)
plt.figure()
for m in range(nsweeps):
    plt.plot(xp1,ts[m*2*(d-2):(m+0.5)*2*(d-2)],label="Sweep%dF"%m)
    plt.plot(xp2,ts[(m+0.5)*2*(d-2):(m+1)*2*(d-2)],label="Sweep%dB"%m)
plt.legend(loc=4)
plt.savefig(resdir+"Timescales.pdf")
plt.show()


''' Histogram:'''
# Physical time step:
dt = 100
# Filename:
filename = resdir + "Histogram.pdf"
# Load the eigenfunctions:
ntraj = 1
ev_traj = [resdir + "Eigenfunction_%d.npy"%m for m in range(ntraj)]

# Create a histogram:
bins = 150
m = np.array([1])
H = TTU.CreateEVHistogram(ev_traj,bins,filename,m=m)

''' Write out frames to a trajectory file:'''
#Define centers:
c = np.array([[-0.2],[6.6]])
# Define allowed distance to the centers:
di = np.array([0.5,0.5])
# Define trajectories:
trajpath = "/storage/mi/pycon/BPTI_CA/"
trajfiles = [trajpath + "all.xtc"]
# Topology:
topfile = trajpath + "bpti_ca.pdb"
# Filename:
filename = resdir + "EigenfunctionFrames"
# Number of frames per traj and center:
nframes = 200
# Write out trajs:
TTU.SaveEVFrames(trajfiles,dt,ev_traj,c,di,filename,topfile,nframes)