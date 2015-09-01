import numpy as np
import matplotlib.pyplot as plt

import sys
fundamental_path = "/storage/mi/pycon/"
sys.path.append(fundamental_path + "TensorTrain/")
import TensorTrain2.Util as UT

# Directory for results:
resdir = fundamental_path + "TTApplications/BPTI_Contacts/ResultsCG_QR/"
# Directory for figures:
figdir = fundamental_path + "TTApplications/BPTI_Contacts/Figure5/"

''' Timescale over iteration: '''
ts = np.loadtxt(resdir+"Timescales.dat")
d = 258
nsweeps = ts.shape[0]/(2*(d-2))
xp1 = np.arange(0,d-2)
xp2 = np.arange(d-1,1,-1)
markers = ["s", "o", "*", ">", "^", "<"]
colors = ["b", "r", "g", "k", "m", "c"]
sizes = [6, 6, 10, 6, 6, 6]
plt.figure()
q = 0
for m in range(nsweeps):
    plt.plot(xp1[::6], ts[m*2*(d-2):(m+0.5)*2*(d-2):6], linestyle='None',
             marker = markers[q], markerfacecolor = colors[q],
             markersize=sizes[q])
    plt.plot(xp2[::6], ts[(m+0.5)*2*(d-2):(m+1)*2*(d-2):6], linestyle='None',
             marker = markers[q+1], markerfacecolor = colors[q+1],
             markersize=sizes[q+1])
    plt.plot([],[], linestyle='None',
             marker = markers[q], markerfacecolor = colors[q],
             markersize=sizes[q], label=" ")
    plt.plot([],[],linestyle='None',
             marker = markers[q+1], markerfacecolor = colors[q+1],
             markersize=sizes[q+1], label=" ")
    q += 2
plt.legend(loc=4,frameon=False)

''' Histogram of eigenfunction.'''
ev_traj = np.load(resdir + "Eigenfunction_0.npy")
# Create a histogram:
bins = 80
H,xe = np.histogram(ev_traj[:,1],bins)
# Normalize the histogram and get the bin centers:
H = 1.0*H/np.sum(H)
xe = 0.5*(xe[:-1] + xe[1:])
# Plot a log-scale histogram:
plt.figure()
plt.semilogy(xe, H, linestyle='None', marker="o", markersize=8)


''' Extract frames and show the contacts for these frames:'''
# Define centers:
c = np.array([[-0.2],[6.5]])
# Define allowed distance to the centers:
di = np.array([0.5, 1])
# Number of frames per traj and center:
nframes = 200
# Trajectory files:
traj = [fundamental_path + "BPTI_CA/all.xtc"]
# Topology:
topfile = fundamental_path + "BPTI_CA/bpti_ca.pdb"
# Filename:
filename = resdir + "BPTI_QR"
# Time-step:
dt = 100
# Write out trajs:
UT.SaveEVFrames(dt, [ev_traj], c, di, traj, filename, topfile, nframes=nframes)

''' Create normalized least-squares error:'''
lsq = np.loadtxt(resdir + "LSQErrors.dat")
lsq2 = np.zeros(d)
lsq2[1:d-1] = lsq/np.max(lsq)
np.savetxt(resdir + "LSQErrorNormalized.dat",lsq2)