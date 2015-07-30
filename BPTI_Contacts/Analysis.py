import numpy as np
import matplotlib.pyplot as plt

import sys
fundamental_path = "/storage/mi/pycon/"
sys.path.append(fundamental_path + "TensorTrain/")
import TensorTrain2.Util as TTU

# Result directory:
resdir = fundamental_path + "TTApplications/BPTI_Contacts/Results_eps995/"
datadir = fundamental_path + "TTApplications/BPTI_Contacts/SwitchingFunction/"

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

''' Least-Squares-Error:'''
lsq = np.loadtxt(resdir + "LSQErrors.dat")
plt.figure()
plt.plot(lsq)
plt.savefig(resdir+"LSQErrors.pdf")
plt.show()

''' Histogram:'''
# Physical time step:
dt = 1
# Filename:
filename = resdir + "Histogram.pdf"
# Load the eigenfunctions:
ntraj = 1
ev_traj = [resdir + "Eigenfunction_%d.npy"%m for m in range(ntraj)]

# Create a histogram:
bins = 150
m = np.array([1])
H = TTU.CreateEVHistogram(ev_traj,bins,filename,m=m)

''' Extract frames and show the contacts for these frames:'''
# Define centers:
c = np.array([[-6.5],[0.2]])
# Define allowed distance to the centers:
di = np.array([1,0.5])
# Number of frames per traj and center:
nframes = 200
# Trajectory files:
traj = [fundamental_path + "BPTI_CA/all.xtc"]
# Topology:
topfile = fundamental_path + "BPTI_CA/bpti_ca.pdb"
# Filename:
filename = resdir + "BPTI_995"
# Write out trajs:
indices = TTU.SaveEVFrames(dt,ev_traj,c,di,traj,filename,topfile,nframes=nframes)
  
''' Check agreement between least-squares error and contact.'''
  
# Load the reduced basis trajectory:
data = np.load(datadir + "SwitchingFunctionReduced.npy")
#plt.figure()
# Get the number of centers:
nc = len(indices)
diff = np.zeros((2,d))
for i in range(nc):
    ilist = indices[i][0][:,1]
    # Get the corresponding timesteps from the function trajectory:
    idata = data[ilist,:]
    # Compute the average value of each switching function:
    diff[i,:] = np.mean(idata,axis=0)
     
# Compute the difference between the averages in absolute value:
diff = np.abs(diff[1,:]-diff[0,:])
# Save them to file:
np.savetxt(resdir + "DifferenceSwitchingFunction.dat",diff)
# Create a normalized vector of the lsq errors:
lsq2 = np.zeros(d)
lsq2[1:d-1] = lsq/np.max(lsq)
np.savetxt(resdir + "LSQErrorNormalized.dat",lsq2)
 
plt.figure()
 
plt.subplot(2,1,1)
plt.bar(np.arange(d), diff)
 
plt.subplot(2,1,2)
plt.semilogy(np.arange(d), lsq2,linestyle = 'None',marker='o',markersize=6)
plt.savefig(resdir + "Comparison.pdf")
plt.show() 