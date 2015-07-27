import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import sys
fundamental_path = "/Users/fnueske/Documents/Uni/"
sys.path.append(fundamental_path + "TensorTrain/")
import TensorTrain2.Util as TTU

# Result directory:
resdir = fundamental_path + "TTApplications/BPTI_Contacts/Results_eps99/"
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
#plt.show()

''' Least-Squares-Error:'''
lsq = np.loadtxt(resdir + "LSQErrors.dat")
plt.figure()
plt.plot(lsq)
plt.savefig(resdir+"LSQErrors.pdf")
#plt.show()

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
#H = TTU.CreateEVHistogram(ev_traj,bins,filename,m=m)

''' Extract frames and show the contacts for these frames:'''
# Define centers:
c = np.array([[-0.2],[6.5]])
# Define allowed distance to the centers:
di = np.array([0.5,1])
# Number of frames per traj and center:
nframes = None
# Write out trajs:
indices = TTU.SaveEVFrames(dt,ev_traj,c,di,nframes=nframes)
 
''' Check agreement between least-squares error and contact.'''
 
# Load the reduced basis trajectory:
data = np.load(datadir + "SwitchingFunctionReduced.npy")
#plt.figure()
# Get the number of centers:
nc = len(indices)
npix = 40
diff = np.zeros((3*npix,d))
for i in range(nc):
    ilist = indices[i][0][:,1]
    # Get the corresponding timesteps from the function trajectory:
    idata = data[ilist,:]
    # Compute the average value of each switching function:
    diff[i*npix:(i+1)*npix,:] = np.mean(idata,axis=0)
indlsq = np.where(lsq >= 0.01)[0]
diff[2:npix:3*npix,indlsq] = 1.0

plt.figure()
plt.imshow(diff,cmap = cm.Greys_r)
plt.show() 