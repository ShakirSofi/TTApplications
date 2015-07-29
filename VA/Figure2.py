import numpy as np
import matplotlib.pyplot as plt

# Set the paths:
fundamental_path = "/storage/mi/pycon/"
resdir = fundamental_path + "TTApplications/VA/ResultsFull/"
# Load the timescales:
ts_full = np.loadtxt(resdir + "Timescales.dat")
ts_msm = np.loadtxt(resdir + "Timescales_MSM.dat")
ts_msm = 0.001*ts_msm
taulist1 = ts_full[:,0]
taulist2 = ts_msm[:,0]

''' Timescale plot:'''
plt.figure()
plt.plot(taulist1,ts_full[:,2],"b--*",linewidth = 3,markersize = 12)
plt.plot(taulist2,ts_msm[:,2],"b--o",linewidth = 3,markersize = 6)

plt.plot(taulist1,ts_full[:,1],"r--*",linewidth = 3,markersize=12)
plt.plot(taulist2,ts_msm[:,1],"r--o",linewidth = 3,markersize = 6)

plt.plot([],[],"k--*",linewidth=3,markersize=12,label=" ")
plt.plot([],[],"k--o",linewidth=3,markersize=8,label=" ")
plt.ylim([0,10])
frame1 = plt.gca()
frame1.axes.get_xaxis().set_ticklabels([])
frame1.axes.get_yaxis().set_ticklabels([])
plt.legend(loc=2,frameon=False)

plt.savefig(resdir + "Timescales_VA.pdf")

''' Eigenvector coefficients:'''
VW = np.loadtxt(resdir + "Eigenvectors_Transformed.dat")
v0 = np.argsort(np.abs(VW[:,1]))[::-1]
V0 = VW[v0,1]
V0 = np.cumsum(V0**2)
plt.figure()
plt.plot(V0,"r--s",linewidth=2.5)
plt.ylim([0,1.1])
frame1 = plt.gca()
frame1.axes.get_xaxis().set_ticklabels([])
frame1.axes.get_yaxis().set_ticklabels([])
plt.savefig(resdir + "Coefficients_VA.pdf")

plt.show()
