import numpy as np
import matplotlib.pyplot as plt

fundamental_path = "/Users/fnueske/Documents/Uni/"

# Directory for results:
resdir = fundamental_path + "TTApplications/BPTI_Contacts/Results_eps995/"
# Directory for figures:
figdir = fundamental_path + "TTApplications/BPTI_Contacts/Figure5/"

''' Timescale over iteration: '''
ts = np.loadtxt(resdir+"Timescales.dat")
d = 258
nsweeps = ts.shape[0]/(2*(d-2))
xp1 = np.arange(0,d-2)
xp2 = np.arange(d-1,1,-1)
markers = ["s","o","*",">"]
colors = ["b","r","g","k"]
sizes = [6,6,10,6]
plt.figure()
q = 0
for m in range(nsweeps):
    plt.plot(xp1[::6],ts[m*2*(d-2):(m+0.5)*2*(d-2):6],linestyle='None',
             marker = markers[q],markerfacecolor = colors[q],
             markersize=sizes[q])
    plt.plot(xp2[::6],ts[(m+0.5)*2*(d-2):(m+1)*2*(d-2):6],linestyle='None',
             marker = markers[q+1],markerfacecolor = colors[q+1],
             markersize=sizes[q+1])
    plt.plot([],[],linestyle='None',
             marker = markers[q],markerfacecolor = colors[q],
             markersize=sizes[q],label=" ")
    plt.plot([],[],linestyle='None',
             marker = markers[q+1],markerfacecolor = colors[q+1],
             markersize=sizes[q+1],label=" ")
    q += 2
fig = plt.gca()
fig.axes.get_xaxis().set_ticklabels([])
fig.axes.get_yaxis().set_ticklabels([])
plt.legend(loc=4,frameon=False)
plt.savefig(figdir+"Timescales.pdf")

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
plt.semilogy(xe,H,linestyle='None',marker="o",markersize=8)
fig = plt.gca()
fig.axes.get_xaxis().set_ticklabels([])
fig.axes.get_yaxis().set_ticklabels([])
plt.savefig(figdir + "Histogram.pdf")

''' Contact Differences and LSQ Errors:'''
# Load the contact differences::
diff = np.loadtxt(resdir + "DifferenceSwitchingFunction.dat")
# Create a normalized vector of the lsq errors:
lsq = np.loadtxt(resdir + "LSQErrorNormalized.dat")

plt.figure()

plt.subplot(2,1,1)
plt.bar(np.arange(d), diff)
fig = plt.gca()
fig.axes.get_xaxis().set_ticklabels([])
fig.axes.get_yaxis().set_ticklabels([])

plt.subplot(2,1,2)
plt.semilogy(np.arange(d), lsq,linestyle = 'None',marker='o',markersize=7)
fig = plt.gca()
fig.axes.get_xaxis().set_ticklabels([])
fig.axes.get_yaxis().set_ticklabels([])
plt.savefig(figdir + "Comparison.pdf")

''' Reduced Models from LSQ Errors and Contact Differences:'''
# Load the different timescale models:
ts_lsq = np.loadtxt(fundamental_path + "TTApplications/"+
                    "BPTI_Contacts/ResultsReduced/ResultsLSQ.dat")
ts_diff = np.loadtxt(fundamental_path + "TTApplications/"+
                    "BPTI_Contacts/ResultsReduced/ResultsContactDiff.dat")
plt.figure()
plt.subplot(2,1,1)
plt.plot(ts_diff[2,:],ts_diff[0,:],'b--',linewidth=3,marker="o",markersize=8)
#fig = plt.gca()
#fig.axes.get_xaxis().set_ticklabels([])
#fig.axes.get_yaxis().set_ticklabels([])

plt.subplot(2,1,2)
plt.semilogx(ts_lsq[2,:],ts_lsq[0,:],'b--',linewidth=3,marker="o",markersize=8)
#fig = plt.gca()
#fig.axes.get_xaxis().set_ticklabels([])
#fig.axes.get_yaxis().set_ticklabels([])


plt.show()