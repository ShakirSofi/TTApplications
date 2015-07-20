import numpy as np
import matplotlib.pyplot as plt

# Result directory:
resdir = "/storage/mi/pycon/TTApplications/BPTI_Contacts/Results_eps99Trials/"

# Set the number of trials:
ntrials = 20
# Set the dimension:
d = 258

# Produce a plot of the number of runs and of the final timescale:
runs = []
ts = []
variables = []
for i in range(ntrials):
    its = np.loadtxt(resdir+"Run%d/Timescales.dat"%i)
    # Check if this run terminated:
    if np.remainder(its.shape[0],(2*(d-2)))==0:
        variables.append(i)
        runs.append(its.shape[0]/(2*(d-2)))
        ts.append(its[-1])

variables = np.array(variables)
ntrials = variables.shape[0]
runs = np.array(runs)
ts = np.array(ts)
 
plt.figure()
plt.plot(variables,runs)
plt.savefig(resdir+"NumberOfRuns.pdf")
plt.figure()
plt.plot(variables,ts)
plt.savefig(resdir+"FinalTimescale.pdf")


# Produce a plot of the ranks:
ranks = np.zeros((d-1,ntrials))
plt.figure()
for i in range(ntrials):
    # Load the file and fill in:
    ranks[:,i] = np.loadtxt(resdir + "Run%d/Ranks.dat"%variables[i])
    plt.plot(ranks[:,i],label="%d"%variables[i])
plt.legend()
plt.savefig(resdir+"Ranks.pdf")

# Make a full timescale plot:
plt.figure()
for i in range(ntrials):
    its = np.loadtxt(resdir+"Run%d/Timescales.dat"%variables[i])
    plt.plot(its,label="%d"%variables[i])
plt.legend()
plt.savefig(resdir+"FullTimescales.pdf")

# Make a retry-plot:
plt.figure()
for i in range(ntrials):
    ret = np.loadtxt(resdir+"Run%d/Retries.dat"%variables[i])
    plt.plot(ret,label="%d"%variables[i])
plt.legend()
plt.savefig(resdir+"NumberOfRetries.pdf")

plt.show()