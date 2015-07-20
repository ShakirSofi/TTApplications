import numpy as np


''' 1. Set paths:'''
# Path where trajectory data is stored:
trajpath = "/storage/mi/pycon/TTApplications/BPTI_CA/SwitchingFunction/"
# Path where function evaluations are stored:
datapath = "/storage/mi/pycon/TTApplications/BPTI_Contacts/Evaluations/"


''' 2. Settings:'''
print "Loading trajectories."
# Number of angles:
d = 1540
# Cutoff for taking a contact into account:
cutoff = 9e-1


''' 3. Load the trajectories and evaluate the basis:'''
# Load the data for this dihedral:
data = np.load(trajpath + "SwitchingFunctionCA_BPTI.npy")
# Compute the difference between max and min of each basis functions:
diff = np.max(data,axis=0) - np.min(data,axis=0)
# Get the number of timesteps:
T = data.shape[0]
# Evaluate the basis and save to file if the contact ever changes:
q = 0
for n in range(d):
    # Get the next contact:
    ndata = data[:,n]
    # Check if it meets the condition:
    if diff[n] > cutoff:
        fdata = np.hstack((np.ones((T,1)),ndata[:,None]))
        # Save to disc:
        print "Saving Trajectory %d."%n
        np.save(datapath+"Basis%d.npy"%(q),fdata)
        q += 1