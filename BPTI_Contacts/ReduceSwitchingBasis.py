import numpy as np
import matplotlib.pyplot as plt

''' 1. Set paths:'''
fundamental_path = "/Users/fnueske/Documents/Uni/"
# Path where trajectory data is stored:
trajpath = fundamental_path + "TTApplications/BPTI_Contacts/SwitchingFunction/"


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
# Extract the required indices:
ind = np.where(diff >= cutoff)[0]

''' 4. Save the results:'''
# Save the indices that were selected:
np.savetxt(trajpath + "Indices.dat",ind)
# Reduce the trajectory and save it:
data = data[:,ind]
np.save(trajpath + "SwitchingFunctionReduced.npy",data)
