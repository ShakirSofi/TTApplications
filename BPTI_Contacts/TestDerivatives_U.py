import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as scl
import numdifftools as nd
import functools as ft
import pyemma.util.linalg as pla

import sys
sys.path.append("/Users/fnueske/Documents/Uni/TensorTrain/")
import TensorTrain2.Optimize as TOP

# Objective function:
def TestObjective(u,Ctau,C0,R,tp):
    # Reshape u:
    U = np.reshape(u,(sp,R-1))
    # Add a column encoding the constant:
    U = np.hstack((np.eye(sp,1),U))
    # Compute contracted correlation matrices:
    Ctaup = np.einsum('ij,iklm,ln->jknm',U,Ctau,U)
    C0p = np.einsum('ij,iklm,ln->jknm',U,C0,U)
    # Remove doubled variables:
    iu = np.triu_indices(R*tp)
    Ctaup = np.reshape(Ctaup,(R*tp,R*tp))
    Ctaup = Ctaup[iu]
    C0p = np.reshape(C0p,(R*tp,R*tp))
    C0p = C0p[iu]
    C = np.hstack((Ctaup,C0p))
    return C

# Set parameters:
sp = 4
tp = 2
R = 3
M = 2
# Load the test matrices:
Ctau = np.load("TestCtau.npy")
C0 = np.load("TestC0.npy")
# Solve the full problem:
D, X = pla.eig_corr(C0, Ctau)
D = D[:M]
X = X[:,:M]

# Create a U for testing:
X2 = np.reshape(X,(sp,tp*M)).copy()
U,_,_ = scl.svd(X2,full_matrices=False)
U = U[:,1:R].copy()
u = U.flatten()
# Run the function:
Ctaup = np.reshape(Ctau,(sp,tp,sp,tp))
C0p = np.reshape(C0,(sp,tp,sp,tp))
C = TestObjective(u,Ctaup,C0p,R,tp)

# Compute the jacobian numerically:
f = ft.partial(TestObjective,Ctau=Ctaup,C0=C0p,R=R,tp=tp)
Jd = nd.Jacobian(f)
Ju = Jd(u)

# Compare to our result:
U2 = np.hstack((np.eye(sp,1),U)) 
J = TOP.Jacobian(Ctaup,C0p,U2)

print np.max(np.abs(J - Ju),axis=1)