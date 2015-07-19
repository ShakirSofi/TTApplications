import numpy as np
import scipy.linalg as scl
import numdifftools as nd
import functools as ft

import sys
sys.path.append("/Users/fnueske/Documents/Uni/TensorTrain/")
import TensorTrain2.Optimize as TOP

import pyemma.util.linalg as pla

def TestObjective(c,sp,tp,M):
    n = c.shape[0]
    iu = np.triu_indices(sp*tp)
    Ctau = np.zeros((sp*tp,sp*tp))
    Ctau[iu] = c[n/2:]
    Ctau = Ctau + Ctau.T - np.diag(np.diag(Ctau))
    C0 = np.zeros((sp*tp,sp*tp))
    C0[iu] = c[:n/2]
    C0 = C0 + C0.T - np.diag(np.diag(C0))
    D,_ = pla.eig_corr(C0, Ctau)
    D = D[:M]
    return -np.sum(D)

# Load the test case:
sp = 7
tp = 7
R = 2
M = 3
Ctau = np.load("TestCtau.npy")
C0 = np.load("TestC0.npy")

D, X = pla.eig_corr(C0, Ctau)
D = D[:M]
X = X[:,:M]
# Reshape Upp and perform SVD:
#Upp = np.reshape(Upp,(sp,tp*M))
#V,_,_ = scl.svd(Upp)
#Ctau = np.reshape(Ctau,(sp,tp,sp,tp))
#C0 = np.reshape(C0,(sp,tp,sp,tp))

#Up = (V[:,1:R]).copy()
#u = np.reshape(Up,(sp*(R-1)))

# Test derivatives of the eigenvalues:
grad = TOP.GradCMatrix(X, D, sp, tp)
print grad
iu = np.triu_indices(sp*tp)
ctau = Ctau[iu].copy()
c0 = C0[iu].copy()
c = np.hstack((c0,ctau))
print TestObjective(c, sp, tp, M)


# Compute gradient by numdifftools:
f = ft.partial(TestObjective,sp=sp,tp=tp,M=M)
G = nd.Gradient(f)
print f(c)
print G(c)