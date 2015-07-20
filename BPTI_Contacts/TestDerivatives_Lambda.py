import numpy as np
import matplotlib.pyplot as plt
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
    Ctau[iu] = c[:n/2]
    Ctau = Ctau + Ctau.T - np.diag(np.diag(Ctau))
    C0 = np.zeros((sp*tp,sp*tp))
    C0[iu] = c[n/2:]
    C0 = C0 + C0.T - np.diag(np.diag(C0))
    D,_ = pla.eig_corr(C0, Ctau)
    D = D[:M]
    return -np.sum(D)

# Load the test case:
sp = 4
tp = 2
R = 2
M = 2
Ctau = np.load("TestCtau.npy")
C0 = np.load("TestC0.npy")

D, X = pla.eig_corr(C0, Ctau)
D = D[:M]
X = X[:,:M]

# Check the perturbation theory:
eps_array = 1e-7*np.array([2,1,0.8,0.6,0.4,0.2,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001,0.00005,0.00001])
lambdas = np.zeros(eps_array.shape[0])
lambdasp = np.zeros(eps_array.shape[0])
q = 0
iu = np.triu_indices(Ctau.shape[0])
for eps in eps_array:
    # Create  perturbation for Ctau:
    pe1 = eps*np.random.rand(iu[0].shape[0])
    Ctaue = np.zeros(Ctau.shape)
    Ctaue[iu] = pe1
    Ctaue = Ctaue + Ctaue.T -np.diag(np.diag(Ctaue))
    # The same for C0:
    pe2 = eps*np.random.rand(iu[0].shape[0])
    C0e = np.zeros(C0.shape)
    C0e[iu] = pe2
    C0e = C0e + C0e.T -np.diag(np.diag(C0e))
    # Join the perturbations:
    pe = np.hstack((pe1,pe2))
    # Solve the perturbed problem:
    De,_ = pla.eig_corr(C0+C0e,Ctau+Ctaue)
    lambdas[q] = -np.sum(De[:M])
    # Compare to perturbation theory:
    grad = TOP.GradCMatrix(X, D, sp, tp)
    lambdasp[q] = -np.sum(D) + np.dot(grad.flatten(),pe)
    q += 1

# Test numerical differentiation tool:
ctau = Ctau[iu].copy()
c0 = C0[iu].copy()
c = np.hstack((ctau,c0))
f = ft.partial(TestObjective,sp=sp,tp=tp,M=M)
G = nd.Gradient(f)
gc = G(c)
print "Agreement between perturbation theory and numerical diff.:"
print grad
diff =  np.abs((G(c) - grad))
print diff

plt.loglog(eps_array,np.abs(lambdas - lambdasp),label='Absolute Error')
plt.legend()
plt.show()
