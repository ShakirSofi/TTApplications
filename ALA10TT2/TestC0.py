import numpy as np
import scipy.linalg as scl
import matplotlib.pyplot as plt


def correct_scaling(V,Vref):
    N = Vref.shape[1]
    for j in range(N):
        Vij = np.dot(V[:,j],Vref[:,j])
        if Vij < 0:
            V[:,j] *= -1
    return V

C0 = np.loadtxt("C0_1439900011.740475.dat")
Ct = np.loadtxt("Ct_1439900011.743895.dat")
N = C0.shape[0]

ep = 1e-6

nev = 10

ntry = 10000
sarr = np.zeros((ntry-1,N))
darr = np.zeros((ntry-1,nev))

for i in range(ntry):
    
    S,V = scl.schur(C0)
    s = np.diag(S)
    
    ind = np.argsort(s)[::-1]
    s = s[ind]
    V = V[:,ind]
    
    if i > 0:
        V = correct_scaling(V,Vref)
    evmin = np.min(s)
    if evmin < 0:
        epsilon = max(ep, -evmin + 1e-16)
    else:
        epsilon = ep

    # determine effective rank m and perform low-rank approximations.
    evnorms = np.abs(s)
    n = np.shape(evnorms)[0]
    m = n - np.searchsorted(evnorms[::-1], epsilon)
    if m > 0:
        Vm = V[:, 0:m]
        sm = s[0:m]                      
        # transform Ct to orthogonal basis given by the eigenvectors of C0
        Sinvhalf = 1.0 / np.sqrt(sm)
        T = np.dot(np.diag(Sinvhalf), Vm.T)
        Ct_trans = np.dot(np.dot(T, Ct), T.T)
        d,W = scl.eigh(Ct_trans)
        
        ind = np.argsort(d)[::-1]
        d = d[ind]
        W = W[:,ind]
        
        if i == 0:
            sref = s.copy()
            Vref = V.copy()
            Ct_ref = Ct_trans.copy()
            M = Ct_ref.shape[0]
            Wref = W.copy()
            dref = d.copy()
            Ctarr = np.zeros((ntry-1,M,M))
            Warr = np.zeros((ntry-1,M,nev))
        else:
            sarr[i-1,:] = s.copy()
            Ctarr[i-1,:,:] = Ct_trans.copy()
            darr[i-1,:] = d[:nev].copy()
            W = correct_scaling(W,Wref)
            Warr[i-1,:,:] = W[:,:nev].copy()

Ct_diff = np.max(Ctarr,axis=0) - np.min(Ctarr,axis=0)
d_diff = np.max(darr,axis=0) - np.min(darr,axis=0)
W_diff = np.max(Warr,axis=0) - np.min(Warr,axis=0)

print np.max(d_diff)
print np.max(W_diff)