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

nev = 3

Tref,Sref,Qref,Zref = scl.qz(Ct,C0)
alpha_ref = np.diag(Tref)
beta_ref = np.diag(Sref)
dref = alpha_ref/beta_ref

ntry = 20
Qarr = np.zeros((ntry,N,N))
Zarr = np.zeros((ntry,N,N))
tarr = np.zeros((ntry,N))
sarr = np.zeros((ntry,N))

Varr = np.zeros((ntry,N,nev))

for i in range(ntry):
    T,S,Q,Z = scl.qz(Ct,C0)
    Qarr[i,:,:] = Q.copy()
    Zarr[i,:,:] = Z.copy()
    tarr[i,:] = np.diag(T).copy()
    sarr[i,:] = np.diag(S).copy()
    # Sort the eigenvalues:
    d = np.diag(T)/np.diag(S)
    ind = np.argsort(d)[::-1]
    ds = d[ind]
    for n in range(nev):
        Bn = T-d[n]*S
        if not(np.linalg.matrix_rank(Bn) == N-1):
            print "Degenerate eigenvalue detected."
        else:
            _,_,Wn = scl.svd(Bn)
            xn = Wn[-1,:]
            Varr[i,:,n] = np.dot(Z,xn)
            
    
    
print np.max(np.max(Qarr,axis=0) - np.min(Qarr,axis=0))
print np.max(np.max(Zarr,axis=0) - np.min(Zarr,axis=0))
print np.max(np.max(tarr,axis=0) - np.min(tarr,axis=0))
print np.max(np.max(sarr,axis=0) - np.min(sarr,axis=0))
print np.max(np.max(Varr,axis=0) - np.min(Varr,axis=0))