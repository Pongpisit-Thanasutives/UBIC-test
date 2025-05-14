import numpy as np
from scipy.special import loggamma

def log_evidence(X_full, y, effective_indices, v=0.5, k=3, standardize=False):
    N = len(y)
    p = len(effective_indices)

    if standardize:
        k = 1 + 1/v
        X_full = (X_full-X_full.mean(axis=0))/X_full.std(axis=0)
        y = (y-y.mean())/y.std()

    K = X_full[:, effective_indices]

    KT = K.T # intermediate var
    yT = y.T # intermediate var
    KTy = KT@y # intermediate var
    yTy = yT@y # intermediate var

    mu = np.linalg.lstsq(K, y, rcond=None)[0]
    muT = mu.T # intermediate var
    Sigma = np.diag(np.ones(p)) * (1 - p/N)/(yTy + muT@KTy)[0][0]

    Smu = Sigma@mu
    A = KT@K + Sigma
    A_inv = np.linalg.pinv(A)
    b = KTy + Smu
    posterior_mean = A_inv@b
    xi = yTy + muT@Smu - b.T@posterior_mean
    xi = xi[0][0]
    
    return N*((np.linalg.slogdet(Sigma)[1] - np.linalg.slogdet(A)[1])/(2*N) - 0.5*np.log(2*np.pi) - \
              (0.5 + k/N)*np.log(xi/2 + 1/v) - (k*np.log(v))/N + (loggamma(N/2 + k) - loggamma(k))/N)

