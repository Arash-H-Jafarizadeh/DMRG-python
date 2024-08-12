import numpy as np


def mps_svd(M, chi_max=None, threshold=None):

    chi1, p1, p2, chi2 = M.shape

    M = M.reshape((M.shape[0]*M.shape[1],M.shape[2]*M.shape[3]))  # (v1*p1, p2*v2)

    # svd
    A, S, B = np.linalg.svd(M, full_matrices=False)  # (v1*p1, p2*v2) -> (v1*p1, K), (K,), (K, p2*v2)  K=min(p1*v1, p2*v2)

    chi = len(S)

    if chi_max is not None:
        chi = min(chi, chi_max)
        S = S[:chi]
        A = A[:, :chi]
        B = B[:chi, :]

    if threshold is not None:
        S_sq = S**2
        S_sq_sum = np.cumsum(S_sq)
        chi = max(1,np.sum(S_sq_sum < 1-threshold))
        S = S[:chi]
        A = A[:, :chi]
        B = B[:chi, :]

    S = S/np.linalg.norm(S)  # normalize
    schmidt_matrix = np.diag(S)  # (chi, chi)

    A = A.reshape((chi1, p1, chi))  # (v1, p1, v2=chi)

    B = B.reshape((chi, p2, chi2))  # (v1=chi, p2, v2)

    return A, schmidt_matrix, B

    
