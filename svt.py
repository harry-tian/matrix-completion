import numpy as np
import numpy.linalg as LA

def svt_solve(X, mask, threshold, eps=1e-5, max_iters=1000):
    """  A naive implementation of a singular value thresholding algorithm  """

    ### initialization
    X_hat = np.zeros_like(X)
    X_hat = fill_vals(X_hat, X, mask)

    for _ in range(max_iters):
        X_old = X_hat
        U, S, V = LA.svd(X_hat)

        ### threshold the SVs
        SVs = [sv if sv > threshold else 0 for sv in S]
        
        #### construct S_hat and X_hat with thresholded SVs
        S_hat = np.zeros_like(X_hat)
        for i, sv in enumerate(SVs):  S_hat[i,i] = sv
        X_hat = U @ S_hat @ V
        X_hat = fill_vals(X_hat, X, mask)


        err = LA.norm(X_hat-X_old)
        if err < eps:   break

    return X_hat

def fill_vals(X_hat, X, mask):
    """  Filling X_hat with the observed values from X; only works for 2d matrices"""
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if mask[i,j] != 0: 
                X_hat[i,j] = X[i,j]

    return X_hat