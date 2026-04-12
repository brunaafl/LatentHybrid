"""Euclidean Alignment as pre-processed."""

import copy

import numpy as np
from numpy import any, iscomplexobj, isfinite, real
from scipy.linalg import inv, sqrtm


def euclidean_alignment(data, y=None):
    data = copy.deepcopy(data)

    assert len(data.shape) == 3

    r = 0
    for trial in data:
        cov = np.cov(trial, rowvar=True)
        r += cov

    r = r / len(data)

    compare = np.allclose(r, np.identity(r.shape[0]))

    if not compare:

        if iscomplexobj(r):
            print("covariance matrix problem")
        if iscomplexobj(sqrtm(r)):
            print("covariance matrix problem sqrt")

        r_op = inv(sqrtm(r))

        if iscomplexobj(r_op):
            print("WARNING! Covariance matrix was not SPD somehow. " +
                  "Can be caused by running ICA-EOG rejection, if " +
                  "not, check data!!")
            r_op = real(r_op).astype(np.float64)
        elif not any(isfinite(r_op)):
            print("WARNING! Not finite values in R Matrix")

        result = np.matmul(r_op, data)

    else:
        print("Already aligned!")
        result = data
        r_op = 0

    return result, r_op


def compute_EA(X, size=24, domain=None, estimator='lwf', dtype='raw'):
    X_aux = []

    if domain is not None:
        for d in np.unique(domain):
            X_batch = X[domain == d]
            X_batch_EA, _ = euclidean_alignment(X_batch)
            X_aux.append(X_batch_EA)
        covmat_EA = np.concatenate(X_aux)
    else:
        if size is None:
            m = X.shape[0]
        else:
            m = size
        n = X.shape[0]

        for k in range(int(n / m)):
            X_batch = X[k * m:(k + 1) * m]
            X_batch_EA, _ = euclidean_alignment(X_batch)
            X_aux.append(X_batch_EA)
        covmat_EA = np.concatenate(X_aux)
    return covmat_EA

def split_runs_EA(X, len_run, y=None):
    X_aux = []
    m = len_run
    n = X.shape[0]
    for k in range(int(n / m)):
        run = X[k * m:(k + 1) * m]
        run_EA, _ = euclidean_alignment(run)
        X_aux.append(run_EA)
    X_EA = np.concatenate(X_aux)
    return X_EA


