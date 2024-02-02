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
