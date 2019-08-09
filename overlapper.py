import logging
import itertools as itt
import functools as fct
import warnings
import os
from ncon import ncon
from copy import deepcopy
import scipy as sp
import numpy as np
import evoMPS
import mps_ext
import h5py
from evoMPS import mps_gen
from collections import Iterable

# Doesn't evoMPS have this function? Regardless, this doesn't really belong in
# this module, but for testing purposes...
def mps_overlap(M1, M2, op=None, opsite=None):
    if op is not None and opsite is None:
        if isinstance(opsite, Iterable):
            opsite = np.random.randint(opsite[0], opsite[1]+1)
        else:
            opsite = np.random.randint(1, M1.N+2)
    if op == "rand" or op == "random":
        op = np.random.randn(M1.q[opsite], M2.q[opsite])
        op /= np.linalg.norm(op)

    one = sp.array([1], dtype=M1.typ)
    double_one = ncon((one, one), ([-1], [-2]))
    l = double_one
    for pos in range(1, M1.N+1):
        A1 = M1.A[pos]
        A2 = M2.A[pos]
        if op is not None and pos == opsite:
            A1 = ncon((A1, op), ([1,-2,-3], [1,-1]))
        l = ncon((l, A1, A2.conjugate()), ([1,2], [3,1,-1], [3,2,-2]))
    assert(len(l) == 1)
    return l[0,0]

def mps_overlap_matrix(mpses_rows, mpses_columns=None, normalize=True,
                       op=None, opsite=None):
    if op is not None and opsite is None:
        if isinstance(opsite, Iterable):
            opsite = np.random.randint(opsite[0], opsite[1]+1)
        else:
            opsite = np.random.randint(1, M1.N+2)
    if op == "rand" or op == "random":
        op = np.random.randn(M1.q[opsite], M2.q[opsite])
        op /= np.linalg.norm(op)
    # Given two lists of MPSes, returns a matrix of overlaps.
    # Given one list of MPSes, returns the Grammian matrix.
    # By default, vectors are normalized before overlaps are computed.
    hermitian = mpses_columns is None
    if hermitian:
        mpses_columns = mpses_rows
    overlaps = np.empty((len(mpses_rows), len(mpses_columns)),
                        dtype=np.complex_)
    for i in range(len(mpses_rows)):
        for j in range(len(mpses_columns)):
            if i <= j or not hermitian:
                f = mps_overlap(mpses_rows[i], mpses_columns[j],
                                op=op, opsite=opsite)
            else:
                f = np.conjugate(overlaps[j,i])
            overlaps[i, j] = f
    if normalize:
        row_norms = np.array([np.sqrt(mps_overlap(m, m))
                              for m in mpses_rows])
        column_norms = np.array([np.sqrt(mps_overlap(m, m))
                                 for m in mpses_columns])
        # Divide elementwise. The newaxis trick makes the division be across
        # rows, as opposed to columns.
        overlaps /= column_norms
        overlaps /= row_norms[:, np.newaxis]
    return overlaps

def state_from_tensors(tensors):
    qn = tensors[0].shape[0]
    N = len(tensors)
    q = [qn]*(N+1)
    s = mps_gen.EvoMPS_MPS_Generic(N, np.ones(N+1,dtype=int), q)
    A = [None] + [tensors[j] for j in range(N)]
    s.set_state_from_tensors(A)
    return s

def main():
    #folder = "testjob_60coeffcap/data"
    folder = "teststates"
    files = (
        "testjob_60coeffcap_state_t2.0.h5",
        "testjob_nob100_state_t2.0.h5",
    )

    paths = [os.path.join(folder, f) for f in files]
    states = []
    for path in paths:
        with h5py.File(path, "r") as f:
            tensors_flat = f['tensors_flat']
            q = f['q']
            D = f['D']
            N = len(q)
            tensors = []
            for j in range(N):
                tensors.append(tensors_flat[j].reshape(q[j],D[j],D[j+1]))
            state = state_from_tensors(tensors)
            states.append(state)

    overlap_matrix = mps_overlap_matrix(states)
    logging.info("Overlap matrix:\n{}\n".format(overlap_matrix))
    logging.info("Overlap matrix abs:\n{}\n".format(abs(overlap_matrix)))

    N = states[0].N
    site = N//6
    interference_matrix_L = mps_overlap_matrix(states)
    logging.info("Interference matrix at site {}/{}:\n{}\n".format(site, N, interference_matrix_L))
    logging.info("Interference matrix at site {}/{} abs:\n{}\n".format(site, N, abs(interference_matrix_L)))

    site = N//2
    interference_matrix_L = mps_overlap_matrix(states)
    logging.info("Interference matrix at site {}/{}:\n{}\n".format(site, N, interference_matrix_L))
    logging.info("Interference matrix at site {}/{} abs:\n{}\n".format(site, N, abs(interference_matrix_L)))

    site = 5*N//6
    interference_matrix_L = mps_overlap_matrix(states)
    logging.info("Interference matrix at site {}/{}:\n{}\n".format(site, N, interference_matrix_L))
    logging.info("Interference matrix at site {}/{} abs:\n{}\n".format(site, N, abs(interference_matrix_L)))

if __name__ == "__main__":
    main()
