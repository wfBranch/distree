import logging
import itertools as itt
import functools as fct
import warnings
from ncon import ncon
from copy import deepcopy
import scipy as sp
import numpy as np
import evoMPS
import mps_ext
import pickle
import find_branches

# Doesn't evoMPS have this function? Regardless, this doesn't really belong in
# this module, but for testing purposes...
def mps_overlap(M1, M2):
    one = sp.array([1], dtype=M1.typ)
    double_one = ncon((one, one), ([-1], [-2]))
    l = double_one
    for pos in range(1, M1.N+1):
        A1 = M1.A[pos]
        A2 = M2.A[pos]
        l = ncon((l, A1, A2.conjugate()), ([1,2], [3,1,-1], [3,2,-2]))
    assert(len(l) == 1)
    return l[0,0]

def mps_overlap_matrix(mpses_rows, mpses_columns=None, normalize=True):
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
                f = mps_overlap(mpses_rows[i], mpses_columns[j])
            else:
                f = np.conjugate(overlaps[j,i])
            overlaps[i, j] = f
    if normalize:
        if hermitian:
            row_norms = column_norms = np.diag(overlaps)
        else:
            row_norms = np.array([np.sqrt(mps_overlap(m, m))
                                  for m in mpses_rows])
            column_norms = np.array([np.sqrt(mps_overlap(m, m))
                                     for m in mpses_columns])
        # Divide elementwise. The newaxis trick makes the division be across
        # rows, as opposed to columns.
        overlaps /= column_norms
        overlaps /= row_norms[:, np.newaxis]
    return overlaps

def main():
    np.set_printoptions(linewidth=120, suppress=True)
    warnings.filterwarnings("error")
    seed = np.random.randint(1, 1e6)
    logging.info("Seed: {}".format(seed))
    np.random.seed(seed)

    N = 100  # Number of sites
    d = 2   # Physical dimension
    q = [d]*(N+1)
    noise_factor = 1e-3  # Coefficients for random noise to add to As
    eliminate_records_in = ""

    # Parameters for the block finding.
    pars = {
        "i1": None,
        "i2": None,
        "system_for_records": "MR",
        "eps_fidelity": 1e-2,
        "eps_M_nonint": 1e-2,
        "coeff_tol": 1e-4,
        "k": 20,
        "comm_tau": 1e-2,
        "comm_threshold": 1e-7,
        "comm_iters": 10000,
        "projopt_iters": 100,
        "projopt_threshold": 1e-10,
    }

    # Bond dimensions of the terms in the sum
    #Ds = [
    #    [1,3,5,8,5,5,8,9,8,3,1],
    #    [1,3,5,6,5,5,8,7,4,2,1]
    #]
    Ds = [
        [20]*(N+1),
        [20]*(N+1),
        [20]*(N+1),
        #[4,7,8,3,5,6,4,2,3,2,3,2,5,3,2],  # Requires N=14
        #[5,4,8,3,6,9,7,4,5,8,6,5,7,4,5],
        #[7,4,8,9,6,3,3,2,4,7,9,7,6,7,4],
        #[1]*(N+1),
        #[1]*(N+1),
        #[25]*(N+1),
        #[25]*(N+1),
    ]
    # Random generate coefficients for the sum. They square sum to 1.
    orig_coeffs = np.random.rand(len(Ds)) + 1j*np.random.rand(len(Ds))
    orig_coeffs /= np.linalg.norm(orig_coeffs)
    orig_coeffs = list(orig_coeffs)

    # Generate random MPSes of the given dimensions...
    orig_terms = []
    for b in range(len(Ds)):
        term = evoMPS.mps_gen.EvoMPS_MPS_Generic(N, Ds[b], q)
        term.randomize()
        Ds[b] = term.D  # This may have changed due to updates by evoMPS
        orig_terms.append(term)
    if eliminate_records_in:
        if pars["i1"] and pars["i2"] and 1 <= pars["i1"] <= pars["i2"] <= N:
            i1_temp = pars["i1"]
            i2_temp = pars["i2"]
        else:
            i1_temp = int(N/3) + 1
            i2_temp = N - int(N/3)
        s0 = orig_terms[0]
        for s in orig_terms[1:]:
            if "R" in eliminate_records_in:
                for i in range(i2_temp+1, N+1):
                    s.A[i] = deepcopy(s0.A[i])
                    s.D[i] = s0.D[i]
            if "M" in eliminate_records_in:
                for i in range(i1_temp, i2_temp+1):
                    s.A[i] = deepcopy(s0.A[i])
                    s.D[i] = s0.D[i]
            s.update()

    # ...sort them by coefficient, for convenience...
    order = np.argsort(-abs(np.array(orig_coeffs)))
    orig_coeffs = list(map(orig_coeffs.__getitem__, order))
    orig_terms = list(map(orig_terms.__getitem__, order))

    assert(all(len(D) == N+1 for D in Ds))

    # ...take their weighted sum and add some noise to it...
    lincomb = mps_ext.add_MPS_list(orig_terms, coeffs=orig_coeffs)
    lincomb_nonoise = deepcopy(lincomb)
    lincomb.add_noise(noise_factor)

    logging.info("Original coefficients: {}".format(orig_coeffs))
    logging.info("Bond dimension of linear combination: \n{}"
                 .format(lincomb.D))
    logging.info("Bond dimensions of the original terms: \n{}"
                 .format([t.D for t in orig_terms]))

    # ...and decompose the linear combination back into terms with the same
    # bond dimensions as the original ones.
    branch_list, coeff_list = find_branches.find_branches_sparse(
        lincomb, pars=pars
    )

    # Print possibly interesting things.
    logging.info("Noise factor: {}".format(noise_factor))
    logging.info("Overlap matrix of the original terms: \n{}"
                 .format(abs(mps_overlap_matrix(orig_terms))))
    logging.info("Overlap matrix of the found terms: \n{}"
                 .format(abs(mps_overlap_matrix(branch_list))))
    logging.info("Overlap matrix between found and original: \n{}"
                 .format(abs(mps_overlap_matrix(branch_list, orig_terms))))
    logging.info("Bond dimension of linear combination: \n{}"
                 .format(lincomb.D))
    logging.info("Bond dimensions of the original terms: \n{}"
                 .format([t.D for t in orig_terms]))
    logging.info("Bond dimensions of the found terms: \n{}"
                 .format([t.D for t in branch_list]))
    logging.info("Original coefficients: {}".format(orig_coeffs))
    logging.info("abs thereof:           {}".format(abs(np.array(orig_coeffs))))
    logging.info("Found coefficients:    {}".format(coeff_list))

    # Sum up the found terms, and compute the fidelity with the original linear
    # combination explicitly.
    lincomb_reco = mps_ext.add_MPS_list(branch_list, coeffs=coeff_list)
    fid_reco = mps_overlap(lincomb_reco, lincomb)
    logging.info("Reconstructed fidelity: {} (abs {})"
                 .format(fid_reco, abs(fid_reco)))
    logging.info("Fidelity of no-noise vs target: {}"
                 .format(abs(mps_overlap(lincomb, lincomb_nonoise))))
    logging.info("Fidelity of no-noise vs reco:   {}"
                 .format(abs(mps_overlap(lincomb_reco, lincomb_nonoise))))

if __name__ == "__main__":
    main()
