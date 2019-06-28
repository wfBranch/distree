import mps_ext
import logging
from copy import deepcopy
import scipy.sparse.linalg as spsla
import scipy as sp
import functools as fct
import operator as opr
from evoMPS.core_common import eps_l_noop, eps_r_noop

def find_branches(s, pars):
    if 1 <= pars["i1"] <= pars["i2"] <= s.N:
        i1 = pars["i1"]
        i2 = pars["i2"]
    else:
        i1=int(s.N/3)+1
        i2=s.N-int(s.N/3)
    records = mps_ext.find_records_mps(
        s,i1,i2,
        system_for_records=pars["system_for_records"],
        eps_ker_C=pars["eps_ker_C"], coeff_tol=pars["coeff_tol"],
        degeneracy_tol=pars["degeneracy_tol"],
        transfer_operator_method=pars["transfer_operator_method"],
        max_branch_ratio = pars["max_branch_ratio"]
        )
    branch_list, coeff_list, rank_list, projector_list, C, D = records
    return branch_list, coeff_list

def shift_canpoint_evomps(mps, point, direction):
    """
    Move the point of center canonicalization for an evoMPS object. The list of
    MPS tensors A is modified in place, with no effort to update the state of
    the object to match. The current point of canonicalization is required as
    an argument. The third argument, direction, should be +1 or -1, for
    shifting the point right or left.
    """
    neigh = point + direction
    A_point = mps.A[point]
    A_neigh = mps.A[neigh]
    q_point = mps.q[point]
    if direction == -1:
        A_point = A_point.transpose((0,2,1))
        A_neigh = A_neigh.transpose((0,2,1))
    D1, D2 = A_point.shape[1:]
    A_point_mat = A_point.reshape((q_point * D1, D2))
    Q, R = sp.linalg.qr(A_point_mat, mode='economic')
    B_point = Q.reshape((q_point, D1, D2))
    B_neigh = sp.tensordot(R, A_neigh, axes=[1,1])
    B_neigh = B_neigh.transpose((1,0,2))
    if direction == -1:
        B_point = B_point.transpose((0,2,1))
        B_neigh = B_neigh.transpose((0,2,1))
    mps.A[point] = B_point
    mps.A[neigh] = B_neigh

# TODO Doesn't evoMPS have a function for this?
# TODO Even if it doesn't, this should make use of the assumption that both
# states are up-to-date and then use ls and rs.
def mps_overlap(s1, s2):
    """
    Compute the overlap of two evoMPS objects. The second one is conjugated.
    """
    one = sp.array([1], dtype=s1.typ)
    double_one = sp.outer(one, one)
    l = double_one
    for pos in range(1, s1.N+1):
        l = eps_l_noop(l, s1.A[pos], s2.A[pos])
    assert(len(l) == 1)
    return l[0,0]

def find_branches_sparse(s, pars):
    """ Recursively find branches in s, two at a time, using
    find_two_branches_sparse. See it's docstring for description of the
    parameters.
    """
    outer_branches, outer_coeffs = find_two_branches_sparse(s, pars)
    if len(outer_branches) > 1:
        total_branches = []
        total_coeffs = []
        for branch, coeff in zip(outer_branches, outer_coeffs):
            inner_branches, inner_coeffs = find_branches_sparse(branch, pars)
            inner_coeffs = [coeff*c for c in inner_coeffs]
            total_branches += inner_branches
            total_coeffs += inner_coeffs
    else:
        total_branches = outer_branches
        total_coeffs = outer_coeffs
    total_coeffs, total_branches = zip(*sorted(
        zip(total_coeffs, total_branches),
        key=lambda pair: abs(pair[0]),
        reverse=True
    ))
    return total_branches, total_coeffs

def find_blocks_in_basis(A):
    dim = A.shape[0]
    corner_sizes = [sp.sum(A[:i,i:]) for i in range(1, dim)]
    i_min = sp.argmin(corner_sizes)
    s1 = list(range(i_min+1))
    s2 = [i for i in range(dim) if i not in s1]
    while True:
        candidate_changes = []
        for i in range(dim):
            if i in s1:
                if len(s1) == 1:
                    cost_change = sp.inf
                else:
                    cost_change = (sp.sum(A[s1,i]) - sp.sum(A[i,s2]) -
                                   A[i,i])
            else:
                if len(s2) == 1:
                    cost_change = sp.inf
                else:
                    cost_change = (-sp.sum(A[s1,i]) + sp.sum(A[i,s2])
                                   - A[i,i])
            candidate_changes.append(cost_change)
        i = sp.argmin(candidate_changes)
        cost_change = candidate_changes[i]
        if cost_change < 0:
            if i in s1:
                s1.remove(i)
                s2.append(i)
            else:
                s2.remove(i)
                s1.append(i)
        else:
            break
    return s1, s2

def find_two_branches_sparse(s, pars):
    """ Try to find two branches with records on regions L, M, R, where
    L = {1,..,i1-1}, M = {i1,...,i2}, and R = {i2+1,...,N}.
    The input MPS s does not need to be in any particular canonical form.
    Parameters:
    - pars["i1"] and pars["i2"] set the boundaries of L, M and R.
      If these are set to nonsense values (-1, None, ...) the defaults i1 =
      N/3+1 and i2 = 2N/3 are used.
    - pars["system_for_records"] = 'RM', 'R', or 'M'.
    - pars["eps_fidelity"] is the tolerance for the error measure 1 - fidelity
      of the decomposition. If the error is worse than this, the proposed
      branching will be rejected.
    - pars["eps_M_nonint"] is the tolerance for non-interference on M. It is
      only used if system_for_records in M and RM.
    - pars["k"] is the number of random product states to use in checking that
      the branch projectors commute with the MPS over M. this is only used if
      system_for_records is 'R' or 'RM', since finding records on 'M' only can
      be done in O(D^3) without sampling.
    """
    # Set the default i1 and i2 if no sensible values are provided.
    if pars["i1"] and pars["i2"] and 1 <= pars["i1"] <= pars["i2"] <= s.N:
        i1 = pars["i1"]
        i2 = pars["i2"]
    else:
        i1 = int(s.N/3) + 1
        i2 = s.N - int(s.N/3)

    dimL = s.D[i1-1]
    dimR = s.D[i2]

    if dimL < 2:
        msg = "Can't branch because at D at i1 is already 1."
        logging.info(msg)
        return [s], [1.0]
    if dimR < 2:
        msg = "Can't branch because at D at i2 is already 1."
        logging.info(msg)
        return [s], [1.0]

    # We make a copy of s because we will change its gauge to search for
    # branches: We'll put it into central gauge with the point of canonicality
    # lying within M.
    s_orig = s
    s = deepcopy(s)
    s.update()
    for i in range(1, i1):
        shift_canpoint_evomps(s, i, 1)

    if 'R' in pars["system_for_records"]:
        matrices_R = []
        for j in range(pars["k"]):
            M = sp.eye(dimL, dimL)
            for i in range(i1, i2+1):
                state_i = sp.random.randn(s.q[i])
                if s.typ == sp.complex_:
                    state_i = state_i + 1j*sp.random.randn(s.q[i])
                state_i = state_i/sp.linalg.norm(state_i)
                A_i = sp.tensordot(s.A[i], state_i, axes=(0, 0))
                M = sp.dot(M, A_i)
            M = sp.dot(M, M.conjugate().transpose())
            matrices_R.append(M)
    k_factor_R = sp.prod(sp.array(s.q[i1:i2+1], dtype=sp.float_))/pars["k"]
    matrices_R = [M*k_factor_R for M in matrices_R]
    if 'M' in pars["system_for_records"]:
        matrices_M = []
        for j in range(pars["k"]):
            # TODO Do we want positive definite matrices like below, or should
            # we only pick product states?
            r = sp.random.randn(dimR, dimR)
            if s.typ == sp.complex_:
                r = r + 1j*sp.random.randn(dimR, dimR)
            r = sp.dot(r, r.conjugate().transpose())
            r = r/sp.linalg.norm(r)
            for i in reversed(range(i1, i2+1)):
                r = eps_r_noop(r, s.A[i], s.A[i])
            matrices_M.append(r)
    k_factor_M = dimR/pars["k"]
    matrices_M = [M*k_factor_M for M in matrices_M]
    matrices = matrices_R + matrices_M
    average_norm = sp.average([sp.linalg.norm(M) for M in matrices])
    matrices = [M/average_norm for M in matrices]
    conjugates = [r.conjugate().transpose() for r in matrices]
    squaresL = [sp.dot(rdg, r) for r, rdg in zip(matrices, conjugates)]
    squaresR = [sp.dot(r, rdg) for r, rdg in zip(matrices, conjugates)]
    def C_func(X):
        X = sp.reshape(X, (dimL, dimL))
        res = sp.zeros_like(X)
        for i in range(len(matrices)):
            term1 = sp.dot(squaresL[i], X)
            term2 = sp.dot(X, squaresR[i])
            term3 = sp.dot(conjugates[i], sp.dot(X, matrices[i]))
            term4 = sp.dot(matrices[i], sp.dot(X, conjugates[i]))
            res += term1 + term2 - term3 - term4
        res = sp.reshape(res, (dimL, dimL))
        return res
    C = spsla.LinearOperator((dimL**2, dimL**2), C_func, dtype=s.typ)

    # DEBUG
    #eyeL = sp.reshape(sp.eye(dimL, dtype=s.typ), (dimL**2,))
    #print(sp.linalg.norm(C.matvec(eyeL)))
    #print(C.matvec(eyeL))
    # END DEBUG

    # Find the largest magnitude eigenvalue of C. This will be used to scale C
    # so that all its eigenvalues are negative, making the smallest ones the
    # largest in magnitude.
    # TODO There's probably a way to give a reasonable upper bound for the
    # largest eigenvalue using Frobenius norms, avoiding this eigenvalue
    # search.
    S, U = spsla.eigsh(C, k=1)
    shift = abs(S[0])
    # TODO Make a function that creates this and C_func?
    def C_scaled_func(X):
        X = sp.reshape(X, (dimL, dimL))
        res = -shift*X
        # TODO This is far from optimal: We could sum up all squares
        # before-hand, and also use reshapes for term3 and term4. Also, the
        # last two are the same since all our matrices are hermitian (and in
        # fact pos. semidef.).
        for i in range(len(matrices)):
            term1 = sp.dot(squaresL[i], X)
            term2 = sp.dot(X, squaresR[i])
            term3 = sp.dot(conjugates[i], sp.dot(X, matrices[i]))
            term4 = sp.dot(matrices[i], sp.dot(X, conjugates[i]))
            res += term1 + term2 - term3 - term4
        res = sp.reshape(res, (dimL, dimL))
        return res
    C_scaled = spsla.LinearOperator((dimL**2, dimL**2), C_scaled_func,
                                    dtype=s.typ)
    # DEBUG
    #C_scaled = C - spsla.LinearOperator((dimL**2, dimL**2), lambda x: shift*x)
    # END DEBUG

    # Find the lowest two eigenpairs of C. Two because we are looking for two
    # branches.
    S, U = spsla.eigsh(C_scaled, k=2)
    # DEBUG
    #S, U = spsla.eigsh(C, k=2, which="SM")
    # END DEBUG
    U_tensor = sp.reshape(U, (dimL, dimL, len(S)))
    S += shift  # Correct for the above shift.
    # DEBUG
    print("Spectrum of C: {} (shift: {})".format(S, shift))
    # END DEBUG

    # Find X, a linear combination of U[:,0] and U[:,1] that has a clear
    # separation between blocks. This is done by expressing both U0 and U1 in
    # the basis that diagonalizes U0, and choosing a linear combination that
    # makes sure there's a zero somewhere on the diagonal. If there are blocks
    # to be found then U0 and U1 are approximately diagonal in the same basis,
    # and this choice then guarantees that X has an eigenvalue that is almost
    # zero, which must be clearly separated from some other eigenvalues, since
    # X has norm 1. Another entirely valid approach would be to make X be a
    # random linear combination of U0 and U1. What we are doing here is just
    # minimizing the probability for accidental near-degeneracies in the
    # specturm of X.
    U0 = U_tensor[:,:,0]
    U1 = U_tensor[:,:,1]
    # If a matrix is in the kernel of C, then it's conjugate must be as well.
    U0 = (U0 + U0.conjugate().transpose())/2
    U1 = (U1 + U1.conjugate().transpose())/2
    S, V = sp.linalg.eigh(U0)
    U1_diag = sp.dot(V.conjugate().transpose(), sp.dot(U1, V))
    theta = sp.arctan(-U1_diag[-1,-1]/S[-1])
    X = sp.sin(theta)*U0 + sp.cos(theta)*U1

    # Diagonalize X. Its eigenbasis is the basis in which we will project onto
    # the branches. In an ideal world with perfect records, the spectrum of X
    # would consist of exactly degenerate groups, one group for each block that
    # can be found. In practice these degeneracies will be muddied. To separate
    # out the blocks we constact a matrix called A, such that the amount of
    # non-commutation of the projectors with the matrices we've sampled is
    # measured by the off-diagonal blocks of A. We then search for the division
    # of A into two blocks that minimizes this error measure.
    S, V = sp.linalg.eigh(X)
    # DEBUG
    print("Spectrum of X: {}".format(S))  # DEBUG
    print(sp.linalg.eigh(U0)[0])
    print(sp.linalg.eigh(U1)[0])
    print("Whether U0, U1 and X are in the kernel of C:")
    print(sp.linalg.norm(C.matvec(sp.reshape(U0, (-1,)))))
    print(sp.linalg.norm(C.matvec(sp.reshape(U1, (-1,)))))
    print(sp.linalg.norm(C.matvec(sp.reshape(X, (-1,)))))
    #print(U0)
    #print(U1)
    # END DEBUG
    matrices_V = [sp.dot(V.conjugate().transpose(), sp.dot(M, V))
                    for M in matrices]
    # DEBUG
    #print("="*50)
    #[print(sp.linalg.norm(M)) for M in matrices]
    #print("="*50)
    #[print(sp.linalg.norm(M)) for M in matrices_V]
    #input()
    # END DEBUG
    A = sum(abs(M)**2 for M in matrices_V)
    print(A)  # DEBUG
    s1, s2 = find_blocks_in_basis(A)
    P1 = sp.dot(V[:,s1], V[:,s1].conjugate().transpose())
    P2 = sp.dot(V[:,s2], V[:,s2].conjugate().transpose())
    projector_list_L = [P1, P2]
    # DEBUG
    print("s1 :", s1)
    print("s2 :", s2)
    print(sp.linalg.norm(C_func(P1)))
    print(sp.linalg.norm(C_func(P2)))
    # END DEBUG

    if "R" in pars["system_for_records"]:
        # Now that we have the projectors at the boundary of L and M, find the best
        # ones to match at the boundary of M and R.
        # TODO Transport P1 and P2 to i2. Take the difference, use that as the
        # environment to find the best basis. Maybe run the same thing as above to
        # find the correct partition? Note that it's going to have negative
        # eigenvalues on one branch and positive on the other, so this should be
        # easy.
        P1_env = P1
        P2_env = P2
        for i in range(i1, i2+1):
            P1_env = eps_l_noop(P1_env, s.A[i], s.A[i])
            P2_env = eps_l_noop(P2_env, s.A[i], s.A[i])
        env = P1_env - P2_env
        S, V = sp.linalg.eigh(env)
        s1 = S > 0
        s2 = S <= 0
        print("env spec: ", S)  # DEBUG
        R1 = sp.dot(V[:,s1], V[:,s1].conjugate().transpose())
        R2 = sp.dot(V[:,s2], V[:,s2].conjugate().transpose())
        projector_list_R = [R1, R2]

    # Construct the branches.
    branch_list = []
    for i in range(len(projector_list_L)):
        branch = deepcopy(s)
        branch.A[i1-1] = sp.tensordot(branch.A[i1-1], projector_list_L[i],
                                      axes=(2,0))
        if "R" in pars["system_for_records"]:
            branch.A[i2] = sp.tensordot(branch.A[i2], projector_list_R[i],
                                        axes=(2,0))
        branch.update(auto_truncate=True)
        branch_list.append(branch)
    dim_list_L = [b.D[i1-1] for b in branch_list]
    dim_list_R = [b.D[i2] for b in branch_list]

    # Compute the coefficients the branches should have. Note that we compute
    # the fidelity with respect to s_orig, since updating s may have caused
    # global phase changes.
    # TODO This could be done in a more clever way using canonicality
    # properties of the original MPS/the fact that we know what l and r are and
    # how the branches are related to the original state through virtual
    # projectors. Note that the optimality of this choice already relies on the
    # branches being orthogonal. I'll leave this the way it is for now though,
    # because this is fool-proof and not very costly.
    coeff_list = [mps_overlap(s_orig, branch) for branch in branch_list]

    if "M" in pars["system_for_records"]:
        TM_shp = (dim_list_L[0]*dim_list_L[1], dim_list_R[0]*dim_list_R[1])
        if 0 in TM_shp:
            M_nonint = 0
        else:
            def TM_func_rmatvec(X):
                shp = X.shape
                X = X.conjugate()
                X = sp.reshape(X, (dim_list_L[0], dim_list_L[1]))
                for i in range(i1, i2+1):
                    X = eps_l_noop(X, branch_list[0].A[i], branch_list[1].A[i])
                X = X.conjugate()
                X = sp.reshape(X, TM_shp[1])
                return X
            def TM_func_matvec(X):
                shp = X.shape
                X = sp.reshape(X, (dim_list_R[0], dim_list_R[1]))
                for i in reversed(range(i1, i2+1)):
                    X = eps_r_noop(X, branch_list[0].A[i], branch_list[1].A[i])
                X = sp.reshape(X, TM_shp[0])
                return X
            TM = spsla.LinearOperator(TM_shp, matvec=TM_func_matvec,
                                      rmatvec=TM_func_rmatvec, dtype=s.typ)
            if 1 in TM_shp or 2 in TM_shp:
                TM_dense = TM.matmat(sp.eye(TM_shp[1]))
                M_nonint = sp.linalg.svd(TM_dense)[1][0]
            else:
                M_nonint = spsla.svds(
                    TM, k=1, return_singular_vectors=False
                )[0]
    
    fid = sum(abs(sp.array(coeff_list))**2)
    msg = "Found a branch decomposition with with local (i1, i2) bond dimensions ({}, {}) and ({}, {}) and coefficients {} and {}.".format(dim_list_L[0], dim_list_R[0], dim_list_L[1], dim_list_R[1], *coeff_list)
    logging.info(msg)
    msg = "Fidelity: {}".format(fid)
    logging.info(msg)
    if "M" in pars["system_for_records"]:
        msg = "Non-interference on M: {}".format(M_nonint)
        logging.info(msg)

    if 1 - fid > pars["eps_fidelity"]:
        msg = "Rejecting this decomposition due to too low fidelity."
        logging.info(msg)
        # s_orig instead of s, because s.update() may have caused a phase
        # change.
        branch_list = [s_orig]
        coeff_list = [1.0]
    elif "M" in pars["system_for_records"] and M_nonint > pars["eps_M_nonint"]:
        msg = "Rejecting this decomposition due to too high interference on M."
        logging.info(msg)
        # s_orig instead of s, because s.update() may have caused a phase
        # change.
        branch_list = [s_orig]
        coeff_list = [1.0]
    else:
        msg = "Accepting this decomposition."
        logging.info(msg)

    return branch_list, coeff_list

