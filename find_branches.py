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
    # Normalize the rows and columns of A so that the diagonal is ones.
    A = deepcopy(A)
    for i in range(0, dim):
        d = A[i,i]
        A[i,:] /= sp.sqrt(d)
        A[:,i] /= sp.sqrt(d)
    corner_avgs = [sp.sum(A[:i,i:])/(i*(dim-i)) for i in range(1, dim)]
    i_min = sp.argmin(corner_avgs)
    cost = corner_avgs[i_min]
    s1 = list(range(i_min+1))
    s2 = [i for i in range(dim) if i not in s1]
    while True:
        candidate_costs = []
        for i in range(dim):
            if i in s1:
                if len(s1) == 1:
                    new_cost = sp.inf
                else:
                    new_elements = (sp.sum(A[s1,i]) - sp.sum(A[i,s2]) - A[i,i])
                    l1, l2 = len(s1), len(s2)
                    new_cost = (cost*l1*l2 + new_elements)/((l1-1)*(l2+1))
            else:
                if len(s2) == 1:
                    new_cost = sp.inf
                else:
                    new_elements = (-sp.sum(A[s1,i]) + sp.sum(A[i,s2]) - A[i,i])
                    l1, l2 = len(s1), len(s2)
                    new_cost = (cost*l1*l2 + new_elements)/((l1+1)*(l2-1))
            candidate_costs.append(new_cost)
        i = sp.argmin(candidate_costs)
        new_cost = candidate_costs[i]
        if new_cost < cost:
            cost = new_cost
            if i in s1:
                s1.remove(i)
                s2.append(i)
            else:
                s2.remove(i)
                s1.append(i)
        else:
            break
    return s1, s2

def optimize_branch_projectors(s, i1, i2, P1, P2):
    P1_env = P1
    P2_env = P2
    for i in range(i1, i2+1):
        P1_env = eps_l_noop(P1_env, s.A[i], s.A[i])
        P2_env = eps_l_noop(P2_env, s.A[i], s.A[i])
    env = P1_env - P2_env
    S, V = sp.linalg.eigh(env)
    s1 = S > 0
    s2 = S <= 0
    R1 = sp.dot(V[:,s1], V[:,s1].conjugate().transpose())
    R2 = sp.dot(V[:,s2], V[:,s2].conjugate().transpose())
    
    R1_env = R1
    R2_env = R2
    for i in reversed(range(i1, i2+1)):
        R1_env = eps_r_noop(R1_env, s.A[i], s.A[i])
        R2_env = eps_r_noop(R2_env, s.A[i], s.A[i])
    env = R1_env - R2_env
    S, V = sp.linalg.eigh(env)
    s1 = S > 0
    s2 = S <= 0
    P1 = sp.dot(V[:,s1], V[:,s1].conjugate().transpose())
    P2 = sp.dot(V[:,s2], V[:,s2].conjugate().transpose())
    return P1, P2, R1, R2

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
    - pars["comm_tau"] is parameter similar to a Trotter step, used in
      approximating exp(sum_i -K_i) as (prod_i exp(-tau K_i))^1/tau, where K_i
      are commutator superoperators squared. pars["comm_threshold"] and
      pars["comm_iters"] are the threshold for convergence and maximum number
      of iterations allowed when looking for the dominant eigenvector of this
      operator. "comm" stands for commutator.
    - pars["projopt_iters"] and pars["projopt_threshold"] are the maximum
      number of iterations and threshold for convergence when optimizing the
      projectors, alternating between the ones at i1 and the ones at i2.
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
        msg = "Can't branch because D at i1 is already 1."
        logging.info(msg)
        return [s], [1.0]
    if dimR < 2:
        msg = "Can't branch because D at i2 is already 1."
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

    matrices_R = []
    if 'R' in pars["system_for_records"]:
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
            # TODO Choose how to normalize the M's. Here's one choice, a couple
            # of lines below is another candidate.
            M /= sp.linalg.norm(M)
            matrices_R.append(M)
    #k_factor_R = sp.prod(sp.array(s.q[i1:i2+1], dtype=sp.float_))/pars["k"]
    #matrices_R = [M*k_factor_R for M in matrices_R]

    matrices_M = []
    if 'M' in pars["system_for_records"]:
        for j in range(pars["k"]):
            # TODO Do we want positive definite matrices like below, or should
            # we only pick product states?
            M = sp.random.randn(dimR, dimR)
            if s.typ == sp.complex_:
                M = M + 1j*sp.random.randn(dimR, dimR)
            M = sp.dot(M, M.conjugate().transpose())
            M = M/sp.linalg.norm(M)
            for i in reversed(range(i1, i2+1)):
                M = eps_r_noop(M, s.A[i], s.A[i])
            # TODO Choose how to normalize the M's. Here's one choice, a couple
            # of lines below is another candidate.
            M /= sp.linalg.norm(M)
            matrices_M.append(M)
    #k_factor_M = dimR/pars["k"]
    #matrices_M = [M*k_factor_M for M in matrices_M]

    matrices = matrices_R + matrices_M
    average_norm = sp.average([sp.linalg.norm(M) for M in matrices])
    matrices = [M/average_norm for M in matrices]

    decompositions = [sp.linalg.eigh(M) for M in matrices]
    bases = [d[1] for d in decompositions]
    bases_dg = [U.conjugate().transpose() for U in bases]
    diff_matrices = [sp.reshape(d[0], (1,-1)) - sp.reshape(d[0], (-1,1))
                     for d in decompositions]
    filter_matrices = [sp.exp(-pars["comm_tau"]*abs(M)**2) for M in diff_matrices]
    X = sp.zeros_like(matrices[0])
    for M in matrices:
        X += sp.random.randn(1)*M
    X /= sp.linalg.norm(X)
    change = sp.inf
    counter = 0
    eye = sp.eye(X.shape[0])/X.shape[0]

    while counter < pars["comm_iters"] and change > pars["comm_threshold"]:
        X_old = X
        for i in range(len(matrices)):
            U = bases[i]
            U_dg = bases_dg[i]
            filter_matrix = filter_matrices[i]
            # TODO Is there a faster way to do the three next line, with some
            # precomputation? One could at least multiply consecutive Us
            # together.
            X = sp.dot(U_dg, sp.dot(X, U))
            X = X * filter_matrix
            X = sp.dot(U, sp.dot(X, U_dg))
            X = X - eye*sp.trace(X)
            X /= sp.linalg.norm(X)
        change = sp.linalg.norm(X - X_old)
        counter += 1

    # Diagonalize X. Its eigenbasis is the basis in which we will project onto
    # the branches. In an ideal world with perfect records, the spectrum of X
    # would consist of exactly degenerate groups, one group for each block that
    # can be found. In practice these degeneracies will be muddied. To separate
    # out the blocks we constact a matrix called A, such that the amount of
    # non-commutation of the projectors with the matrices we've sampled is
    # measured by the off-diagonal blocks of A. We then search for the division
    # of A into two blocks that minimizes this error measure.
    S, V = sp.linalg.eigh(X)
    matrices_V = [sp.dot(V.conjugate().transpose(), sp.dot(M, V))
                    for M in matrices]
    A = sum(abs(M)**2 for M in matrices_V)
    s1, s2 = find_blocks_in_basis(A)
    P1 = sp.dot(V[:,s1], V[:,s1].conjugate().transpose())
    P2 = sp.dot(V[:,s2], V[:,s2].conjugate().transpose())

    if "R" in pars["system_for_records"]:
        # Now that we have the projectors at the boundary of L and M, find the
        # best ones to match at the boundary of M and R.
        # To do this, "transfer" P1 and P2 through M with the MPS transfer
        # matrix. This gives you two environment tensors, living at i2,
        # representing the two branches. To find the optimal projectors at i2,
        # take the difference of the two environments and diagonalize it.
        # Positive eigenvalues should then belong in one branch, negative ones
        # in the other. To see that this really is the optimal choice, go draw
        # some diagrams for
        # ||P_{1,L} TM - P_{1,L} TM P_{1,R}||^2
        # + ||P_{2,L} TM - P_{2,L} TM P_{2,R}||^2.
        counter = 0
        change = sp.inf
        while (counter < pars["projopt_iters"]
               and change > pars["projopt_threshold"]):
            P1_old, P2_old = P1, P2
            P1, P2, R1, R2 = optimize_branch_projectors(s, i1, i2, P1, P2)
            change = sp.linalg.norm(P1 - P1_old) + sp.linalg.norm(P2 - P2_old)
            counter += 1
        projector_list_R = [R1, R2]
    projector_list_L = [P1, P2]

    if 0 in map(sp.linalg.norm, projector_list_L):
        msg = "Can't branch since projectors are trivial."
        logging.info(msg)
        return [s_orig], [1.0]

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
    # projectors. Note that the optimality of this choice of coefficients
    # already relies on the branches being orthogonal. I'll leave this the way
    # it is for now though, because this is fool-proof and not very costly.
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
    msg = "Found a branch decomposition with local (i1, i2) bond dimensions ({}, {}) and ({}, {}) and coefficients {} and {}.".format(dim_list_L[0], dim_list_R[0], dim_list_L[1], dim_list_R[1], *coeff_list)
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

