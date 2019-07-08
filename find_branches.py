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
    recurse = len(outer_branches) > 1
    # Filter out branches with tiny coefficients. They will simply be truncated
    # away. Note that we do this after checking whether branching happened at
    # all, so that even if we only keep one branch, we still try recurse on it,
    # since it has changed due to the truncation.
    i = 0
    while i < len(outer_coeffs):
        if abs(outer_coeffs[i]) < pars["coeff_tol"]:
            msg = "Filtering out a branch with a small coefficient ({}).".format(outer_coeffs[i])
            logging.info(msg)
            del(outer_coeffs[i])
            del(outer_branches[i])
        else:
            i += 1
    if recurse:
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

def optimize_branch_projectors(s, i1, i2, P1, P2, at=None):
    assert(at == "i1" or at == "i2")
    P1_env = P1
    P2_env = P2
    if at == "i1":
        positions = reversed(range(i1, i2+1))
        eps_map = eps_r_noop
    else:
        positions = range(i1, i2+1)
        eps_map = eps_l_noop
    for i in positions:
        P1_env = eps_map(P1_env, s.A[i], s.A[i])
        P2_env = eps_map(P2_env, s.A[i], s.A[i])
    env = P1_env - P2_env
    S, V = sp.linalg.eigh(env)
    s1 = S > 0
    s2 = S <= 0
    R1 = sp.dot(V[:,s1], V[:,s1].conjugate().transpose())
    R2 = sp.dot(V[:,s2], V[:,s2].conjugate().transpose())
    d1 = sp.count_nonzero(s1)
    d2 = sp.count_nonzero(s2)
    return R1, R2, d1, d2

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
    - pars["verbosity"] is pretty self-explanatory.
    """
    verb = pars["verbosity"]
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
        if verb > 0:
            msg = "Can't branch because D at i1 is already 1."
            logging.info(msg)
        return [s], [1.0]
    if dimR < 2:
        if verb > 0:
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
            M = M.conjugate().transpose()
            for i in reversed(range(i1, i2+1)):
                state_i = sp.random.randn(s.q[i])
                if s.typ == sp.complex_:
                    state_i = state_i + 1j*sp.random.randn(s.q[i])
                state_i = state_i/sp.linalg.norm(state_i)
                A_i = sp.tensordot(s.A[i], state_i, axes=(0, 0))
                M = sp.dot(A_i, M)
            M = M + M.conjugate().transpose()
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
            ket_state = sp.random.randn(dimR)
            bra_state = sp.random.randn(dimR)
            if s.typ == sp.complex_:
                ket_state = ket_state + 1j*sp.random.randn(dimR)
                bra_state = bra_state + 1j*sp.random.randn(dimR)
            M = sp.outer(ket_state, bra_state)
            M = M + M.conjugate().transpose()
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
    bases_comb_R = [sp.dot(bases_dg[i], bases[i+1])
                    for i in range(len(bases)-1)]
    bases_comb_L = [sp.dot(bases_dg[i+1], bases[i])
                     for i in range(len(bases)-1)]
    diff_matrices = [abs(sp.reshape(d[0], (1,-1))
                         - sp.reshape(d[0], (-1,1)))**2
                     for d in decompositions]
    filter_matrices = [sp.exp(-pars["comm_tau"]*M) for M in diff_matrices]

    def C_func(X):
        U = bases[0]
        U_dg = bases_dg[0]
        filter_matrix = filter_matrices[0]
        X = sp.dot(U_dg, sp.dot(X, U))
        X *= filter_matrix
        for i in range(1, len(matrices)):
            UL = bases_comb_L[i-1]
            UR = bases_comb_R[i-1]
            filter_matrix = filter_matrices[i]
            # TODO Is there a faster way to do the three next line, with some
            # precomputation? One could at least multiply consecutive Us
            # together.
            X = sp.dot(UL, sp.dot(X, UR))
            X *= filter_matrix
        U = bases[-1]
        U_dg = bases_dg[-1]
        X = sp.dot(U, sp.dot(X, U_dg))
        return X

    # An initial guess for the element in the kernel.
    X = sp.zeros_like(matrices[0])
    for M in matrices:
        X += sp.random.randn(1)*M
    eye = sp.eye(X.shape[0])/X.shape[0]
    X -= eye*sp.trace(X)
    X /= sp.linalg.norm(X)

    # To find the element in the kernel, we do a power method combined with a
    # line search: At each step i, we compute C X_i. In a power method, we
    # would simply set X_{i+1} = C X_i. Instead, we take the difference
    # D_i = C X_i - X_i, and search for the optimal alpha, such that the cost
    # function (known as Rayleigh quotient)
    # (X_i + alpha*D)^dagger C (X_i + alpha*D) / ||X_i + alpha*D||
    # is maximized. We then set X_{i+1} = X_i + alpha*D. This was motivated by
    # the slow convergence of the power method and the observation that the
    # steps D that were being taken had very little variance.
    # TODO I have a strong feeling I'm reinventing the wheel here, and probably
    # Lanczos methods as implemented by ARPACK are strictly better than my
    # custom thing. However, ARPACK often simply throws convergence errors
    # because C is a tough matrix, whereas I would always want the best
    # approximation available, regardless of how bad it is. Would there be some
    # way around this issue with ARPACK?
    # TODO Or could we go all in with this approach, and do a polynomial line
    # search? The alphas at consecutive steps are suspiciously similar.
    CX = C_func(X)
    D = CX - X
    change = sp.inf
    counter = 0
    while counter < pars["comm_iters"] and change > pars["comm_threshold"]:
        # We compute the optimal alpha.
        # Note that the following line is the leading order cost of this
        # algorithm; All other steps in an iteration only cost O(D^2), but this
        # one is of course O(D^3).
        CD = C_func(D)
        # Note that vdot automatically complex conjugates the first argument.
        DCD = sp.vdot(D, CD)
        XCX = sp.vdot(X, CX)
        DCX = sp.vdot(D, CX)
        XCD = sp.vdot(X, CD)
        XX = sp.linalg.norm(X)**2
        DD = sp.linalg.norm(D)**2
        XD = sp.vdot(X, D)
        DX = XD.conjugate()
        # The optimal alpha is the root of a second order polynomial. Find
        # the two possible roots, and check which one gives a higher value
        # for the cost function ||C(X+alpha*D)|| / ||X+alpha*D||.
        a = DCD*XD - XCD*DD
        b = DCD*XX + DCX*XD - XCX*DD - XCD*DX
        c = DCX*XX - XCX*DX
        alpha_plus = (-b + sp.sqrt(b**2 - 4*a*c))/(2*a)
        alpha_minus = (-b - sp.sqrt(b**2 - 4*a*c))/(2*a)
        # TOOD Can we get clever about knowing which of the two solutions
        # is the maximum? It's just a rational function after all. Not that
        # this takes long, but just for beauty and simplicity.
        cost_plus = sp.sqrt(abs(
            (XCX + abs(alpha_plus)**2*DCD
             + sp.conj(alpha_plus)*DCX + alpha_plus*XCD)
            /
            (XX + abs(alpha_plus)**2*DD
             + sp.conj(alpha_plus)*DX + alpha_plus*XD)
        ))
        cost_minus = sp.sqrt(abs(
            (XCX + abs(alpha_minus)**2*DCD
             + sp.conj(alpha_minus)*DCX + alpha_minus*XCD)
            /
            (XX + abs(alpha_minus)**2*DD
             + sp.conj(alpha_minus)*DX + alpha_minus*XD)
        ))

        alpha = alpha_plus if cost_plus > cost_minus else alpha_minus

        X_old = deepcopy(X)
        X += alpha*D
        X_trace = sp.trace(X)
        X -= eye*X_trace
        X_norm = sp.linalg.norm(X)
        X /= X_norm
        # C is a linear operator and the identity should be its eigenvector
        # with eigenvalue 1, so we know the following should hold.
        CX = (CX + alpha*CD - eye*X_trace)/X_norm
        D = CX - X

        change = sp.linalg.norm(X - X_old)
        counter += 1
    rayleigh_quotient = sp.vdot(X, CX)
    if verb > 1:
        msg = "Finding an element in the kernel finished in {} iterations and found an element with Rayleigh quotient {}.".format(counter, rayleigh_quotient)
        logging.info(msg)

    # Diagonalize X. Its eigenbasis is the basis in which we will project onto
    # the branches. In an ideal world with perfect records, the spectrum of X
    # would consist of exactly degenerate groups, one group for each block that
    # can be found. In practice these degeneracies will be muddied. To separate
    # out the blocks we constact a matrix called A, such that the amount of
    # non-commutation of the projectors with the matrices we've sampled is
    # measured by the off-diagonal blocks of A. We then search for the division
    # of A into two blocks that minimizes this error measure.
    S, V = sp.linalg.eigh(X)
    if verb > 3:
        msg = "Spectrum of the matrix in the kernel: {}".format(S)
        logging.info(msg)
    matrices_V = [sp.dot(V.conjugate().transpose(), sp.dot(M, V))
                  for M in matrices]
    A = sum(abs(M)**2 for M in matrices_V)
    s1, s2 = find_blocks_in_basis(A)
    P1 = sp.dot(V[:,s1], V[:,s1].conjugate().transpose())
    P2 = sp.dot(V[:,s2], V[:,s2].conjugate().transpose())
    P1d = len(s1)
    P2d = len(s2)
    if verb > 2:
        msg = "Sizes of initial i1 branch projectors: {} & {}".format(P1d, P2d)
        logging.info(msg)

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
            R1, R2, R1d, R2d = optimize_branch_projectors(s, i1, i2, P1, P2,
                                                          at="i2")
            P1, P2, P1d, P2d = optimize_branch_projectors(s, i1, i2, R1, R2,
                                                          at="i1")
            change = sp.linalg.norm(P1 - P1_old) + sp.linalg.norm(P2 - P2_old)
            counter += 1
        projector_list_R = [R1, R2]
        if verb > 1:
            msg = "Iterative optimization of projectors at i1 and i2 finished in {} iterations and dimensions ({}, {}) at i1 and ({}, {}) at i2.".format(counter, P1d, P2d, R1d, R2d)
            logging.info(msg)
    projector_list_L = [P1, P2]

    if 0 in map(sp.linalg.norm, projector_list_L):
        if verb > 0:
            msg = "Can't branch since found projectors are trivial."
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

    # Compute the coefficients the branches should have. Note that we compute
    # the fidelity with respect to s_orig, since updating s may have caused
    # global phase changes.
    # TODO This could be done in a more clever way using canonicality
    # properties of the original MPS/the fact that we know what l and r are and
    # how the branches are related to the original state through virtual
    # projectors. Note that the optimality of this choice of coefficients
    # already relies on the branches being orthogonal. I'll leave this the way
    # it is for now though, because this is fool-proof and not very costly.
    coeffs_complex = [mps_overlap(s_orig, branch) for branch in branch_list]
    coeff_list = [abs(c) for c in coeffs_complex]
    coeff_phases = [c/ac for c, ac in zip(coeffs_complex, coeff_list)]
    for i in range(len(branch_list)):
        branch_list[i].A[1] *= coeff_phases[i]

    if "M" in pars["system_for_records"]:
        dim_list_L = [b.D[i1-1] for b in branch_list]
        dim_list_R = [b.D[i2] for b in branch_list]
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

    if verb > 0:
        if "R" in pars["system_for_records"]:
            msg = "Found a branch decomposition with bond dimensions ({}, {}) at i1 and ({}, {}) at i2, and coefficients {} and {}.".format(P1d, P2d, R1d, R2d, *coeff_list)
        else:
            msg = "Found a branch decomposition with bond dimensions ({}, {}) at i1, and coefficients {} and {}.".format(P1d, P2d, *coeff_list)
        logging.info(msg)
        msg = "Fidelity: {}".format(fid)
        logging.info(msg)
        if "M" in pars["system_for_records"]:
            msg = "Interference on M: {}".format(M_nonint)
            logging.info(msg)

    if 1 - fid > pars["eps_fidelity"]:
        if verb > 0:
            msg = "Rejecting this decomposition due to too low fidelity."
            logging.info(msg)
        # s_orig instead of s, because s.update() may have caused a phase
        # change.
        branch_list = [s_orig]
        coeff_list = [1.0]
    elif "M" in pars["system_for_records"] and M_nonint > pars["eps_M_nonint"]:
        if verb > 0:
            msg = "Rejecting this decomposition due to too high interference on M."
            logging.info(msg)
        # s_orig instead of s, because s.update() may have caused a phase
        # change.
        branch_list = [s_orig]
        coeff_list = [1.0]
    else:
        if verb > 0:
            msg = "Accepting this decomposition."
            logging.info(msg)

    return branch_list, coeff_list

