import mps_ext
import logging
from copy import deepcopy
import scipy.sparse.linalg as spsla
import scipy as sp
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
    return total_branches, total_coeffs

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
    s = deepcopy(s)
    s.update()
    for i in range(1, i1):
        shift_canpoint_evomps(s, i, 1)

    A_matrices = []

    if 'M' in pars["system_for_records"]:
        y = sp.eye(dimL)
        for i in range(i1, i2+1):
            y = eps_l_noop(y, s.A[i], s.A[i])
        y = y.conjugate()
        for i in reversed(range(i1, i2+1)):
            y = eps_r_noop(y, s.A[i], s.A[i])
        def C_M_func(X):
            X = sp.reshape(X, (dimL, dimL))
            term1 = sp.dot(X, y)
            term2 = sp.dot(y.conjugate(), X)
            term3 = X
            for i in range(i1, i2+1):
                term3 = eps_l_noop(term3, s.A[i], s.A[i])
            term3 = term3.conjugate()
            for i in reversed(range(i1, i2+1)):
                term3 = eps_r_noop(term3, s.A[i], s.A[i])
            res = term1 + term2 - 2*term3
            res = sp.reshape(res, (dimL, dimL))
            # DEBUG
            print(term1, term2, term3)  # DEBUG
            print(res)
            print()
            input()
            # END DEBUG
            return res
        C_M = spsla.LinearOperator((dimL**2, dimL**2), C_M_func, dtype=s.typ)
        # Sample some product states for the purpose of forming A.
        for j in range(pars["k"]):
            M = sp.eye(dimL, dimL)
            for i in range(i1, i2+1):
                # TODO Should state_i be complex?
                state_i = sp.random.randn(s.q[i]) 
                state_i = state_i/sp.linalg.norm(state_i)
                A_i = sp.tensordot(s.A[i], state_i, axes=(0, 0))
                M = sp.dot(M, A_i)
            M = sp.dot(M, M.conjugate().transpose())
            A_matrices.append(M)

    if 'R' in pars["system_for_records"]:
        matrices = []
        for j in range(pars["k"]):
            # TODO Should r be complex?
            r = sp.random.randn(dimR, dimR) 
            r = sp.dot(r, r.conjugate().transpose())
            r = r/sp.linalg.norm(r)
            for i in reversed(range(i1, i2+1)):
                r = eps_r_noop(r, s.A[i], s.A[i])
            matrices.append(r)
        A_matrices = A_matrices + matrices
        conjugates = [r.conjugate().transpose() for r in matrices]
        squaresL = [sp.dot(rdg, r) for r, rdg in zip(matrices, conjugates)]
        squaresR = [sp.dot(r, rdg) for r, rdg in zip(matrices, conjugates)]
        def C_R_func(X):
            X = sp.reshape(X, (dimL, dimL))
            res = sp.zeros_like(X)
            for i in range(pars["k"]):
                term1 = sp.dot(squaresL[i], X)
                term2 = sp.dot(X, squaresR[i])
                term3 = sp.dot(conjugates[i], sp.dot(X, matrices[i]))
                res += term1 + term2 - 2*term3
            res = sp.reshape(res, (dimL, dimL))
            return res
        C_R = spsla.LinearOperator((dimL**2, dimL**2), C_R_func, dtype=s.typ)

    if pars["system_for_records"] == "R":
        C = C_R
    elif pars["system_for_records"] == "M":
        C = C_M
    else:
        # TODO How should we weigh these two contributions? Should for instance
        # C_R come with a factor that is the ratio of k to the total number of
        # matrices?
        C = C_M + C_R

    # DEBUG
    eyeL = sp.reshape(sp.eye(dimL, dtype=s.typ), (dimL**2,))
    print(sp.linalg.norm(C_M.matvec(eyeL)))
    print(sp.linalg.norm(C_R.matvec(eyeL)))
    print(sp.linalg.norm(C.matvec(eyeL)))
    print(C_M.matvec(eyeL))
    print(C_R.matvec(eyeL))
    input()
    # END DEBUG

    # Find the largest magnitude eigenvalue of C. This will be used to scale C
    # so that all its eigenvalues are negative, making the smallest ones the
    # largest in magnitude.
    # TODO There's probably a way to give a reasonable upper bound for the
    # largest eigenvalue using Frobenius norms, avoiding this eigenvalue
    # search.
    S, U = spsla.eigsh(C, k=1)
    shift = abs(S[0])
    scaling_op = spsla.LinearOperator((dimL**2, dimL**2), lambda x: shift*x)
    C_scaled = C - scaling_op

    # Find the lowest two eigenpairs of C. Two because we are looking for two
    # branches.
    S, U = spsla.eigsh(C_scaled, k=2)
    U_tensor = sp.reshape(U, (dimL, dimL, len(S)))
    S += shift  # Correct for the above shift.
    # DEBUG
    print("Spectrum of C: {}".format(S))
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
    print(X.shape, theta)  # DEBUG
    print("Spectrum of X: {}".format(S))  # DEBUG
    print(U0)
    print(U1)
    # END DEBUG
    A_matrices_V = [sp.dot(V.conjugate().transpose(), sp.dot(M, V))
                    for M in A_matrices]
    # DEBUG
    #[print(abs(M)) for M in A_matrices]
    #print("="*50)
    #[print(abs(M)) for M in A_matrices_V]
    #input()
    # END DEBUG
    A = sum(abs(M)**2 for M in A_matrices_V)
    print(A)  # DEBUG
    corner_sizes = [sp.sum(A[:i,i:]) for i in range(1, dimL)]
    i_min = sp.argmin(corner_sizes)
    s1 = list(range(i_min+1))
    s2 = [i for i in range(dimL) if i not in s1]
    while True:
        candidate_changes = []
        for i in range(dimL):
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
    P1 = sp.dot(V[:,s1], V[:,s1].conjugate().transpose())
    P2 = sp.dot(V[:,s2], V[:,s2].conjugate().transpose())
    projector_list = [P1, P2]
    dim_list = [len(s1), len(s2)]
    # DEBUG
    print(sp.linalg.norm(C_M_func(P1)))
    print(sp.linalg.norm(C_R_func(P1)))
    # END DEBUG

    # Construct the branches.
    branch_list = []
    for i in range(len(projector_list)):
        branch = deepcopy(s)
        branch.A[i1-1] = sp.tensordot(branch.A[i1-1], projector_list[i],
                                      axes=(2,0))
        branch.update(auto_truncate=True)
        branch_list.append(branch)

    # Compute the coefficients the branches should have.
    # TODO This could be done in a more clever way using canonicality
    # properties of the original MPS/the fact that we know what l and r are and
    # how the branches are related to the original state through virtual
    # projectors. Note that the optimality of this choice already relies on the
    # branches being orthogonal. I'll leave this the way it is for now though,
    # because this is fool-proof and not very costly.
    coeff_list = [mps_overlap(s, branch) for branch in branch_list]

    fid = sum(abs(sp.array(coeff_list))**2)
    msg = "Found a branch decomposition with with local bond dimensions {} and {} and coefficients {} and {}, with fidelity {}.".format(*dim_list, *coeff_list, fid)
    logging.info(msg)
    if 1 - fid > pars["eps_fidelity"]:
        msg = "Rejecting this decomposition."
        logging.info(msg)
        branch_list = [s]
        coeff_list = [1.0]
    else:
        msg = "Accepting this decomposition."
        logging.info(msg)

    return branch_list, coeff_list

