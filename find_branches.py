import mps_ext

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

