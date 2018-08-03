import mps_ext

def find_branches(s, pars):
    records = mps_ext.find_records_mps(
        s,
        pars["i1"], pars["i2"],
        system_for_records=pars["system_for_records"],
        eps_ker_C=pars["eps_ker_C"], coeff_tol=pars["coeff_tol"],
        degeneracy_tol=pars["degeneracy_tol"],
        transfer_operator_method=pars["transfer_operator_method"]
        )
    branch_list, coeff_list, rank_list, projector_list, C, D = records
    return branch_list, coeff_list

