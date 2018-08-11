#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
simulating two parallel heisenberg chains coupled by an on-site two-spin transverse-field Ising interaction

@author: Jess Riedel with Ashley Milsted, Dan Ranard, Curt von Keyserlingk, and Markus Hauru
"""

import scipy as sp
import copy
import logging
import evoMPS.tdvp_gen as tdvp  # Ash's evo code
import MPO_tebd as tebd  # Dan and Curt's TEBD methods
import operator as op
from functools import reduce
from scipy.linalg import expm

def is_wellconditioned(s, D_max, tolerance):
    # Check that the full bond dimension is utilized everywhere.
    for n in range(s.N+1):
        left_physical = reduce(op.mul, s.q[1:n], 1)
        right_physical = reduce(op.mul, s.q[n+1:], 1)
        # D_max_n is the maximum dimension this index can have.  It may be
        # dominated by being so close to the left or right end of the chain,
        # that the physical dimension limits the virtual one, or otherwise by
        # D_max.
        D_max_n = min(left_physical, right_physical, D_max)
        if s.D[n] < D_max_n:
            # This index is not at its maximum possible bond dimension.
            return False

    # Check that the Schmidt values have a large enough condition number.
    for n in range(s.N+1):
        schmidts = sp.sqrt(s.schmidt_sq(n).real)
        condition_number = sp.amin(schmidts)/sp.amax(schmidts)
        if condition_number < tolerance:
            return False
    return True


def make_U_bond(TEBD_Ham, N, imag_step_size): 
    d_list = [int(sp.around(sp.sqrt(TEBD_Ham[n].shape[0])))
              for n in range(N-1)]
    u_list = [sp.reshape(expm(-imag_step_size*TEBD_Ham[n]),
                         (d_list[n], d_list[n], d_list[n], d_list[n]))
              for n in range(N-1)]
    return u_list


def take_step_TEBD(s, D_max, TEBD_steps_factor, imag_step_size):
    s.update()
    N = s.N
    B_list = s.A[1:]
    # The list of Hamiltonian terms is reshaped and zero-indexed, to fit the
    # TEBD convention.
    TEBD_ham = [s.ham[n].reshape(s.q[n]*s.q[n+1], s.q[n]*s.q[n+1])
                for n in range(1,N)]
    # We time-evolve by a smaller step_size, to make up for the inaccuarcy of
    # the Suzuki-Trotter decomposition.
    u_list = make_U_bond(TEBD_ham, N, imag_step_size/TEBD_steps_factor)
    # TODO Is this deepcopy necessary?
    s_list = copy.deepcopy([sp.sqrt(s.schmidt_sq(n)) for n in range(N+1)])
    for _ in range(TEBD_steps_factor):
        tebd.tebd(B_list, s_list, u_list, D_max)
    s.set_state_from_tensors(sp.insert(B_list, 0, None), do_update=True)


def evolve_mps(s, pars):
    # The parameters that are needed here.
    TEBD_safety_net = pars["TEBD_safety_net"]
    TEBD_schmidt_tol = pars["TEBD_schmidt_tol"]
    real_step_size = pars["real_step_size"]
    auto_truncate = pars["auto_truncate"]
    TEBD_steps_factor = pars["TEBD_steps_factor"]
    D_max = pars["D_max"]

    s.update(auto_truncate=auto_truncate)
    # We are using Runge-Kutta order 4, but if the MPS is ill-conditioned then
    # fall back to TEBD "safety net".
    if TEBD_safety_net and not is_wellconditioned(s, D_max, TEBD_schmidt_tol):
        logging.info("Taking a TEBD step.")
        take_step_TEBD(s, D_max, TEBD_steps_factor, real_step_size*1.j)
    else:
        logging.info("Taking an RK4 step.")
        s.take_step_RK4(real_step_size*1.j)
    s.update(auto_truncate=auto_truncate)
    return s, real_step_size

