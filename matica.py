#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Methods for taking an evoMPS object, calculating important info about it, and transcribing to a mathematica file

@author: Jess Riedel
"""


import scipy as sp
import numpy as np
import scipy.linalg as la
import time
import copy

import sys# blindly following this: https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
sys.path.insert(0, '/Users/jriedel/Dropbox/WavefunctionBranch/Code/Jess/evoMPS')

import evoMPS.tdvp_gen as tdvp #Ash's evo code
import mps_ext #Dan's MPS helper, with additions by Jess


mathematica_dict_parameters = {# simulation-wide parameters
    "N":"N",
    "bond_dim":"BondDim",
    "take_off_time":"TakeOffTime",
    "init_cond":"InitCond",
    "hamparam":"Hamparam",
    "mi_bounds":"MIBounds",
    "simparam":"Simparam"
    }
mathematica_dict_data = {# per-timestep data 
    'evo_algo':'EvoAlgo',
    'T':"T",
    'H':"H",
    'H_Heis':"HHeis",
    'mag':"Mag",
    'bond_entropies':"BondEntropies",
    'middle_schmidt_sqs':"MiddleSchmidtSqs",
    'all_schmidt':"AllSchmidt",
    'mi':"MI",
    't_mi':'tMI',
    'spin_cov':"SpinCov",
    'sz_cov':"SzCov",
    'sigma_cov':'SigmaCov',
    'total_spin_cov':"TotalSpinCov", # Next 3 same as last 3, only uses MPO.  Faster to compute, but no single-site index
    'total_sz_cov':"SzCov", 
    'total_sigma_cov':'TotalSigmaCov',
    't_cov':'tCov',
    'mag_cov':'MagCov',
    'coeff':'Coeff'
    }

mathematica_dict = {**mathematica_dict_parameters,**mathematica_dict_data}
data_kinds = list(mathematica_dict_data)# Takes the keys as a list of strings

### 2x2 Pauli matrices ###
Sx_pauli = sp.array([[0,   1],[1,  0]])
Sy_pauli = sp.array([[0, -1j],[1j, 0]])
Sz_pauli = sp.array([[1,   0],[0, -1]])
Id_pauli = sp.array([[1,   0],[0,  1]])
Zr_pauli = sp.array([[0,   0],[0,  0]])


### Single-site operators ###
#we use S and T for the two different spins at each site.
Sx = sp.kron(Sx_pauli, Id_pauli)
Sy = sp.kron(Sy_pauli, Id_pauli)
Sz = sp.kron(Sz_pauli, Id_pauli)
Tx = sp.kron(Id_pauli, Sx_pauli)
Ty = sp.kron(Id_pauli, Sy_pauli)
Tz = sp.kron(Id_pauli, Sz_pauli)
Id = sp.kron(Id_pauli, Id_pauli)
Zr = sp.kron(Zr_pauli, Zr_pauli)
mag_ops = [Sx, Sy, Sz, Tx, Ty, Tz]

### MPO for creating magnon with momentum k ###
def magnon_MPO(op,k,N):
    return [[[op,Id]]] + [[[Id,Zr],[op*sp.exp(1.j*k*n),Id]] for n in range(1,N-1)] + [[[Id],[op*sp.exp(1.j*k*(N-1))]]]

### MPO for computing variance of total spin ###
def MPO_sq_mid_element(op1,op2):
    # #op_ct = op.conj().transpose()
    # diag_el = [[Id, Zr],[op1, Id]]
    # up_rt_el = [[Zr, Zr],[Zr, Zr]]
    # lw_lf_el = [[op2, Zr],[sp.dot(op1,op2), op2]]
    # return [[diag_el, up_rt_el],[diag_el,lw_lf_el]]
    # OK, previously I built this matrix up systematically, but I can't use 
    # the Kron product b/c I need to dot op1 and op2 together, and I need to 
    # combine the virtual indicies, so I'm just doing it by hand
    return ([
        [             Id,  Zr,  Zr, Zr], 
        [            op1,  Id,  Zr, Zr], 
        [            op2,  Zr,  Id, Zr], 
        [sp.dot(op1,op2), op2, op1, Id]
        ])
def MPO_sq_left_element(op1,op2):
    # lf_el = [[sp.dot(op1,op2),op2]]
    # rt_el = [[op1,Id]]
    # return [[lf_el,rt_el]]
    return sp.array([[sp.dot(op1,op2),op2,op1,Id]])
def MPO_sq_right_element(op1,op2):
    # up_el = [[Id],[op1]]
    # lw_el = [[op2],[sp.dot(op1,op2)]]
    # return [[up_el],[lw_el]]
    return sp.array([[Id],[op1],[op2],[sp.dot(op1,op2)]])

mag_MPO_sq_left_elements  = [[MPO_sq_left_element(op1,op2) for op1 in mag_ops] for op2 in mag_ops]
mag_MPO_sq_mid_elements   = [[MPO_sq_mid_element(op1,op2) for op1 in mag_ops] for op2 in mag_ops]
mag_MPO_sq_right_elements = [[MPO_sq_right_element(op1,op2) for op1 in mag_ops] for op2 in mag_ops]


### Hamiltonian ###
#
# A nearest-neighbour Hamiltonian is a sequence of 4-dimensional arrays, one for
# each pair of sites.
# For each term, the indices 0 and 1 are the 'bra' indices for the first and
# second sites and the indices 2 and 3 are the 'ket' indices:
#      ham[n][s,t,u,v] = <st|h|uv> (for sites n and n+1)
#
# A factor of N is *not* necessary to match up with classical model
#
# We need covert h_zy, a single-site Hamiltonian, into two-site Hamiltonian
# We do this by just tensoring the identity to the right or left (h_ZY_right and h_ZY_left)
#
# We use h_ZY_left on almost every site.  But in order to make sure that h_zy is applied
# to *every* site when we have open boundary conditions, we need to add h_ZY_right for the very
# last (right-most) two-site Hamiltonian
qn = 4
h_dbl_heis = (sp.kron(Sx, Sx) + sp.kron(Sy, Sy) + sp.kron(Sz, Sz)   # Double Heisenberg on two sites, each 4 dim
    + sp.kron(Tx, Tx) + sp.kron(Ty, Ty) + sp.kron(Tz, Tz)
    ).reshape(qn, qn, qn, qn)

def ham_heisenberg_ising(J,omega,chi,N): 
    h_i_single = omega*(Sz + Tz + chi*sp.dot(Sx,Tx))                      # Single-site Ising coupling
    h_i_left  = (sp.kron(h_i_single, Id)).reshape(qn, qn, qn, qn)        # Two-site Hamiltonian with h_zy on left
    h_i_right = (sp.kron(Id, h_i_single)).reshape(qn, qn, qn, qn)        # Two-site Hamiltonian with h_zy on right
    return [J*h_dbl_heis + h_i_left]*(N - 1) + [J*h_dbl_heis + h_i_left + h_i_right]



### Data recording ###
# Functions that eat an evoMPS and an existing dataset, append some data about the MPS into the dataset, and then return their runtime

def record_data_from_mps(s, dataset, nn_ops):
    stop_watch = time.time()   
    #mag_ops = [Sx, Sy, Sz, Tx, Ty, Tz]
    for key, op in nn_ops.items():
        dataset[key].append(sp.sum([s.expect_2s(op[n], n) for n in range(1,s.N)]).real)
    dataset['mag'].append([#Expected Magnetization
            [s.expect_1s(ST, n).real for ST in mag_ops]
        for n in range(1, s.N + 1)])
    dataset['bond_entropies'].append([#Entropy between left half (sites 1 through n) and right half (sites n+1 though N)
        s.entropy(n) for n in range(1, s.N)
        ])
    dataset['middle_schmidt_sqs'].append(list(s.schmidt_sq(int(s.N/2)).real)) #squares of the Schmidt coefficients across the bond at the center of the middle of the lattice
    dataset['all_schmidt'].append([list(sp.sqrt(s.schmidt_sq(n).real).real) for n in range(s.N+1)]) #squares of the Schmidt coefficients across the bond at the center of the middle of the lattice

    return time.time()-stop_watch

def record_spin_cov_from_mps(s, t, dataset, thoroughness = "All"):
    stop_watch = time.time()          
    def get_half_cov(op1,op2,n1,n2): 
        if n1 < n2:
            return s.expect_1s_1s(op1,op2,n1,n2).real
        elif n1 == n2:
            return s.expect_1s((sp.dot(op1,op2)+sp.dot(op2,op1))/4,n2).real  # factor of 1/4 because we are symmeterizing *and* we will add matrix to its transpose
        else:
            return 0
    def cov_mat(op1,op2):
        upper_and_half_diag = sp.array([
            [get_half_cov(op1,op2,n1,n2) for n1 in range(1, s.N + 1)]
            for n2 in range(1, s.N + 1)
        ])
        return sp.add(upper_and_half_diag, upper_and_half_diag.transpose()).tolist()
    dataset['t_cov'].append(t)  #this is special subset of times when we calc the spin cov
    dataset['mag_cov'].append([ #this is special subset of magnetizations when we calc the spin cov
            [s.expect_1s(ST, n).real for ST in mag_ops]
        for n in range(1, s.N + 1)])

    # I *think* the following array isn't very costly to construct each time 
    # we want to compute the magnetization covariance because it just copies references.
    # We need to do it here because we won't know N when the constant arrays are defined
    # at the top of this module
    mag_MPO_sq = [( 
       [(
            [mag_MPO_sq_left_elements[i1][i2]] 
                + [mag_MPO_sq_mid_elements[i1][i2] for n in range (1,s.N-1)] 
                + [mag_MPO_sq_right_elements[i1][i2]]
        ) for i1 in range(6)] #6 = len(mag_ops)
    ) for i2 in range(6)]  #6 = len(mag_ops)
    # Note: Currently calculating <(S)^2> takes O(N) time but <sigma_n^z sigma_m^z> 
    # for all n, m takes O(N^3) time.  This is because we use s.expect_1s_1s for
    # the latter, and this scales proportional to the distance between the sites (and
    # because that's calculating N^2 EV's in the first place).  In order to get site-
    # to-site correlation functions in the lattice, you can adapt the function evoMPS
    # uses to calc the MPO expectation value in order to get <sigma_n^z sigma_m^z> 
    # for n=N/2 and m varying over the entire lattice in O(N).  Not implemented yet.

    if thoroughness == "Sz":
        dataset['sz_cov'].append(cov_mat(Sz,Sz))
        dataset['total_sz_cov'].append(s.expect_MPO(mag_MPO_sq[2][2], 1).real) #2 = Sz's position in mag_ops
        # sz_cov_tot = sp.sum(dataset['sz_cov'][-1])
        # print("sz_cov_tot:",sz_cov_tot)
        # print("total_sz_cov:",dataset['total_sz_cov'][-1])
    elif thoroughness == "Total_Sz":
        dataset['total_sz_cov'].append(s.expect_MPO(mag_MPO_sq[2][2], 1).real) #2 = Sz's position in mag_ops
    elif thoroughness == "Sigma":
        dataset['sigma_cov'].append([cov_mat(Sx,Sx),cov_mat(Sy,Sy),cov_mat(Sz,Sz)])
        dataset['sz_cov'].append(dataset['sigma_cov'][-1][2]) # -1 = thing just appended, 2 = Sz's position in mag_ops
        dataset['total_sigma_cov'].append([s.expect_MPO(mag_MPO_sq[r][r], 1).real  for r in range(3)])
        dataset['total_sz_cov'].append(dataset['total_sigma_cov'][-1][2]) # -1 = thing just appended, 2 = Sz's position in mag_ops
    elif thoroughness == "Total_Sigma":
        dataset['total_sigma_cov'].append([s.expect_MPO(mag_MPO_sq[r][r], 1).real  for r in range(3)])
        dataset['total_sz_cov'].append(dataset['total_sigma_cov'][-1][2]) # -1 = thing just appended, 2 = Sz's position in mag_ops
    elif thoroughness == "Full":
        dataset['spin_cov'].append([[cov_mat(op1,op2) for op1 in mag_ops] for op2 in mag_ops])
        dataset['sigma_cov'].append([dataset['spin_cov'][-1][r][r] for r in (0,1,2)])  
        dataset['sz_cov'].append(dataset['spin_cov'][-1][2][2])  
        dataset['total_spin_cov'].append([[s.expect_MPO(mag_MPO_sq[r1][r2], 1).real for r1 in range(6)] for r2 in range(6)])
        dataset['total_sigma_cov'].append([dataset['total_spin_cov'][-1][r][r] for r in range(3)])  #0,1,2 = Sx, Sy, Sz
        dataset['total_sz_cov'].append(dataset['total_spin_cov'][-1][2][2]) #2=Sz
    elif thoroughness == "Total_Full":
        dataset['total_spin_cov'].append([[s.expect_MPO(mag_MPO_sq[r1][r2], 1).real for r1 in range(6)] for r2 in range(6)])
        dataset['total_sigma_cov'].append([dataset['total_spin_cov'][-1][r][r] for r in range(3)])  #0,1,2 = Sx, Sy, Sz
        dataset['total_sz_cov'].append(dataset['total_spin_cov'][-1][2][2]) #2=Sz
    else:
        print ("Shouldn't be here!")
        assert False
    return time.time()-stop_watch

def record_mi_from_mps(s, t, dataset, mi_bounds, truncate=None):
    stop_watch = time.time()          
    miLboundLreg,miRboundLreg,miLboundRreg,miRboundRreg = mi_bounds
    miTypes = len(miLboundLreg)
    dataset['t_mi'].append(t) #this is special subset of times when we calc the spin cov
    dataset['mi'].append([mps_ext.mutual_information_disconn(s,
        miLboundLreg[mt],miRboundLreg[mt],miLboundRreg[mt],miRboundRreg[mt]
        ,truncate=truncate) for mt in range(miTypes)])
    return time.time()-stop_watch

def record_blank_from_mps(dataset):
    for dk in data_kinds:
        dataset[dk].append(None)
    


### Data transcription ###
# Functions used to transcribe collect data to a text file, to be later read out and visualized in Mathematica


def transcribe_data_to_file(output_filename, dataset, transcrip_res = 1, parameters = {}):
    # transcribe_data() opens a file and writes all the data we want to export to Mathematica
    # it uses var_line() as a subroutine.
    assert type(transcrip_res)==int, "transcrip_res isn't an integer"
    output_file = open(output_filename, 'w')
    output_file.write("newPr = <|\n\n")
    for key, val in parameters.items():  
        output_file.write("\"" + mathematica_dict[key] + "\" -> " + var_line(val) + ",\n\n")  # This is a mathematica format for reading in data by text
    # output_file.write("\"transcripRes\" -> " + var_line(transcrip_res) + "\n\n")
    # output_file.write("|>; (* end newPr *) \n\n")
    output_file.write("\"dummy\" -> 0|>; (* end newPr *) \n\n")  #worst possible way to solve an extra-comma problem.  Eat me
    output_file.write("transcripRes = " + str(transcrip_res) + "\n\n")
    output_file.write("newDs = <|\n\n")  
    for key, val in dataset.items(): 

        output_file.write("\"" + mathematica_dict[key] + "\" -> " + var_line(val[0::transcrip_res]) + ",\n\n")  
    output_file.write("\"dummy\" -> 0|>; (* end newDs *)\n\n")
    output_file.close()


# var_line() takes a list/array and trascribes it into a string using the Mathematica syntax with { and } brackets
# It is applied recursively for nested lists
def var_line(var):  
    line = "" # the string that will be returned
    if type(var)==list:
        if len(var)==0:  #it's an empty list
            line += "{}"
        else:  # A non-empty list, so apply recursion
            line += "{"
            for item in var:
                line += var_line(item)
                line += ", "
            line = line[0:-2] #drop the last comma and space
            line += "}"
    else: #it's not a list (end of recursion), so just transcribe the variable into the string
        if type(var)==str:
            line += ("\"{}\"".format(var)) #put quotes around strings so mathematica recognizes them as such
        else:
            line += ("{}".format(var))
    return line 


def find_records_utility_function(s,i1,i2):
    dimA = s.D[i1-1]
    dimC = s.D[i2]
    
    if not isinstance(s.l[i1-1],np.ndarray): # Need to handle the different formats that s.l is kept in 
        left =  s.l[i1-1].toarray()
    else:
        left=s.l[i1-1]
    if not isinstance(s.r[i2],np.ndarray):
        right =  s.r[i2].toarray()
    else:
        right = s.r[i2]
    # in right/left(?) canonical form, matrices left/right are [Schmidt coefficients squared]/identity
    left_sqrt = np.sqrt(np.abs(left))
    right_sqrt = np.sqrt(np.abs(right))
    return dimA, dimC, left, right


#This is a copy of Dan's branch-finding algorithm find_records_mps(), only split up 
# into pieces so it returns (and saves) the diagonalized superoperator
def diagonalize_superoperator(s,i1,i2,system_for_records='BC', transfer_operator_method = 'low_bond_dimension' ):
    """
        For EvoMPS objects
        Finds records on regions A, B, C, where A = {1,..,i1-1}, B = {i2+1,...,N}
        Inputs:
            system_for_records = 'BC', 'B', or 'C'
            eps_ker_C: tolerance for defining the kernel of C; want 1 >> eps_ker_C >~ eps_noise**2
            coeff_tol: only report branches with coefficient (amplitude squared) greater than coeff_tol
            degeneracy_tol:  tolerance for looking for degenerate eigenvalues of random X in kernel
            transfer_operator_method: see mps_ext.transfer_operator() 
        MPS should be updated [s.update()], in right canonical form, or possibly handling left canonical form also
    """     
    dimA, dimC, left, right = find_records_utility_function(s,i1,i2)


    T = mps_ext.transfer_operator(s,i1,i2,method=transfer_operator_method)
    rho_AC = T.reshape(T.shape[0]*T.shape[1],-1)
    
#    tic = time.time()
    assert ('C' in system_for_records or 'B' in system_for_records)
    C = np.zeros([dimA]*4)
    eye_A = np.eye(dimA)
    if 'C' in system_for_records:
        rho_C = np.diag(s.schmidt_sq(i2))
        """ Shouldn't I take the square root of the schmidt_square values to get the right expression above? Idk """
        rho_AC = rho_AC.conj()
        rho_AC = rho_AC.reshape(dimA,dimC,dimA,dimC)
#        C = np.einsum('gdcb,ae,bd -> aceg',rho_AC,eye_A,rho_C)
        C = np.einsum('cg,ae -> aceg',np.einsum('gbcb,b -> cg',rho_AC,np.diag(rho_C)),eye_A)
#        print('Time to calculate C first term: %f'%(time.time()-tic)); tic = time.time()
#        C = C + np.einsum('adeb,cg,bd -> aceg',rho_AC,eye_A,rho_C) # Slower
        C = C + np.einsum('ae,cg -> aceg',np.einsum('abeb,b -> ae',rho_AC,np.diag(rho_C)),eye_A)
#        print('Time to calculate C second term: %f'%(time.time()-tic)); tic = time.time()
#        C = C - 2*np.einsum('abcd,gdeb->aceg',rho_AC,rho_AC) # Slower
        C = C - 2*np.tensordot(rho_AC,rho_AC,axes=[(1,3),(3,1)]).transpose([0,1,3,2])
#        print('Time to calculate C third term: %f'%(time.time()-tic)); tic = time.time()
    if 'B' in system_for_records:
        X = T.conj().transpose([1, 3, 0, 2])
        # Could save constant factor by first calculating contraction of X, X.conj() along first 2 or 3 indices
#        C = C + np.einsum('ijac,ijaf,dg -> cdfg',X.conj(),X,eye_A) # Slower
        C = C + np.tensordot(np.tensordot(X.conj(),X,axes=[(0,1,2),(0,1,2)]),eye_A,axes=0).transpose([0,2,1,3])
#        print('Time to calculate B first term: %f'%(time.time()-tic)); tic = time.time()
#        C = C + np.einsum('ijdb,ijgb,cf -> cdfg',X.conj(),X,eye_A) # Slower
        C = C + np.tensordot(np.tensordot(X.conj(),X,axes=[(0,1,3),(0,1,3)]),eye_A,axes=0).transpose([2,0,3,1])
#        print('Time to B second term: %f'%(time.time()-tic)); tic = time.time()
#        C = C - 2*np.einsum('ijdg,ijcf -> cdfg',X.conj(),X) # Slower
        C = C - 2*np.tensordot(X.conj(),X,axes=[(0,1),(0,1)]).transpose([2,0,3,1])
#        print('Time to B third term: %f'%(time.time()-tic))

    C = C.reshape((dimA*dimA,-1))  

#    tic = time.time()
    D,V = la.eigh(C) # format: H = V D V^+, i.e. H = V.dot(np.diag(D).dot(V.conjugate().transpose()), i.e. V[i,j] = i'th component of j'th eigenvector
    # i'th operator in kernel of C is X_i = V[:,i].reshape((dimA,dimA))
#    print('Time to diagonalize superoperator: %f'%(time.time()-tic))
    return C,D,V



def cut_on_superoperator(s,i1,i2,C,D,V,eps_ker_C = 1E-5, coeff_tol = 1e-6, degeneracy_tol = 1E-4):
    dimA, dimC, left, right = find_records_utility_function(s,i1,i2)

    dimKerC = sum(D<eps_ker_C)
#    print('dim ker C: '+str(dimKerC))
    assert dimKerC>0
    X = np.tensordot(V[:,0:dimKerC],np.ones(dimKerC)+np.random.rand(dimKerC),axes=(1,0)).reshape((dimA,dimA)) # random operator in kernel of C
    X = X + X.conj().transpose()
    D,V = la.eigh(X) # format: H = V D V^+, i.e. H = V.dot(np.diag(D).dot(V.conjugate().transpose()), i.e. V[i,j] = i'th component of j'th eigenvector
    projectorList = [] # list of recorded projectors
    branchList = []
    coeffList=[]
    rankList=[]
    P = np.zeros((dimA,dimA)) # block of algebra of records on A, i.e. recorded 
    for i in range(dimA):
        if np.abs(D[i]) < 1E-5: # only necessary if state isn't truncated, or...?  Why was this here?
            print('Near-zero eigenvalue of X')
            continue
        P = np.outer(V[:,i],V[:,i].conj())
        if i==0 or np.abs(D[i]-D[i-1])>degeneracy_tol:
            projectorList.append(P)
        else:
            projectorList[-1] = projectorList[-1] + P
    projectorListUntrimmed = projectorList
    projectorList = []
    for i in range(len(projectorListUntrimmed)):
        coeff = np.abs(np.squeeze(np.tensordot(projectorListUntrimmed[i],left,axes=[(0,1),(0,1)]))) # p_i = Tr(\rho_A P)
        """ The coefficients need to be saved separately, because MPS automatically normalize 
        Note that the projectors should be near-diagonal in Schmidt basis,
            (only near-diagonal due to degeneracies or noise), because records commute with RDM 
        """
        if coeff > coeff_tol:
            projectorList.append(projectorListUntrimmed[i])
            coeffList.append(coeff)
            branch = mps_ext.copy_mps(s)
            branch.A[i1-1] = np.tensordot(branch.A[i1-1],projectorList[-1],axes=[2,0])
            branch.update()
            branchList.append(branch)
            rankList.append(int(np.real(np.round(np.trace(projectorList[-1])))))
        
    # List branches in order of largest to smallest coefficient
    coeffList, branchList, rankList, projectorList = (list(t) for t in zip(*sorted(zip(coeffList, branchList, rankList, projectorList),reverse=True)))
    return branchList, coeffList, rankList, projectorList, C, D
