# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 17:05:38 2018

@author: Daniel
"""
import numpy as np
import scipy as sp
import scipy.linalg as la
import evoMPS.mps_gen as mp
import evoMPS.tdvp_gen as tdvp
import functools as ft
import copy
import time
from collections import Iterable

def vectorize(s):
    """ Input: evoMPS.mps_gen.EvoMPS_MPS_Generic (in mps_gen.py), object s
    Output: the MPS as a vector in the full physical Hilbert space
    Only possible for small systems, intended for debugging,
    and checking results against exact diagonalization
    """  
    """ s.A[i] is i'th tensor of s, with indices (physical, left bond, right bond)
    """
    
    assert s.q.prod()<=2**12 # too large computation
    
    v=s.A[1].reshape(s.A[1].shape[0],s.A[1].shape[2])
    
    for i in range(2,s.N+1):
        v=np.einsum('ij,kjl->ikl',v,s.A[i])
        v=v.reshape(v.shape[0]*v.shape[1],v.shape[2])
    
    return np.squeeze(v) # was previously just v, not np.squeeze(v), so errors may occur

def mps_from_vector(v,N,q,do_update=True):
    """ Input: vector v
        Output: MPS s, with N sites, and constant physical dimenison q
        Does not truncate bond dimension, but you can use me.truncate_max_D() after
    """
    s = mp.EvoMPS_MPS_Generic(N, np.ones(N+1,dtype=int)*q**(N//2), np.ones(N+1,dtype=int)*q)
    A = np.ndarray(N+1,dtype=np.ndarray)
    U_prev = np.eye(1)
    for i in range(1,N+1):
        U, S, V = la.svd(v.reshape((q**i,-1)), full_matrices=True) # H == U.dot(np.diag(s)).dot(V[0:len(s),:])
        U = U[:,0:len(S)]
        A[i] = np.tensordot(U_prev.conj(),U.reshape((q**(i-1),q,-1)),axes=[0,0]).transpose([1,0,2])
        U_prev=U
    s.A = A
    if do_update:
        s.update()
    """ If updating, global phases may occur """
    return s

def truncate_bonds(s,zero_tol=None,do_update=True):
    """ Input: evoMPS.mps_gen.EvoMPS_MPS_Generic (in mps_gen.py), object s
     No output
     Changes the tensors s.A[i] by truncating bond dimension to remove zero Schmidt values
     Then performs EvoMPS_MPS_Generic.update()
     This method should be used in situations when EvoMPS_MPS_Generic.restore_CF() will not work because
     the bond dimensions of A are too large, otherwise update() will give an error  
    """
    if zero_tol==None:
        zero_tol=1E-6
    for i in range(1,s.N):
        T = np.tensordot(s.A[i].conj(),s.A[i],axes=([0,1],[0,1]))
        T = (T + T.transpose().conj())/2    
        U, S, Vh = la.svd(T)
        num_trunc = sum(S>zero_tol)
        assert num_trunc>0
        U_trunc = U[:,0:num_trunc]
        S_trunc = S[0:num_trunc]
        Vh_trunc = Vh[0:num_trunc,:]
        s.A[i]=np.tensordot(s.A[i],Vh_trunc.conj().transpose(),axes=(2,0))
        s.A[i+1]=np.tensordot(Vh_trunc,s.A[i+1],axes=(1,1)).transpose([1,0,2])
        s.D[i]=num_trunc
        
    for i in range(s.N,1,-1):
        T = np.tensordot(s.A[i].conj(),s.A[i],axes=([0,2],[0,2]))
        T = (T + T.transpose().conj())/2    
        U, S, Vh = la.svd(T)
        num_trunc = sum(S>zero_tol)
        assert num_trunc>0
        U_trunc = U[:,0:num_trunc]
        S_trunc = S[0:num_trunc]
        Vh_trunc = Vh[0:num_trunc,:]    
        s.A[i]=np.tensordot(s.A[i],Vh_trunc.conj().transpose(),axes=(1,0)).transpose([0,2,1])
        s.A[i-1]=np.tensordot(Vh_trunc,s.A[i-1],axes=(1,2)).transpose([1,2,0])
        s.D[i-1]=num_trunc   
    if do_update:
        s.update()


def partial_trace(v,N,ind,complement=False):
    """ Input: State v (as vector, density matrix, or MPS), for N subsystems (infer subsystem size)
        ind: list of indices to trace out (SITES LABELED 1 TO N)
        boolean complement: whether to instead use the complement of ind
        Output: partial trace of v
    """
    if isinstance(v,np.ndarray):
        return partial_trace_vector(v,N,ind,complement)
    else:
        return partial_trace_MPS(v,ind,complement)
    
    
def partial_trace_vector(v,N,ind, complement = False):
    """ Input: vector v (vector or density matrix) as array, for N subsystems (infer subsystem size)
        (If v is 2D square matrix, treat as density matrix)
        ind: list of indices to trace out (sites labeled 1 to N)
        boolean complement: whether to use the complement of ind
        Output: partial trace of v
    """
    indC=list(range(1,N+1))
    for i in sorted(ind, reverse=True):
        del indC[i-1]
    if complement:
        tmp = ind
        ind = indC
        indC = tmp
    
    ind = [i-1 for i in ind] # re-label indices of sites as 0 to N-1
    indC = [i-1 for i in indC]
    q=int(round(max(v.shape)**(1.0/N))) # subsystem dimension
    assert q**N == max(v.shape) # consistent dimensions specified
    
    if np.prod(v.shape) == max(v.shape): # 1D, or 2D with singleton dimension
        v=v.reshape([q]*N)
        rho = np.tensordot(v,v.conj(),(ind,ind))
    else: # 2D, should be square
        rho=v.reshape([q]*(2*N))
        for i in sorted(ind,reverse=True): # reverse order to avoid changing meaning of index
            rho=np.trace(rho,axis1=i,axis2=i+len(rho.shape)/2) # trace out indices

    rho = rho.reshape((q**(len(rho.shape)//2),-1))
    rho = (rho + rho.conj().transpose())/2
    rho = rho/rho.trace()
    return rho

def partial_trace_MPS(s,ind, complement = False):
    """ Input s: evoMPS.mps_gen.EvoMPS_MPS_Generic (in mps_gen.py), for N subsystems (infer subsystem size)
        ind: list of indices to trace out (sites labeled 1 to N)
        boolean complement: whether to use the complement of ind
        Output: partial trace of s
    """
    """ At the moment, only works if the remaining subsystem is contiguous """
    N=s.N
    indC = list(range(1,N+1)) # complement of subset of indices
    for i in sorted(ind, reverse=True):
        del indC[i-1]
    if complement:
        tmp=ind
        ind=indC
        indC=tmp
    if not isinstance(s.l[min(indC)-1],np.ndarray): # Need to handle the different formats that s.l is kept in 
        left =  s.l[min(indC)-1].toarray()
    else:
        left=s.l[min(indC)-1]
    if not isinstance(s.r[max(indC)],np.ndarray):
        right =  s.r[max(indC)].toarray()
    else:
        right = s.r[max(indC)]
        
    # recall the i'th tensor of MPS s is s.A[i], a 3D array with indices (physical, left bond, right bond)
    X=s.A[min(indC)].transpose([1,0,2]) # X now has indices (left bond, physical, right bond)
    for i in range(min(indC)+1,max(indC)+1):
        X = np.tensordot(X,s.A[i],axes=(2,1)).reshape((X.shape[0],-1,s.A[i].shape[2]))
        # physical index expands, and X remains in form (left bond, physical [combined], right bond)
    rho = np.einsum('ijk,lmn,il,kn -> jm',X,X.conj(),left,right)
    
    return rho
    
def schmidt(rho):
    # 2-dim array rho
    # returns eigenvalues of density matrix (shouldn't really be called Schmidt)
    rho=rho/rho.trace()
    rho=(rho+rho.conj().transpose())/2
    return np.sort(np.abs(la.eig(rho)[0]))[::-1]

def entropy(x):
    x=np.array(x)
    x[x<1E-15]=1 # to avoid log(0), this will give right answer because 0*log(0)=1*log(1)=0
    logx=np.log2(x)
    return -sp.sum(x *logx).real

def mutual_information(v,N,ind1,ind2):
    """ Input: vector v as array, for N subsystems (infer subsystem size)
        Two regions indicated by ind1 and ind2, lists of indices
        Output: mutual informaiton of v w.r.t. regions given by ind1, ind2
    """
    """ THIS IS INEFFICIENT BY FACTOR OF ~ 2 
        becaues it should really calculate \rho_A,B from \rho_{AB} 
    """
    """ Can take MPS input, but it calculates the RDM rho, so not most efficient
    """
    """ Use mutual_information_disconn() or mutual_information_middle() for more efficient MPS calculation in those cases """
    S1 = entropy(schmidt(partial_trace(v,N,ind1,complement = True)))
    S2 = entropy(schmidt(partial_trace(v,N,ind2,complement = True)))
    S12 = entropy(schmidt(partial_trace(v,N,list(set(ind1) | set(ind2)),complement = True)))
    return S1+S2-S12

def mutual_information_middle(s,N,i1,i2):
    """ Could just use mutual_information_disconn probably """
    """ Input: evoMPS.mps_gen.EvoMPS_MPS_Generic (in mps_gen.py)
        Calculates mutual information of regions A and B, where A = {1,..,i1-1}, B = {i2+1,...,N}
        sites index (1,..,N)
    """
    """ Assumes s is pure state, otherwise calculation for S12 is wrong """
    if isinstance(s,np.ndarray):
        return mutual_information(s,N,range(1,i1),range(i2+1,N+1))
    
    S1 = s.entropy(i1-1)
    S2 = s.entropy(i2)
    # S12 = entropy(schmidt(partial_trace(s,N,range(i1,i2+1),complement=True)))  
    # Cosntruting pratial_trace() is bad! Use schmidt_MPS_conn_region()
    S12 = entropy(schmidt_MPS_conn_region(s,i1,i2))  
    return S1+S2-S12


def mutual_information_disconn(s,i1,i2,i3,i4, N = None, truncate = None):
    """ Input: evoMPS.mps_gen.EvoMPS_MPS_Generic (in mps_gen.py)
    Calculates the mutual information of the regions [i1,...,i2],[i3,...,i4] (sites labeled from 1 to N)
    Sites i1 < i2 < i3 < i4
    without explicitly calculating the reduced density matrix, just using bond space
    Assumes RCF of s, so call s.update() first (LCF should work also?)
    Also takes vectorized input, in which case needs input N 
    truncate: (integer) number of schmidt vectors to preserve, or None --> no truncation
    """
    if isinstance(s,np.ndarray):
        assert not N==None # Must specify N if input is vector
        return mutual_information(s,N,range(i1,i2+1),range(i3,i4+1))
    
    if not truncate==None:
        new_D=copy.copy(s.D)
        new_D[s.D>truncate]=truncate
        if not all(new_D == s.D):
            s=copy.deepcopy(s)
            s.truncate(new_D)
    
    S12 = entropy(schmidt_MPS_disconn_region(s,i1,i2,i3,i4))
    S1 = entropy(schmidt_MPS_conn_region(s,i1,i2))
    S2 = entropy(schmidt_MPS_conn_region(s,i3,i4))
    return S1 + S2 - S12

def add_MPS(s1,s2,coeffs=[1,1],do_update=True,i1=1,i2='end'):
    """ Input: s1 and s2 are MPS objects (evoMPS.mps_gen.EvoMPS_MPS_Generic, in mps_gen.py)
    coeffs: the respective coefficients for adding s1 and s2 (output updated and normalized regardless)
    i1,i2: start and end of region over which MPS differ (can save time if i1>1 or i2<N) (bond dimensions should otherwise match)
    Output: another MPS object, which is the sum (as states), s3 = s1 + s2
    Creates s3 with at most double the bond dimenison at each bond that's the max bond dimension of s1, s2 at the bond
    Method: doubles bond dimension at each site, and uses trick like the trick to create GHZ state in MPS
    """
    assert s1.N == s2.N
    assert np.allclose(s1.q,s2.q)
    N=s1.N
    if i2=='end':
        i2=N
    # sites labeled 1 to N, A[i] is tensor at site i, D[i] is bond dimension to right of site i
    D = np.empty(N+1,dtype=int)
    A = np.empty(N+1,dtype=np.ndarray)
    q = s1.q
    D[0] = 1
    for i in range(1,N+1):
        if i >= i1 and i<=i2:
            if i==i1:
                D[i]=2*max(s1.D[i],s2.D[i])
                A[i]=np.zeros((q[i],s1.D[i-1],D[i]),dtype='complex')
                A[i][:,:,0:s1.D[i]]=s1.A[i]*coeffs[0]
                A[i][:,:,D[i]//2:D[i]//2+s2.D[i]]=s2.A[i]*coeffs[1]
            elif i==i2:
                D[i]=s1.D[i]
                A[i]=np.zeros((q[i],D[i-1],D[i]),dtype='complex')
                A[i][:,0:s1.D[i-1],:]=s1.A[i]
                A[i][:,D[i-1]//2:D[i-1]//2+s2.D[i-1],:]=s2.A[i]
            else:
                D[i]=2*max(s1.D[i],s2.D[i])
                A[i]=np.zeros((q[i],D[i-1],D[i]),dtype='complex')
                A[i][:,0:s1.D[i-1],0:s1.D[i]]=s1.A[i]
                A[i][:,D[i-1]//2:D[i-1]//2+s2.D[i-1],D[i]//2:D[i]//2+s2.D[i]]=s2.A[i]
        else:
            D[i]=s1.D[i]
            A[i]=s1.A[i]
    s = copy.deepcopy(s1)
    s.D=D
    s.A=A
    truncate_bonds(s,do_update=do_update,zero_tol=1e-13)
    return s

def add_MPS_list(sList,coeffs=None,zero_tol = 1E-2, do_update=True,i1=None,i2=None):
    """ Input: list sList of MPS objects (evoMPS.mps_gen.EvoMPS_MPS_Generic, in mps_gen.py)
    coeffs: the respective coefficients with which to add MPS, length len(sList) (default coefficient 1)
    Output: MPS s with sum
    i1, i2: i1[j] (resp. i2[j]) is the starting (resp. ending) index where \sum_k^j sList[k] and \sum_k^{j+1} sList[k] differ
    """
    N=sList[0].N
    if coeffs == None:
        coeffs = [1]*N
    if isinstance(coeffs, Iterable):
        coeffs = np.array(coeffs)
    coeffs=coeffs/la.norm(coeffs)*np.sqrt(N)
    if i1==None:
        i1=[1]*len(sList)
    if i2==None:
        i2=[N]*len(sList)
    s=None
    for i in range(0,len(sList)):
        if np.abs(coeffs[i])>zero_tol:
            if s==None:
                s=copy.deepcopy(sList[i])
                s.A[i1[i]+1]=s.A[i1[i]+1]*coeffs[i]
            else:
                s = add_MPS(s,sList[i],coeffs=[1,coeffs[i]],do_update=False,i1=i1[i],i2=i2[i])
                # avoid udpating to avoid getting an arbitrary phase

    if do_update:
        s.update() # If this fails, check the zero_tol on truncate_bonds() 
        # and check the the bond dimensions after calling truncate_bonds() in add_MPS()
    return s

def schmidt_MPS_conn_region(s,i1,i2):
    """ Input: evoMPS.mps_gen.EvoMPS_MPS_Generic (in mps_gen.py)
        Calculates the schmidt values of the region [i1,...,i2] of mps s (sites labeled from 1 to N)
        without explicitly calculating the reduced density matrix, just using bond space
        Assumes RCF of s, so call s.update() first (LCF should work also?)
    """
    assert s.D[i1-1]*s.D[i2]<=2**11 # Otherwise takes too much memory
    if not isinstance(s.l[i1-1],np.ndarray): # Need to handle the different formats that s.l is kept in 
        left =  s.l[i1-1].toarray()
    else:
        left=s.l[i1-1]
    if not isinstance(s.r[i2],np.ndarray):
        right =  s.r[i2].toarray()
    else:
        right = s.r[i2]
    left_sqrt = np.sqrt(left) # left and right are diagonal, so gives matrix square root
    right_sqrt = np.sqrt(right)
    AA = np.tensordot(s.A[i1].conj(),s.A[i1],axes=(0,0))
    T = AA
    for i in range(i1+1,i2+1):
        AA = np.tensordot(s.A[i].conj(),s.A[i],axes=(0,0))
        T = np.tensordot(T,AA,axes=([1,3],[0,2])).transpose([0,2,1,3])
    T = np.tensordot(T,right_sqrt,axes=(3,1))
    T = np.tensordot(T,right_sqrt,axes=(1,1)).transpose([0,3,1,2])
    T = np.tensordot(left_sqrt,T,axes=[1,0])
    T = np.tensordot(left_sqrt,T,axes=[1,2]).transpose([1, 2, 0, 3])
    T = T.reshape(T.shape[0]*T.shape[1],-1)
    return np.sort(np.abs(la.eig(T)[0]))[::-1][0:min(T.shape[0],np.prod(s.q[i1:i2+1]))]

    
def schmidt_MPS_disconn_region(s,i1,i2,i3,i4):
    """ Input: evoMPS.mps_gen.EvoMPS_MPS_Generic (in mps_gen.py)
        Calculates the entanglement entropy of the region [i1,...,i2,i3,...,i4] (sites labeled from 1 to N)
        Sites i1 < i2 < i3 < i4
        without explicitly calculating the reduced density matrix, just using bond space
        Assumes RCF of s, so call s.update() first (LCF should work also?)
    """
    """ Requires diagonalizing a D^4 by D^4 matrix, D = bond dimension = s.D """
    args = [i1,i2,i3,i4]
    assert all(args[i] <= args[i+1] for i in range(len(args)-1))
    assert s.D[i1-1]*s.D[i2]*s.D[i3-1]*s.D[i4]<=2**11 # Otherwise takes too much memory

    if i3==i2+1:
        return schmidt_MPS_conn_region(s,i1,i4)
    if not isinstance(s.l[i1-1],np.ndarray): # Need to handle the different formats that s.l is kept in 
        left =  s.l[i1-1].toarray()
    else:
        left=s.l[i1-1]
    if not isinstance(s.r[i4],np.ndarray):
        right =  s.r[i4].toarray()
    else:
        right = s.r[i4]
    
    AA = np.tensordot(s.A[i1].conj(),s.A[i1],axes=(0,0))
    T1 = np.tensordot(left,AA,axes=(1,0))
    for i in range(i1+1,i2+1):
        AA = np.tensordot(s.A[i].conj(),s.A[i],axes=(0,0))
        T1 = np.tensordot(T1,AA,axes=([1,3],[0,2])).transpose([0,2,1,3])
    W =  np.tensordot(s.A[i2+1].conj(),s.A[i2+1],axes=(0,0))
    for i in range(i2+2,i3):
        AA = np.tensordot(s.A[i].conj(),s.A[i],axes=(0,0))
        W = np.tensordot(W,AA,axes=([1,3],[0,2])).transpose([0,2,1,3])
    T2 = np.tensordot(s.A[i3].conj(),s.A[i3],axes=(0,0))
    for i in range(i3+1,i4+1):
        AA = np.tensordot(s.A[i].conj(),s.A[i],axes=(0,0))
        T2 = np.tensordot(T2,AA,axes=([1,3],[0,2])).transpose([0,2,1,3])
    T2 = np.tensordot(T2,right,axes=(3,1))
    W = W.transpose([2,3,0,1])
    T = np.tensordot(T1,W.conj(),axes=(1,0))
    T = np.tensordot(T, T2, axes =(3,0)).transpose([0,3,4,5,1,2,6,7])
    T=T.reshape(np.prod(T.shape[0:4]),-1)
    return np.sort(np.abs(la.eig(T)[0]))[::-1][0:min(T.shape[0],np.prod(s.q[i1:i2+1])*np.prod(s.q[i3:i4+1]))]
            
def angle(a,b):
    return np.arccos(min(np.abs(a.flatten().dot(b.flatten().conj())/(la.norm(a)*la.norm(b))),1))

Sx = sp.array([[0, 1],
                 [1, 0]])
Sy = 1.j * sp.array([[0, -1],
                       [1, 0]])
Sz = sp.array([[1, 0],
                 [0, -1]])
    
def make_Heis_for_MPS(N, J, h):
    """For each term, the indices 0 and 1 are the 'bra' indices for the first and
    second sites and the indices 2 and 3 are the 'ket' indices:
    ham[n][s,t,u,v] = <st|h|uv> (for sites n and n+1)
    """
    ham = -J * (sp.kron(Sx, Sx) + h * sp.kron(Sz, sp.eye(2))).reshape(2, 2, 2, 2)
    ham_end = ham # + h * sp.kron(sp.eye(2), Sz).reshape(2, 2, 2, 2)
    return [None] + [ham] * (N - 2) + [ham_end]

def make_ham_for_MPS(N,hamTerm):
    q = int(round(np.sqrt(hamTerm.shape[0])))
    hamTerm = hamTerm.reshape(q,q,q,q)
    return [None] + [hamTerm]*(N-1)

def kronList(listOfArrays): 
    # tensor product of list (or array) of lists (or arrays)
    return ft.reduce(np.kron,listOfArrays)

def make_ham(hamTerm,N):
    '''Input: 2-qubit Hamiltonian H2
    Output: full Hamiltonian H on N qubits'''
    H=sp.zeros((2**N,2**N),dtype='complex')
    for i in range(0,N-1):
        factors=[sp.eye(2)]*(N-1)
        factors[i]=hamTerm
        H=H+kronList(factors)
    return H
    
def random_hermitian(N):
    X = np.random.randn(N,N) + 1j*np.random.rand(N,N)
    X = (X + X.conj().transpose())/np.sqrt(2)
    return X
    
def make_wavepacket_op(s0,op,x0,DeltaX,p):
    s = copy.deepcopy(s0)
    sList = []
    coeffs = []
    for i in range(1,s.N+1):
        s1 = copy.deepcopy(s0)
        s1.apply_op_1s(op,i,do_update=True)
        coeff = np.exp(-(float(i-x0)/DeltaX)**2)
        coeff = coeff * np.exp(-1.j*float(p)*i/s.N*2*np.pi)
        sList.append(s1)
        coeffs.append(coeff)
    return add_MPS_list(sList,coeffs)

def expect_1s_all(s,op):
    a = np.zeros(s.N)
    for i in range(1,s.N+1):
        a[i-1]=s.expect_1s(op,i).real
    return a

def convert_gen_mps(s):
    """ Convert object evoMPS.tdvp_gen.EvoMPS_TDVP_Generic to evoMPS.mps_gen.EvoMPS_MPS_Generic
    For if you want to lose information about Hamiltonian 
    Or because add_MPS() and update() seem to fail with tdvp object, and are tested with mps_gen object 
    """
    s2 = mp.EvoMPS_MPS_Generic(s.N,s.D,s.q)
    s2.A = copy.deepcopy(s.A)
    s2.update()
    return s2

def convert_tdvp_mps(s,hamTermList):
    """Convert object evoMPS.mps_gen.EvoMPS_MPS_Generic to evoMPS.tdvp_gen.EvoMPS_TDVP_Generic
    with Hamiltonian given by hamTermList
    """
    s1 = tdvp.EvoMPS_TDVP_Generic(s.N, s.D,s.q,hamTermList)
    s1.A = copy.deepcopy(s.A)
    s1.update()
    return s1

def make_trans_inv_mps_from_tensor(A, N,truncate=True):
    q = A.shape[0]
    D = A.shape[1]
    s0 = mp.EvoMPS_MPS_Generic(N, np.ones(N+1,dtype=int)*D, np.ones(N+1,dtype=int)*q)
    for i in range(1,N+1):
        s0.A[i]=copy.copy(A)
    s0.A[1]=s0.A[1][:,0:1,:]
    s0.A[N]=s0.A[N][:,:,0:1]
    if truncate:
        truncate_bonds(s0)
        s0.update()
    return s0

def make_wavepacket_from_tensor(X,Y,N,x0,DeltaX,p, i1=1, i2='end', do_update=True):
    ''' Make wavepacket of form YXXXXX... + XYXXXX... + XXYXXXX ... + ... '''
    ''' Only agrees (and only almost, modulo edge handling) with make_wavepacket_from_tensor_old for i1=2, i2=N-1 '''
    assert X.shape == Y.shape
    q = X.shape[0]
    XD = X.shape[1]
    if i2=='end':
        i2 = N-1
    assert i2<N # for implementation reasons, need i2<N
    A = np.empty(N+1,dtype=np.ndarray)
    D = np.zeros(N+1,dtype=int)
    D[0]=XD # temporary
    for i in range(1,N+1):
        coeff = np.exp(-(float(i-x0)/DeltaX)**2)
        coeff = coeff * np.exp(-1.j*float(p)*i)
        if i<i1 or i>i2+1:
            D[i]=XD
            A[i]=copy.copy(X)
        elif i==i1:
            D[i]=2*XD
            A[i]=np.zeros((q,D[i-1],D[i]),dtype=complex)
            A[i][:,:,:XD]=X
            A[i][:,:,XD:]=Y*coeff
        elif i>i1 and i<=i2:
            D[i]=2*XD
            A[i]=np.zeros((q,D[i-1],D[i]),dtype=complex)
            A[i][:,:XD,:XD]=X
            A[i][:,XD:,XD:]=X
            A[i][:,:XD,XD:]=Y*coeff
        elif i==i2+1:
            D[i]=XD
            A[i]=np.zeros((q,D[i-1],D[i]),dtype=complex)
            A[i][:,:XD,:]=0
            A[i][:,XD:,:]=X
    A[1]=A[1][:,0:1,:]
    A[N]=A[N][:,:,0:1]
    D[0]=1
    D[N]=1
    s = mp.EvoMPS_MPS_Generic(N,D,[q]*(N+1))
    s.D = D
    s.A = A
    if do_update:
        truncate_bonds(s)
    return s

def make_wavepacket_from_tensor_old(A,B,N,x0,DeltaX,p,do_truncate_max_D = True, do_update=True):
    q = A.shape[0]
    D = A.shape[1]
    s0 = make_trans_inv_mps_from_tensor(A,N,truncate=False)
    # dont update s0
    coeffs=[]
    sList=[]
    for i in range(2,N):
        s_add = copy.deepcopy(s0)
        s_add.A[i]=copy.copy(B)
        truncate_bonds(s_add)
        sList.append(s_add)
        coeff = np.exp(-(float(i-x0)/DeltaX)**2)
        coeff = coeff * np.exp(-1.j*float(p)*i)
        coeffs.append(coeff)
    s = add_MPS_list(sList,coeffs=coeffs,do_update=do_update)
    # Even if do_update=False, add_MPS_list calls add_MPS, which calls truncate_bonds()
    if do_truncate_max_D:
        truncate_max_D(s,2*D)
    return s

def update_D(s):
    """ Update the bond dimensions of s.D for MPS s
    Useful when you don't want to fully update the MPS, but want to use s.D """
    for i in range(s.N):
        s.D[i]=s.A[i+1].shape[1]
    s.D[s.N]=s.A[s.N].shape[2]

def truncate_max_D(s,D_max):
    new_D = copy.copy(s.D)
    new_D[new_D>D_max]=D_max
    if not all(new_D == s.D):
        s.truncate(new_D)
        
def join_MPS(s1,s2,n,do_update=True):
    """ Join MPS s1, s2, along site n, with s1 to the left, s2 to the right """
    s3 = copy.deepcopy(s2)
    if not all(s1.D == s2.D):
        new_D = copy.copy(s1.D)
        new_D[s2.D>s1.D]=s2.D[s2.D>s1.D]
        expand_bonds(s1,new_D)
        expand_bonds(s2,new_D)
    for i in range(1,n):
        s3.A[i]=copy.copy(s1.A[i])
    if do_update:
        truncate_bonds(s3)
#        s3.update()
    return s3

def expand_bonds(s,new_D,do_update_D=True):
    """ Expand bond dimensions of s to have bond dimensions new_D """
    """ If D is integer, expand bonds to D, but using appropriately smaller bonds near edges """
    if  isinstance(new_D,int):
        new_D = mp.correct_bond_dim_open_chain(np.ones(s.N+1,dtype='int')*new_D,s.q)
    assert all(new_D>=s.D)
    for i in range(1,s.N):
        A=copy.copy(s.A[i])
        s.A[i]=np.zeros((s.q[i],new_D[i-1],new_D[i]),dtype=complex)
        s.A[i][:,0:s.D[i-1],0:s.D[i]]=A
    if do_update_D:
        update_D(s)
        
def commutant_superoperator(v,dims,system_for_records='C',add_state_projector = True, rho_AC = None):
    """ Vector method, not MPS method """
    """ Given v \in H = H_A \otimes H_B \otimes H_C (as flattened vector)
    dims = (dimA,dimB,dimC)
    system_for_records = 'C' for records on (A,C), system_for_records = 'B' for records on (A,B), system_for_records = 'BC' for records recorded on (A,B) and (A,C)
    returns superoperator C whose kernel is commutant of the algebra on A associated to observables recorded on (A,C)
    """
    dimA, dimB, dimC = dims
    v = v.reshape(dimA,dimB,dimC)
    if system_for_records=='B':
        v = v.transpose([0,2,1])
        dimB, dimC = dimC, dimB
    elif system_for_records=='BC':
        return commutant_superoperator(v,dims,system_for_records='B')+commutant_superoperator(v,dims,system_for_records='C')
    elif not system_for_records=='C':
        print('Unknown input for specification of system argument')

    if rho_AC is None:
        rho_AC = np.tensordot(v,v.conj(),axes=[1,1]).reshape((dimA*dimC,dimA*dimC))
    D,V = la.eigh(rho_AC) # format: H = V D V^+, i.e. H = V.dot(np.diag(D).dot(V.conjugate().transpose()), i.e. V[i,j] = i'th component of j'th eigenvector
    C = np.zeros((dimA**2,dimA**2),dtype='complex') # superoperator from L(A) to L(A)
    for i in range(dimA*dimC):
        for j in range(dimA*dimC):
            v_i = V[:,i].reshape((dimA,dimC)) # i'th eigenvector |v_i> of rho_AC
            v_j = V[:,j].reshape((dimA,dimC))
            X_ij = np.tensordot(v_i,v_j.conj(),axes=[1,1])*np.sqrt(np.abs(D[i]*D[j])) # operator X_ij on A, X_ij = Tr_C(|v_i><v_j|)
            # superoperator C_ij on A, C_ij(Y) = [X_ij,Y]
            # (C_ij)_{(ab)(cd)} = X_{ac}\delta_{bd} - X_{db} \delta_{ca}
            C_ij = np.zeros((dimA,dimA,dimA,dimA),dtype='complex')
            for a in range(dimA):
                for b in range(dimA):
                    for c in range(dimA):
                        for d in range(dimA):
                            C_ij[a,b,c,d]=X_ij[a,c]*(b==d)-X_ij[d,b]*(c==a) 
            C_ij = C_ij.reshape((dimA*dimA,dimA*dimA))
            C = C + np.tensordot(C_ij.conj(),C_ij,(0,0))
    if add_state_projector:
    # Add to C a projector on to L(supp(\rho_A)), so that kernel of C lies in supp(rho_A)
        rho_A = np.tensordot(v,v.conj(),([1,2],[1,2]))
        D,V = la.eigh(rho_A)
        P = np.zeros((dimA**2,dimA**2),dtype='complex') # P is projector onto supp(rho_A)
        # Note P = \sum_{ij} |e_i><e_j|, for eigenvectors |e_i> of \rho_A with nonzero eigenvalue
        tol = 1E-8
        for i in range(dimA):
            for j in range(dimA):
                if D[i] > tol and D[j] > tol:
                    P_ij = np.tensordot(V[:,i],V[:,j].conj(),axes=0).reshape((dimA*dimA))
                    P = P + np.tensordot(P_ij,P_ij.conj(),axes=0)
        C =  C + np.eye(dimA**2)-P
    return C

def find_records(v,dims,system_for_records='C',add_state_projector = True, rho_AC = None, C = None, eps_ker_C = 1E-5):
    """ Vector method, not MPS method """
    """ Given v \in H = H_A \otimes H_B \otimes H_C (as flattened vector)
    dims = (dimA,dimB,dimC)
    system_for_records = 'C' for records on (A,C), system_for_records = 'B' for records on (A,B), system_for_records = 'BC' for records recorded on (A,B) and (A,C)
    returns:
    C: superoperator
    projectorList: list of projectors onto branches
    branchesList: list of branches vectors, with branch amplitude absorbed into normalized state, v = sum(branchesList)
    """
    dimA, dimB, dimC = dims
    eps_ker_C = 1E-5 # tolerance for defining the kernel of C, want 1 >> eps_ker_C >~ eps_noise**2

    if C is None:
        C = commutant_superoperator(v,dims,system_for_records,add_state_projector,rho_AC)
    D,V = la.eigh(C) # format: H = V D V^+, i.e. H = V.dot(np.diag(D).dot(V.conjugate().transpose()), i.e. V[i,j] = i'th component of j'th eigenvector
    # i'th operator in kernel of C is X_i = V[:,i].reshape((dimA,dimA))
    dimKerC = sum(D<eps_ker_C)
    assert dimKerC>0
    X = np.tensordot(V[:,0:dimKerC],np.ones(dimKerC)+np.random.rand(dimKerC),axes=(1,0)).reshape((dimA,dimA)) # random operator in kernel of C
    X = X + X.conj().transpose()
    D,V = la.eigh(X) # format: H = V D V^+, i.e. H = V.dot(np.diag(D).dot(V.conjugate().transpose()), i.e. V[i,j] = i'th component of j'th eigenvector
    projectorList = [] # list of recorded projectors, not including projector onto complement of rho_A
    P = np.zeros((dimA,dimA)) # block of algebra of records on A, i.e. recorded 
    for i in range(dimA):
        if np.abs(D[i]) < 1E-4: # only necessary if add_state_projector = False ?
            continue
        P = np.outer(V[:,i],V[:,i].conj())
        if i==0 or np.abs(D[i]-D[i-1])>1E-2:
            projectorList.append(P)
        else:
            projectorList[-1] = projectorList[-1] + P
    branchList = []
    coeffList = []
    rankList = []
    for i in range(len(projectorList)):
        branch = np.ndarray.flatten(np.tensordot(projectorList[i],v.reshape((dimA,dimB,dimC)),axes=[1,0]))
        branchList.append(branch)
        coeffList.append(la.norm(branch)**2)
        rankList.append(int(np.real(np.round(np.trace(projectorList[i])))))
    return branchList, coeffList, rankList, projectorList, C, D

def copy_mps(s):
    return copy.deepcopy(s)


def find_records_mps(s,i1,i2,system_for_records='BC',eps_ker_C = 1E-5, 
                     coeff_tol = 1e-6, degeneracy_tol = 1E-4, transfer_operator_method = 'low_bond_dimension',
                     max_branch_ratio = 0 ):
    """
        For EvoMPS objects
        Finds records on regions A, B, C, where A = {1,..,i1-1}, B = {i2+1,...,N}
        Inputs:
            system_for_records = 'BC', 'B', or 'C'
            eps_ker_C: tolerance for defining the kernel of C; want 1 >> eps_ker_C >~ eps_noise**2
            coeff_tol: only report branches with coefficient (amplitude squared) greater than coeff_tol
            degeneracy_tol:  tolerance for looking for degenerate eigenvalues of random X in kernel
            transfer_operator_method: see mps_ext.transfer_operator() 
            max_branch_ratio: if the number of branches exceeds this, the smallest-coefficient branches
              are bundled together so that the maximum is not exceeded
        MPS should be updated [s.update()], in right canonical form, or possibly handling left canonical form also
    """     
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
    T = transfer_operator(s,i1,i2,method=transfer_operator_method)
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
    dimKerC = sum(D<eps_ker_C)
#    print('dim ker C: '+str(dimKerC))
    assert dimKerC>0
    X = np.tensordot(V[:,0:dimKerC],np.ones(dimKerC)+np.random.rand(dimKerC),axes=(1,0)).reshape((dimA,dimA)) # random operator in kernel of C
    X = X + X.conj().transpose()
    D,V = la.eigh(X) # format: H = V D V^+, i.e. H = V.dot(np.diag(D).dot(V.conjugate().transpose()), i.e. V[i,j] = i'th component of j'th eigenvector
    
    P = np.zeros((dimA,dimA)) # block of algebra of records on A, i.e. recorded 
    projectorListUntrimmed = [] # list of recorded projectors
    for i in range(dimA):
        if np.abs(D[i]) < 1E-5: # only necessary if state isn't truncated, or...?  Why was this here?
            print('Near-zero eigenvalue of X')
            continue
        P = np.outer(V[:,i],V[:,i].conj())
        if i==0 or np.abs(D[i]-D[i-1])>degeneracy_tol:
            projectorListUntrimmed.append(P)
        else:
            projectorListUntrimmed[-1] = projectorListUntrimmed[-1] + P

    # Calculate the coefficient for each candidate branch and drop the branch if it's smaller than coeff_tol
    coeffListPartiallyTrimmed = []
    projectorListPartiallyTrimmed = []
    for i in range(len(projectorListUntrimmed)):
        coeff = np.abs(np.squeeze(np.tensordot(projectorListUntrimmed[i],left,axes=[(0,1),(0,1)]))) # p_i = Tr(\rho_A P)
        """ The coefficients need to be saved separately, because MPS automatically normalize 
        Note that the projectors should be near-diagonal in Schmidt basis,
            (only near-diagonal due to degeneracies or noise), because records commute with RDM 
        """
        if coeff > coeff_tol:
            projectorListPartiallyTrimmed.append(projectorListUntrimmed[i])
            coeffListPartiallyTrimmed.append(coeff)

    # Order branches from largest to smallest coefficient
    coeffListPartiallyTrimmed, projectorListPartiallyTrimmed = (list(t) for t in zip(*sorted(zip(coeffListPartiallyTrimmed, projectorListPartiallyTrimmed),reverse=True)))
    
    # Set max_num_branches so that it is no larger than len(projectorListPartiallyTrimmed)
    # Note: max_branch_ratio = 0 indicates the user is allowing an unlimited number of branches
    if max_branch_ratio <= 0 or max_branch_ratio >= len(projectorListPartiallyTrimmed):
        max_num_branches = len(projectorListPartiallyTrimmed)
    else:
        max_num_branches = max_branch_ratio

    # Assemble final list of recorded projectors, with those for the smallest coefficents
    # bundled together so the number doesn't exceed max_num_branches
    projectorList = []
    for i in range(len(projectorListPartiallyTrimmed)):
        if i < max_num_branches:
            projectorList.append(projectorListPartiallyTrimmed[i])
        else:
            projectorList[-1] = np.add(projectorList[-1],projectorListPartiallyTrimmed[i])

    # Re-calculating coeffs and then re-sorting projector and coeff lists (largest 
    # to smallest coefficient) because projector bundling may have messed up the order
    coeffList= []
    for i in range(len(projectorList)):
        coeffList.append(np.abs(np.squeeze(np.tensordot(projectorList[i],left,axes=[(0,1),(0,1)])))) # p_i = Tr(\rho_A P)
    coeffList, projectorList = (list(t) for t in zip(*sorted(zip(coeffList, projectorList),reverse=True)))
    
    # Finally, using our final, ordered list of the projectors, construct the list of branches 
    # and their list of ranks
    branchList = []
    rankList=[]
    for i in range(len(projectorList)):
        branch = copy_mps(s)
        branch.A[i1-1] = np.tensordot(branch.A[i1-1],projectorList[i],axes=[2,0])
        branch.update()
        branchList.append(branch)
        rankList.append(int(np.real(np.round(np.trace(projectorList[i])))))
        
    return branchList, coeffList, rankList, projectorList, C, D

def schmidt_vectors_mps(s,n):
    """ Returns the Schmidt vectors (as MPS) of the cut along the n'th bond
        where the n'th bond is between sites (n,n+1), with the first site at n=1
        Assumes EvoMPS object s is updated, in right canonical form, 
            i.e. s.l[n] contains schimdt values squared
        Returns:
            schmidt_vectors: list of Schmidt vectors as updated MPS objects, in order of decreasing Schmidt value
            schmidt_values: list of Schmidt values, in order of decreasing Schmidt values
        The relative phases on the Schmidt vectors will not be correct, due to updating of MPS objects
    """
    if not isinstance(s.l[n],np.ndarray): # Need to handle the different formats that s.l is kept in 
        left =  s.l[n].toarray()
    else:
        left = s.l[n]
    left = np.diag(left)
    assert all(s.check_RCF())
    schmidt_vectors = []
    schmidt_values = []
    for i in range(s.D[n]):
        s2 = copy_mps(s)
        s2.A[n][:,:,0:i]=0
        s2.A[n][:,:,i+1:]=0
        truncate_bonds(s2,zero_tol=1e-10,do_update=True)
        schmidt_vectors.append(s2)
        schmidt_values.append(np.abs(left[i]))
    schmidt_vectors.reverse()
    schmidt_values.reverse()
    return schmidt_vectors,schmidt_values
    

def mps_from_tensors(B_list, make_copy = True):
    """ 
    Used to convert B_list (from TEBD code) to EvoMPS object
    Makes EvoMPS object from list of tensors B_list 
    B_list is 0-indexed list of N tensors, each with 3 indices
    The 3 indices are (physical, left-bond, right-bond)
    """
    N = len(B_list)
    q = B_list[0].shape[0]
    s = mp.EvoMPS_MPS_Generic(N, np.ones(N+1,dtype=int)*1, np.ones(N+1,dtype=int)*q)
    if make_copy:
        s.A[1:]=copy.deepcopy(B_list)
    else:
        s.A[1:]=B_list
    update_D(s)
    truncate_bonds(s) # Only necessary if bond dimensions near edges too large
    # in which case s.update() fails unless you run truncate_bonds() first
    s.update()
    return s

def transfer_operator(s,i1,i2,method='low_bond_dimension'):
    """
        Calculate transfer operator on region {i1,...,i2} for EvoMPS object s
        Return 4-index tensor T, with indices [left-bar, right-bar, left, right],
            for i1 on left and i2 on right
        From transfer operator T, one can form the density matrix 
            rho_AC = T.reshape(T.shape[0]*T.shape[1],-1)
        Options:
            method = 'low_bond_dimension' is better for lower bond dimension, D ~< q^2*constant
                for bond dimension D, physical dimension q
            method = 'high_bond_dimension' is better for higher bond dimension, D >~q^2*constant
                value of constant above depends on machine overhead
    """    
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
    
    AA = np.tensordot(s.A[i1].conj(),s.A[i1],axes=(0,0))
    T = AA # T has indices [left-bar, right-bar, left, right], or also where left~a, right~b
    tic = time.time()
    if method == 'low_bond_dimension':
        for i in range(i1+1,i2+1):
            AA = np.tensordot(s.A[i].conj(),s.A[i],axes=(0,0))
            T = np.tensordot(T,AA,axes=([1,3],[0,2])).transpose([0,2,1,3])
    else:
        for i in range(i1+1,i2+1):
            T = np.tensordot(T,s.A[i],axes=[3,1]) # indices [left-bar, right-bar, left, physical, right]
            T = np.tensordot(T,s.A[i].conj(),axes=[1,1]) # indices [left-bar, left, physical, right, physical-bar, right-bar]
            T = np.trace(T,axis1=2,axis2=4) # indices [left-bar, left, right, right-bar]
            T = T.transpose([0,3,1,2]) #  indices [left-bar, right-bar, left, right]
    T = np.tensordot(T,right_sqrt,axes=(3,1))
    T = np.tensordot(T,right_sqrt,axes=(1,1)).transpose([0,3,1,2])
    T = np.tensordot(left_sqrt,T,axes=[1,0])
    T = np.tensordot(left_sqrt,T,axes=[1,2]).transpose([1, 2, 0, 3]) #  indices [left-bar, right-bar, left, right]
    # The above 4 lines are ineffecient if you know left_sqrt and right_sqrt are diagonal or identity
    # rho_AC = T.reshape(T.shape[0]*T.shape[1],-1)
    return T



