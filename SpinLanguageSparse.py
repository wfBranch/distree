import scipy.sparse as sparse
import scipy.sparse.linalg as spalin
import numpy as np
import itertools
from scipy import linalg
import os
import sys
from itertools import chain, combinations
import scipy


# Takes in a binary number b stored as an array.
def bin2dec(b):
    b= np.array(b)[::-1]
    d=0
    for i in range(b.shape[0]):
       d=d+b[i]*2**i
    return d

def dec2bin(d,n):
    a =  [0]*(n - len(list(bin(d)[2:]))) + list(bin(d)[2:])
    return np.array(a,dtype=int)

def powerset(iterable):
  xs = list(iterable)
  # note we return an iterator rather than a list
  return list(chain.from_iterable( combinations(xs,n) for n in range(len(xs)+1) ))

def revbits(x,L):
    return int(bin(x)[2:].zfill(L)[::-1], 2)

def flipbits(x,L):
    dim=2**L
    return dim-x-1


def gen_reflection(L):
    reflectMat = sparse.lil_matrix((2**L, 2**L))
    for i in range(2**L):
        reflectMat[revbits(i,L),i]=1
    return sparse.csr_matrix(reflectMat)
        


def gen_parityProj_xbasis(L):
    dim = 2**L
    Pxp = sparse.lil_matrix((dim/2,dim ))
    Pxm = sparse.lil_matrix((dim/2,dim ))

    for j in range(dim/2):
        Pxp[j,j]=1.0/np.sqrt(2)
        Pxp[j, dim-j-1] = 1.0/np.sqrt(2)
        Pxm[j,j]=1.0/np.sqrt(2)
        Pxm[j, dim-j-1] = -1.0/np.sqrt(2)

    return Pxp, Pxm

def gen_reflectionProj(L):
    dim = 2**L

    dimm= (dim-2**(L/2))/2
    dimp = dimm + 2**(L/2)
    
    Pm = sparse.lil_matrix((dimm,dim))
    Pp = sparse.lil_matrix((dimp,dim))

    dmin={}
    dequal={}
    for i in range(2**L):
        j=revbits(i,L)
        if j!=i:
            dmin[np.min([i,j])]=np.max([i,j])
        else:
            dequal[i]=i
            
    keys = dmin.keys()
    keysEqual = dequal.keys()
    for i in range(dimm):
        k=keys[i]
        Pp[i, k] = 1.0/np.sqrt(2)
        Pp[i, dmin[k]]=1.0/np.sqrt(2)
        Pm[i, k] = 1.0/np.sqrt(2)
        Pm[i, dmin[k]]=-1.0/np.sqrt(2)

    for i in range(2**(L/2)):
        k=keysEqual[i]
        Pp[dimm+i, k]=1.0    

    return sparse.csr_matrix(Pp), sparse.csr_matrix(Pm)


def gen_parityAndReflectionProj(L):
    dim = 2**L

    dimmp = (dim - 2*(2**(L/2)))/4
    dimpp = dimmp + (2**(L/2))/2 + (2**(L/2))/2
    dimpm = dimmp + 2**(L/2)/2
    dimmm = dimmp+ 2**(L/2)/2

    # Ppm = projector onto states with +1 eigenvalue for reflection and -1 for parity etc.
    Ppp = sparse.lil_matrix((dimpp,dim))
    Ppm = sparse.lil_matrix((dimpm,dim))
    Pmp = sparse.lil_matrix((dimmp,dim))
    Pmm = sparse.lil_matrix((dimmm,dim))

    d={}; dRevEqual={}; dParRevEqual={}
    for i in range(2**L):
        j=revbits(i,L); ifl= dim-i-1; jfl=dim-j-1;
        orbit = [i,j, ifl, jfl]
        if i==j:
            dRevEqual[np.min(orbit)] = np.unique(orbit)
        elif j==ifl:
            dParRevEqual[np.min(orbit)] = np.unique(orbit)
        else:
            d[np.min(orbit)]= orbit
            
    k=d.keys(); kRevEqual=dRevEqual.keys(); kParRevEqual=dParRevEqual.keys()
    for i in range(dimmp):
        orbit = d[k[i]]
        for o in range(len(orbit)):
            Ppp[i,orbit[o]] = 0.5
            Ppm[i,orbit[o]] = ((-1)**(o/2))*0.5
            Pmp[i,orbit[o]] = ((-1)**(o))*0.5
            Pmm[i,orbit[o]] = ((-1)**((o+1)/2))*0.5 
        
    for i in range((2**(L/2))/2):
        orbit = dParRevEqual[kParRevEqual[i]]
        for o in range(len(orbit)):
            Ppp[dimmp+i,orbit[o]] = 1/np.sqrt(2)
            Pmm[dimmp+i,orbit[o]] = ((-1)**o)*(1/np.sqrt(2))
            

    for i in range((2**(L/2))/2):
        orbit = dRevEqual[kRevEqual[i]]
        for o in range(len(orbit)):
            Ppp[dimmp+(2**(L/2))/2+i,orbit[o]] = 1/np.sqrt(2)
            Ppm[dimmp+i,orbit[o]] = ((-1)**o)*(1/np.sqrt(2))
            
    return sparse.csr_matrix(Ppp), sparse.csr_matrix(Ppm), sparse.csr_matrix(Pmp), sparse.csr_matrix(Pmm)
        



def diagonalizeWithSymmetries(H, PList,L):
    ind = 0
    Allevecs = np.zeros((2**L, 2**L))
    Allevals = np.zeros((2**L))

    for P in PList:
        H_sym = P*H*(P.T)
        evals, evecs= linalg.eigh(np.real(np.array(H_sym.todense())))
        Allevecs[:, ind:ind+len(evecs)] = np.dot(np.array((P.T).todense()), evecs)
        Allevals[ind:ind+len(evecs)] = evals
        ind = ind+len(evecs)

    ind = np.argsort(Allevals)
    Allevals=Allevals[ind]
    Allevecs=Allevecs[:,ind]

    return Allevals, Allevecs

    

# Because of Sz definition, assumes a basis ordering like 11, 10, 01, 00 for 2 spins 
def gen_spmxyz(L): 
    sx = sparse.csr_matrix([[0., 1.],[1., 0.]]) 
    sy = sparse.csr_matrix([[0.,-1j],[1j,0.]]) 
    sz = sparse.csr_matrix([[1., 0],[0, -1.]])
    sp = sparse.csr_matrix([[0., 1.],[0, 0.]])
    sm = sparse.csr_matrix([[0., 0],[1., 0.]])

    sx_list = [] 
    sy_list = [] 
    sz_list = []
    sp_list = [] 
    sm_list = []
    
    for i_site in range(L): 
        if i_site==0: 
            X=sx 
            Y=sy 
            Z=sz
            P=sp
            M=sm
        else: 
            X= sparse.csr_matrix(np.eye(2)) 
            Y= sparse.csr_matrix(np.eye(2)) 
            Z= sparse.csr_matrix(np.eye(2))
            P= sparse.csr_matrix(np.eye(2))
            M= sparse.csr_matrix(np.eye(2))
            
        for j_site in range(1,L): 
            if j_site==i_site: 
                X=sparse.kron(X,sx, 'csr')
                Y=sparse.kron(Y,sy, 'csr') 
                Z=sparse.kron(Z,sz, 'csr')
                P=sparse.kron(P,sp, 'csr')
                M=sparse.kron(M,sm, 'csr')
            else: 
                X=sparse.kron(X,np.eye(2),'csr') 
                Y=sparse.kron(Y,np.eye(2),'csr') 
                Z=sparse.kron(Z,np.eye(2),'csr') 
                P=sparse.kron(P,np.eye(2), 'csr')
                M=sparse.kron(M,np.eye(2), 'csr')

        sx_list.append(X)
        sy_list.append(Y) 
        sz_list.append(Z) 
        sp_list.append(P) 
        sm_list.append(M) 
    return sp_list, sm_list, sx_list,sy_list,sz_list


def gen_s0sxsysz_generalspin(L,N): 
    dim = int((2*N)+1)
    sp = sparse.csr_matrix((dim,dim))
    sz = sparse.csr_matrix((dim,dim))

    for i in range(dim):
        m= N-i
        if(i>0):
            sp[i-1,i] = np.sqrt((1-(m/N))*(1+((m+1)/N))) 
        sz[i,i] = m/N
    sm = sp.getH()
    sx = 0.5*(sp+sm)
    sy = 0.5*1j*(sm-sp)


    s0_list =[]
    sx_list = [] 
    sy_list = [] 
    sz_list = []
    I = sparse.csr_matrix(np.eye(dim**L))
    for i_site in range(L):
        if i_site==0: 
            X=sx 
            Y=sy 
            Z=sz 
        else: 
            X= sparse.csr_matrix(np.eye(dim)) 
            Y= sparse.csr_matrix(np.eye(dim)) 
            Z= sparse.csr_matrix(np.eye(dim))
            
        for j_site in range(1,L): 
            if j_site==i_site: 
                X=sparse.kron(X,sx, 'csr')
                Y=sparse.kron(Y,sy, 'csr') 
                Z=sparse.kron(Z,sz, 'csr') 
            else: 
                X=sparse.kron(X,np.eye(dim),'csr') 
                Y=sparse.kron(Y,np.eye(dim),'csr') 
                Z=sparse.kron(Z,np.eye(dim),'csr')
        sx_list.append(X)
        sy_list.append(Y) 
        sz_list.append(Z)
        s0_list.append(I)
    return s0_list, sx_list,sy_list,sz_list 
    
def gen_s0sxsysz(L): 
    sx = sparse.csr_matrix([[0., 1.],[1., 0.]]) 
    sy = sparse.csr_matrix([[0.,-1j],[1j,0.]]) 
    sz = sparse.csr_matrix([[1., 0],[0, -1.]])
    s0_list =[]
    sx_list = [] 
    sy_list = [] 
    sz_list = []
    I = sparse.csr_matrix(np.eye(2**L))
    for i_site in range(L):
        if i_site==0: 
            X=sx 
            Y=sy 
            Z=sz 
        else: 
            X= sparse.csr_matrix(np.eye(2)) 
            Y= sparse.csr_matrix(np.eye(2)) 
            Z= sparse.csr_matrix(np.eye(2))
            
        for j_site in range(1,L): 
            if j_site==i_site: 
                X=sparse.kron(X,sx, 'csr')
                Y=sparse.kron(Y,sy, 'csr') 
                Z=sparse.kron(Z,sz, 'csr') 
            else: 
                X=sparse.kron(X,np.eye(2),'csr') 
                Y=sparse.kron(Y,np.eye(2),'csr') 
                Z=sparse.kron(Z,np.eye(2),'csr')
        sx_list.append(X)
        sy_list.append(Y) 
        sz_list.append(Z)
        s0_list.append(I)

    return s0_list, sx_list,sy_list,sz_list 


# efficient way to generate the diagonal Sz lists without doing sparse matrix multiplications
def gen_sz(L):
    sz_list = []
    for i in range(L-1,-1,-1):
        unit =[1]*(2**i)+[-1]*(2**i)
        sz_list.append(np.array(unit*(2**(L-i-1))))
    return sz_list


def gen_op_total(op_list):
    L = len(op_list)
    tot = sparse.csr_matrix((2**L,2**L)) 
    for i in range(L): 
        tot = tot + op_list[i] 
    return tot

def gen_op_prod(op_list):
    L= len(op_list)
    P = op_list[0]
    for i in range(1, L):
        P = P*op_list[i]
    return P

def gen_onsite_field(op_list, h_list):
    L= len(op_list)
    H = sparse.csr_matrix((2**L,2**L)) 
    for i in range(L): 
        H = H + h_list[i]*op_list[i] 
    return H

       
# generates \sum_i O_i O_{i+k} type interactions

def gen_nonloc2bodyalt(op_list, J_list, bc='obc'):
    L= len(op_list)
    H = sparse.csr_matrix((2**L,2**L))
    for i in range(L):
        for j in range(L):
            H = H+ J_list[i,j]*op_list[i]*op_list[j]
    return H    

def gen_nonlocal2body(op_list, J_list, bc='obc'):
    L= len(op_list)
    H = sparse.csr_matrix((2**L,2**L))
    for i in range(L):
        for j in range(i+1,L):
            H = H+ J_list[i,j]*op_list[i]*op_list[j]
    return H    


def gen_interaction_kdist(op_list, k=1, J_list=[], bc='obc'):
    L= len(op_list)
    H = sparse.csr_matrix((2**L,2**L))
    if J_list == []:
        J_list =[1]*L
    Lmax = L if bc == 'pbc' else L-k
    for i in range(Lmax):
        H = H+ J_list[i]*op_list[i]*op_list[np.mod(i+k,L)]
    return H        
    
def gen_nn_int(op_list, J_list=[], bc='obc'):
    return gen_interaction_kdist(op_list, 1, J_list, bc)

##def gen_lr_int(op_list, alpha, bc='obc'):
##    L= len(op_list)
##    r = np.arange(L)
##    rMat = abs(r[:,np.newaxis] - r[np.newaxis,:])
##    if bc =='pbc':
##        rMat = np.minimum(rMat, L-rMat)
##    iu1 = np.triu_indices(L,1)
##    rMat = 1.0/(rMat[iu1]**alpha)
##
##    op_list = np.array(op_list)
##    opMat = op_list[:,np.newaxis]*op_list[np.newaxis,:]
##    return np.dot(opMat[iu1], rMat)
    
def gen_lr_int(op_list, alpha, bc='obc'):
    L= len(op_list)
    interaction=0*op_list[0]
    for i in range(L):
        for j in range(i+1,L):
            r=1.0*abs(i-j)
            if bc=='pbc':
                r=np.min([r, L-r])
            interaction = interaction+ (op_list[i]*op_list[j])/(r**alpha)
    return interaction
 
            
        

def gen_diag_projector(symMatrix, symValue):
    symMatrix = symMatrix.diagonal()
    ind = np.where(symMatrix==symValue)
    dim = len(symMatrix)
    dim0 = np.size(ind)
    P = sparse.lil_matrix((dim0,dim ))
    for i in range(np.size(ind)):
        P[i,ind[0][i]] = 1.0
    return P


def projectedStateNum_ToState(symMatrix, symValue, psnum):
    symMatrixDense = symMatrix.todense()
    ind = np.where(np.diag(symMatrixDense)==symValue)
    dimOrig = len(symMatrixDense)
    
    return dec2bin(dimOrig - 1 -ind[0][psnum], int(np.log2(dimOrig)))
    
    
def gen_state_bloch(thetaList, phiList):
    L=len(thetaList)
    psi = np.kron([np.cos(thetaList[0]/2.),np.exp(1j*phiList[0])*np.sin(thetaList[0]/2.)],
                  [np.cos(thetaList[1]/2.),np.exp(1j*phiList[1])*np.sin(thetaList[1]/2.)])
    for i in range(2,L):
        psi = np.kron(psi, [np.cos(thetaList[i]/2.),np.exp(1j*phiList[i])*np.sin(thetaList[i]/2.)])
    return psi



def isdiag(M):
    if M.ndim!=2:
        return False
    return np.all(M == np.diag(np.diag(M)))

def gen_unifstate_bloch(theta, phi,L):
    return gen_state_bloch([theta]*L, [phi]*L)

def gen_U(H_sparse):
    H = np.array(H_sparse.todense())
    if isdiag(H):
        return np.diag(np.exp(-1j*np.diag(H)))
    return linalg.expm(-1j*H)

def gen_diagonalEnsemble(psi0, evecs,op):
    psi0Init = np.dot(np.conj(evecs.T), psi0)
    dim= len(evecs)
    OPMat = gen_diagonal_ME(op,evecs)
    return np.dot(OPMat, abs(psi0Init)**2)


def EntanglementEntropy_2Cuts(state, c1, c2):
    L = int(np.log2(len(state)))
    Cij = np.reshape(state, (2**c1, 2**(L-c1-c2), 2**c2))
    Cij = np.transpose(Cij, axes = (1,0,2))
    Cij = np.reshape(Cij, (2**(L-c1-c2), 2**(c1+c2)))
    S = np.linalg.svd(Cij, full_matrices = 0, compute_uv = 0)
    S = S[S>(10**-15)]
    return -np.sum((S**2)*np.log(S**2))

def EntanglementEntropy(state, cut_x):
    dim = len(state)
    #print(state.dot(np.conjugate(state)))
    L = int(np.log2(dim))
    Cij = np.reshape(state, (2**cut_x, 2**(L-cut_x)))
    S = np.linalg.svd(Cij, full_matrices = 0, compute_uv = 0)
    #print(S)
    Sorig = S
    S = S[S>(10**-15)]
    # if(-np.sum((S**2)*np.log(S**2))<0):
    #     print("error, schmidt spec",Sorig)
    return -np.sum((S**2)*np.log(S**2))

def MutualInformation(state, n):
    " Mutual information between left and rightmost n sites in state"
    L = int(np.log2(len(state)))
    SA = EntanglementEntropy(state, n)
    SB = EntanglementEntropy(state, L-n)
    SAB = EntanglementEntropy_2Cuts(state, n, n)
    #print(SA,SB,SAB)
    return SA+SB-SAB

def LevelStatistics(energySpec, nbins = 25):
    delta = energySpec[1:] -energySpec[0:-1]
    r = map(lambda x,y: min(x,y)*1.0/max(x,y), delta[1:], delta[0:-1])
    return np.mean(r)

"""returns np.dot(A,B) with speedups for diagonal matrices. """
def mydot(A, B):
    if isdiag(A):
        if isdiag(B):
            return A*B
        return (np.diag(A)*(B.T)).T
    if isdiag(B):
        return A*np.diag(B)
    else:
        return np.dot(A,B)
        
        
def ME(op, state):
    return mydot(mydot(np.conj(state), op),state)

def expct_val(op,state,evecs,op_basis="ebasis"):
    if op_basis=="ebasis":
        return ME(op,state)
    elif op_basis == "phys":
        return ME(gen_allME(op,evecs),state)

"returns a list \langle \alpha |O|\alpha \rangle for all eigenvectors \alpha"
def gen_diagonal_ME(op, evecs):
    return np.sum((np.conj(evecs))*mydot(op, evecs),0)

def gen_allME(op, evecs):
    return mydot(mydot(np.conj(evecs.T),op), evecs)

def timeEvolutionStrobos(evals, evecs, psi0, OpList, nPeriods):
    nops = len(OpList)
    Times=np.arange(0,nPeriods)
    freq = 2*np.pi*np.fft.fftfreq(Times.shape[-1]);freq = freq+2*np.pi*(freq<0)

    OpTimesList = np.zeros((nops, nPeriods))
    OpFourierList = np.zeros((nops, nPeriods),'complex')

    psi0Init = np.dot(np.conj(evecs.T), psi0)

    OpMatList = np.zeros(nops, 'object')
    for i in range(nops):
        OpMatList[i] = gen_allME(OpList[i], evecs)

    for i in range(len(Times)):
        t=Times[i]
        psi = psi0Init*np.exp(-1j*evals*t)
        for j in range(nops):
            OpTimesList[j, i] = ME(OpMatList[j], psi)

    for j in range(nops):
        OpFourierList[j] = np.fft.fft(OpTimesList[j])

    return OpTimesList, OpFourierList

def timeEvolutionMIStrobos(evals, evecs, psi0,n, nPeriods):
    Times=np.arange(0,nPeriods)

    MITimesList = np.zeros(nPeriods)

    psieigb = np.dot(np.conjugate(evecs.T), psi0)
    #print("phieigb norm",psieigb.dot(np.conjugate(psieigb)))

    for i in range(len(Times)):
        
        #print("phieigb norm",psieigb.dot(np.conjugate(psieigb)))
        psiphys=np.dot(evecs, psieigb)
        #print("phiphys norm",psiphys.dot(np.conjugate(psiphys)))
        MITimesList[i]=MutualInformation(psiphys, n)
        t=Times[i]
        psieigb = psieigb*np.exp(-1j*evals)
        

    return MITimesList

def timeEvolutionEntropyStrobos(evals, evecs, psi0,n, nPeriods,cut_x):
    Times=np.arange(0,nPeriods)

    HalfEntropyTimesList = np.zeros(nPeriods)

    psieigb = np.dot(np.conjugate(evecs.T), psi0)
#print("phieigb norm",psieigb.dot(np.conjugate(psieigb)))


    for i in range(len(Times)):
        t=Times[i]
        psieigb = psieigb*np.exp(-1j*evals)
        #print("phieigb norm",psieigb.dot(np.conjugate(psieigb)))
        psiphys=np.dot(evecs, psieigb)
        #print("phiphys norm",psiphys.dot(np.conjugate(psiphys)))
        HalfEntropyTimesList[i]=EntanglementEntropy(psiphys, cut_x)

    return HalfEntropyTimesList




def hello():
    print("hell")

def timeEvolution(evals, evecs, psi0, OpList, Times):
    nops = len(OpList)
    OpTimesList = np.zeros((nops, len(Times)))
    psi0Init = np.dot(np.conj(evecs.T), psi0)

    OpMatList = np.zeros(nops, 'object')
    for i in range(nops):
        OpMatList[i] = gen_allME(OpList[i], evecs)

    for i in range(len(Times)):
        t=Times[i]
        psi = psi0Init*np.exp(-1j*evals*t)
        for j in range(nops):
            OpTimesList[j, i] = ME(OpMatList[j], psi)

    return OpTimesList

# converts the eigenvalues of U(T) -- phases -- to quasienergies       
def phaseToQuasienergy(evals, evecs=[]):
    evals = (np.real(1j*np.log(evals)))
    indSort = np.argsort(evals)
    evals = evals[indSort]
    if evecs == []:
        return evals
    evecs = evecs[:, indSort]
    return evals, evecs

    
def mag(i):
    binstring = bin(i)[2:]
    return np.sum([int(x) for x in binstring])

# a program that takes a physical state, and histograms it by total S^z value
def Szhistogram(psi_phys):
    #the state is assumed to be written in Sz basis
    L=int(np.log2(psi_phys.size))
    Szhistogramloc=np.zeros(L+1)
    for i in range(2**L):
        #magnetization for binary rep of i
        Szhistogramloc[mag(i)]+=abs(psi_phys[i])**2
    
    return Szhistogramloc


def timeEvolutionSzHistogramStrobos(evals, evecs, psi0,nPeriods):
    Times=np.arange(0,nPeriods)
    L=int(np.log2(psi0.size))

    SzHistogramTimesList = np.zeros((L+1,nPeriods))

    psieigb = np.dot(np.conjugate(evecs.T), psi0)
#print("phieigb norm",psieigb.dot(np.conjugate(psieigb)))

    psiphys=psi0
    for i in range(len(Times)):
        t=Times[i]
        SzHistogramTimesList[:,i]=Szhistogram(psiphys)
        psieigb = psieigb*np.exp(-1j*evals)
        #print("phieigb norm",psieigb.dot(np.conjugate(psieigb)))
        psiphys=np.dot(evecs, psieigb)
        #print("phiphys norm",psiphys.dot(np.conjugate(psiphys)))
        # print(Szhistogram(psiphys))
        # print(SzHistogramTimesList)
        

    return SzHistogramTimesList


def eigenspaceindices(D,tol=1e-10):
    O = np.array(D)
    espaces=[[0]]
    uniqueval=[O[0]]

    for i in range(1,len(O)):
        if abs(O[i]-uniqueval[-1])>tol:
            espaces.append([i])
            uniqueval.append(O[i])

        else:
            (espaces[-1]).append(i)

    return espaces,uniqueval
def histogram(O,psi,tol=1e-2):

        
    evals,evecs=linalg.eigh(O)
    arg=np.argsort(evals)
    evals = evals[arg]
    evecs = (evecs[:,arg])

    psinewbasis = np.dot(np.conjugate(evecs.T),psi)
    espaces,uniquevals = eigenspaceindices(evals)
    # print(espaces,uniquevals)
    histrange=len(uniquevals)

    hist = np.zeros(histrange)

    for i in range(histrange):
        for j in espaces[i]:
            hist[i]+=(np.abs(psinewbasis[j])**2)
            
    return hist,uniquevals,espaces

def histogramalt(Ovals,Ovecs,psi,tol=1e-10):
  
    psinewbasis = np.dot(np.conjugate(Ovecs.T),psi)
    espaces,uniquevals = eigenspaceindices(Ovals)
    # print(espaces,uniquevals)
    histrange=len(uniquevals)

    hist = np.zeros(histrange)

    for i in range(histrange):
        for j in espaces[i]:
            hist[i]+=(np.abs(psinewbasis[j])**2)
            
    return hist

def timeEvolutionOHistogramStrobos(O,evals, evecs, psi0,nPeriods,tol=1e-10):
    Times=np.arange(0,nPeriods)
    
    Ovals,Ovecs=linalg.eigh(O)
    arg=np.argsort(-Ovals)
    Ovals = Ovals[arg]
    Ovecs = (Ovecs[:,arg])
    Oespaces,Ouniquevals = eigenspaceindices(Ovals)

    OHistogramTimesList = []

    psieigb = np.dot(np.conjugate(evecs.T), psi0)
#print("phieigb norm",psieigb.dot(np.conjugate(psieigb)))
    psiphys=psi0
    for i in range(len(Times)):
        t=Times[i]
        OHistogramTimesList.append(histogramalt(Ovals,Ovecs,psiphys,tol))
        psieigb = psieigb*np.exp(-1j*evals)
        #print("phieigb norm",psieigb.dot(np.conjugate(psieigb)))
        psiphys=np.dot(evecs, psieigb)
        #print("phiphys norm",psiphys.dot(np.conjugate(psiphys)))
        # print(Szhistogram(psiphys))
        # print(SzHistogramTimesList)
        
        OHistogramTimesList1=np.array(OHistogramTimesList)
    return OHistogramTimesList1.T,Oespaces,Ouniquevals


def dpage(nA,ntot):
    hA = 2**nA
    hcom= 2**(ntot-nA)
    htot  = 2**ntot
    
    return np.sum([(1.0/j) for j in range(hcom+1,htot+1)]) - ((hA-1)/(2*hcom))

def dpageapprox(nA,ntot):
    hA = 2**nA
    hcom= 2**(ntot-nA)
    htot  = 2**ntot
    
    return nA - (hA/(2*hcom))

def dpageMI(nA,ntot):
    return dpage(nA,ntot)+dpage(nA,ntot)- dpage(2*nA,ntot)
def dpageMIapprox(nA,ntot):
    return dpageapprox(nA,ntot)+dpageapprox(nA,ntot)- dpageapprox(2*nA,ntot)

def EntanglementEntropy_2Cuts_variablehs(state, c1, c2,dim_list):
    L = len(dim_list)
    h1 = np.prod(dim_list[:c1])
    hm = np.prod(dim_list[c1:L-c2])
    h2 = np.prod(dim_list[L-c2:L])
    
    Cij = np.reshape(state, (h1, hm, h2))
    Cij = np.transpose(Cij, axes = (1,0,2))
    Cij = np.reshape(Cij, (hm, h1*h2))
    S = np.linalg.svd(Cij, full_matrices = 0, compute_uv = 0)
    S = S[S>(10**-15)]
    return -np.sum((S**2)*np.log(S**2))

def EntanglementEntropy_2Cuts_variablehs_ALT(state, c1, c2,dim_list):
    L = len(dim_list)
    h1 = np.prod(dim_list[:c1])
    h2 = np.prod(dim_list[c1:c2])
    h3 = np.prod(dim_list[c2:])
    
    Cij = np.reshape(state, (h1, h2, h3))
    Cij = np.transpose(Cij, axes = (1,0,2))
    Cij = np.reshape(Cij, (h2, h1*h3))
    S = np.linalg.svd(Cij, full_matrices = 0, compute_uv = 0)
    S = S[S>(10**-15)]
    return -np.sum((S**2)*np.log(S**2))


def EntanglementEntropy_4Cuts_variablehs(state, c1, c2,c3,c4,dim_list):
    L = len(dim_list)
    h2 = np.prod(dim_list[c1:c2])
    h3 = np.prod(dim_list[c2:c3])
    h4 =  np.prod(dim_list[c3:c4])
    h5 = np.prod(dim_list[c4:])
    
    Cij = np.reshape(state, ( h2, h3,h4,h5))
    Cij = np.transpose(Cij, axes = (0,2,1,3))
    Cij = np.reshape(Cij, (h2*h4, h3*h5))
    S = np.linalg.svd(Cij, full_matrices = 0, compute_uv = 0)
    S = S[S>(10**-15)]
    return -np.sum((S**2)*np.log(S**2))

# def EntanglementEntropy_variablehs(state, cut_x,dim_list):
#     L = len(dim_list)
#     h1 = np.prod(dim_list[:cut_x])
#     h2 = np.prod(dim_list[cut_x:])
#     # print("h1,2=",h1,h2)
#     #print(state.dot(np.conjugate(state)))
#     Cij = np.reshape(state, (h1, h2))
#     S = np.linalg.svd(Cij, full_matrices = 0, compute_uv = 0)
#     # print("SVD=",S)
#     S = S[S>(10**-15)]
#     # if(-np.sum((S**2)*np.log(S**2))<0):
#     #     print("error, schmidt spec",Sorig)
#     return -np.sum((S**2)*np.log(S**2))




# def MutualInformation_variablehs(state, cut1,cut2 ,dim_list):
#     " Mutual information between left and rightmost n sites in state"
#     L = len(dim_list)
#     SA = EntanglementEntropy_variablehs(state, cut1,dim_list)
#     SB = EntanglementEntropy_variablehs(state, cut2,dim_list)
#     SAB = EntanglementEntropy_2Cuts_variablehs_ALT(state, cut1,cut2,dim_list)
#     # print("SA,SB,SAB=",SA,SB,SAB)
#     return SA+SB-SAB




def gen_state_list(m_list,spin_list):
    #this is specific to the two site problem.
    dim_list=[int(2*i+1) for i in spin_list]
    print(dim_list)
    L = len(m_list)
 
    i_list = [int(spin_list[j]-m_list[j]) for j in range(L)]
    psi = np.eye(1,dim_list[0],i_list[0])
    print(psi)
    for j in range(1,L):
        psi=np.kron(psi, np.eye(1,dim_list[j],i_list[j]))
    return psi[0]


def gen_s0sxsyszp_down(spin_list):
    dim_list = [int(2*i+1) for i in spin_list]
    L = len(spin_list)


    unique_spin = set(spin_list)

    oper_dict = {}
    for spin in unique_spin:
        dim = int(2*spin+1)
        sp = sparse.csr_matrix((dim,dim))
        sz = sparse.csr_matrix((dim,dim))
        p_down = sparse.csr_matrix((dim,dim)) #projects onto spin down
        oper_dict_dim= {}

        for i in range(dim):
            m= spin-i
            if(i>0):
                sp[i-1,i] = np.sqrt((1-(m/spin))*(1+((m+1)/spin)))   
            sz[i,i] = m/spin
        p_down[dim-1,dim-1]=1.0    
        sm = sp.getH()
        sx = 0.5*(sp+sm)
        sy = 0.5*1j*(sm-sp)
        oper_dict_dim["x"]=sx
        oper_dict_dim["y"]=sy
        oper_dict_dim["z"]=sz
        oper_dict_dim["p_down"]=p_down

        oper_dict[str(dim)]=oper_dict_dim
   

    s0_list =[]
    sx_list = [] 
    sy_list = [] 
    sz_list = []
    p_down_list = []
    I = sparse.csr_matrix(np.eye(np.prod(dim_list)))
    for i_site in range(L):
        dim_i = dim_list[i_site]
        dim_0 = dim_list[0]
        if i_site==0: 
            X=oper_dict[str(dim_0)]["x"] 
            Y=oper_dict[str(dim_0)]["y"]
            Z=oper_dict[str(dim_0)]["z"]
            P_DOWN=oper_dict[str(dim_0)]["p_down"]

        else: 
            X= sparse.csr_matrix(np.eye(dim_0)) 
            Y= sparse.csr_matrix(np.eye(dim_0)) 
            Z= sparse.csr_matrix(np.eye(dim_0))
            P_DOWN = sparse.csr_matrix(np.eye(dim_0))

        for j_site in range(1,L): 
            dim_j = dim_list[j_site]

            if j_site==i_site: 
                X=sparse.kron(X,oper_dict[str(dim_j)]["x"], 'csr')
                Y=sparse.kron(Y,oper_dict[str(dim_j)]["y"], 'csr') 
                Z=sparse.kron(Z,oper_dict[str(dim_j)]["z"], 'csr') 
                P_DOWN = sparse.kron(P_DOWN,oper_dict[str(dim_j)]["p_down"], 'csr') 
            else: 
                X=sparse.kron(X,np.eye(dim_j),'csr') 
                Y=sparse.kron(Y,np.eye(dim_j),'csr') 
                Z=sparse.kron(Z,np.eye(dim_j),'csr')
                P_DOWN=sparse.kron(P_DOWN,np.eye(dim_j),'csr')
        sx_list.append(X)
        sy_list.append(Y) 
        sz_list.append(Z)
        p_down_list.append(P_DOWN)
        s0_list.append(I)
    return  s0_list, sx_list,sy_list,sz_list,p_down_list




#Generate a y rotation
def gen_yrot_spin1(L,theta):
    dim = 3
    c=np.cos(theta)
    s=np.sin(theta)
    u=0.5*(1+c)
    z = s/np.sqrt(2)
    roty = sparse.csr_matrix([[u,-z,1-u],[z,c,-z],[1-u,z,u]])
    rot_list =[]
    for i_site in range(L):
        if i_site==0: 
            R=roty
        else: 
            R= sparse.csr_matrix(np.eye(3))

        for j_site in range(1,L): 
            if j_site==i_site: 
                R=sparse.kron(R,roty, 'csr') 
            else: 
                R=sparse.kron(R,np.eye(3),'csr') 
        rot_list.append(R)
    return rot_list
    
def EntanglementEntropy_variablehs(state, sites, dim_list):

#The sites (in range 0,..., L-1 ) in the entanglement region

    L = len(dim_list)
    sites_compl=[]
    
    if len(sites)==L or len(sites)==0:
        return 0.0
    
    for j in range(L):
        if not(j in sites):
            sites_compl.append(j)
    dsites = np.prod([dim_list[i] for i in sites])
    dsites_compl = np.prod([dim_list[i] for i in sites_compl])
    
    print(sites_compl)
    # print("h1,2=",h1,h2)
    #print(state.dot(np.conjugate(state)))
    Cij = np.reshape(state, dim_list )
    Cij = np.transpose(Cij,sites+sites_compl)
    Cij = np.reshape(Cij,(dsites,dsites_compl))
    print(sites_compl,sites)
    print(np.shape(Cij))
    S = np.linalg.svd(Cij, full_matrices = 0, compute_uv = 0)
    # print("SVD=",S)
    S = S[S>(10**-15)]
    print(S)
    # if(-np.sum((S**2)*np.log(S**2))<0):
    #     print("error, schmidt spec",Sorig)
    return -np.sum((S**2)*np.log(S**2))


def Schmidt_variablehs(state, sites, dim_list):

#The sites (in range 0,..., L-1 ) in the entanglement region

    L = len(dim_list)
    sites_compl=[]
    
    if len(sites)==L or len(sites)==0:
        return np.array([1.0])
    
    for j in range(L):
        if not(j in sites):
            sites_compl.append(j)
    dsites = np.prod([dim_list[i] for i in sites])
    dsites_compl = np.prod([dim_list[i] for i in sites_compl])
    
#     print(sites_compl)
    # print("h1,2=",h1,h2)
    #print(state.dot(np.conjugate(state)))
    Cij = np.reshape(state, dim_list )
    Cij = np.transpose(Cij,sites+sites_compl)
    Cij = np.reshape(Cij,(dsites,dsites_compl))
#     print(sites_compl,sites)
#     print(np.shape(Cij))
    S = np.linalg.svd(Cij, full_matrices = 0, compute_uv = 0)
#     print(S)
    # if(-np.sum((S**2)*np.log(S**2))<0):
    #     print("error, schmidt spec",Sorig)
    return S



def Entanglement_variablehs(state, sites, dim_list):
    S=Schmidt_variablehs(state, sites, dim_list)
#     print(S)
    S = S[S>(10**-15)]
   
    # if(-np.sum((S**2)*np.log(S**2))<0):
    #     print("error, schmidt spec",Sorig)
    return -np.sum((S**2)*np.log(S**2))

def MI_variablehs(state,sitesets,dim_list):
    sitesets_parsed=[]
    for a in sitesets:
        if a !=[]:
         sitesets_parsed.append(a)
    
    num_reg = len(sitesets_parsed) 
#     print("num_reg=",num_reg)
    if num_reg == 1:
        return 0.0
        
    elif num_reg == 2:
        SA = Entanglement_variablehs(state,sitesets[0],dim_list)
#         print("SA=",SA)
        SB = Entanglement_variablehs(state,sitesets_parsed[1],dim_list)
#         print(sitesets[1])
#         print("SB=",SB)
        SAB = Entanglement_variablehs(state,sitesets_parsed[0]+sitesets_parsed[1],dim_list)
        return SA + SB - SAB
    
    elif num_reg == 3:
        SA = Entanglement_variablehs(state,sitesets_parsed[0],dim_list)
#         print("SA=",SA)
        SB = Entanglement_variablehs(state,sitesets_parsed[1],dim_list)
    
        SC = Entanglement_variablehs(state,sitesets_parsed[2],dim_list)

        SAB = Entanglement_variablehs(state,sitesets_parsed[0]+sitesets_parsed[1],dim_list)
        SBC = Entanglement_variablehs(state,sitesets_parsed[1]+sitesets_parsed[2],dim_list)
        SCA = Entanglement_variablehs(state,sitesets_parsed[2]+sitesets_parsed[0],dim_list)
                
        SABC = Entanglement_variablehs(state,sitesets_parsed[0]+sitesets_parsed[1]+sitesets_parsed[2],dim_list)
        
        
        
        return SA + SB + SC - SAB - SBC - SCA + SABC
        

def timeEvolutionMIStrobos_variablehs(evals, evecs, psi0,sitesets, nPeriods,dim_list):
    Times=np.arange(0,nPeriods)

    MITimesList = np.zeros(nPeriods)
    # print("psi0 norm",psi0.dot(np.conjugate(psi0)))

    evecsinv = np.linalg.inv(evecs)
    #Using the inverse here because the spectrum is somewhat degenerate
    psieigb = np.dot(evecsinv, psi0)
    for i in range(len(Times)):
        
        #print("phieigb norm",psieigb.dot(np.conjugate(psieigb)))
        psiphys=np.dot(evecs, psieigb)
        
        # print("phiphys norm",psiphys.dot(np.conjugate(psiphys)))
        MITimesList[i]=MI_variablehs(psiphys, sitesets ,dim_list)
        t=Times[i]
        psieigb = psieigb*np.exp(-1j*evals)
    return MITimesList
    

def timeEvolutionSchmidtStrobos_variablehs(evals, evecs, psi0,sites, nPeriods,dim_list):
    Times=np.arange(0,nPeriods)

    SVDTimesList = []
    # print("psi0 norm",psi0.dot(np.conjugate(psi0)))

    evecsinv = np.linalg.inv(evecs)
    #Using the inverse here because the spectrum is somewhat degenerate
    psieigb = np.dot(evecsinv, psi0)
    


    for i in range(len(Times)):
        
        #print("phieigb norm",psieigb.dot(np.conjugate(psieigb)))
        psiphys=np.dot(evecs, psieigb)
       # print(len(psiphys[np.abs(psiphys)>10**(-5)]))
        # print("phiphys norm",psiphys.dot(np.conjugate(psiphys)))
        SVDTimesList.append(Schmidt_variablehs(psiphys, sites,dim_list))
        t=Times[i]
        psieigb = psieigb*np.exp(-1j*evals)
    return SVDTimesList


def timeEvolutionEntanglementStrobos_variablehs(evals, evecs, psi0,sites, nPeriods,dim_list):
    Times=np.arange(0,nPeriods)

    MITimesList = np.zeros(nPeriods)
    # print("psi0 norm",psi0.dot(np.conjugate(psi0)))

    evecsinv = np.linalg.inv(evecs)
    #Using the inverse here because the spectrum is somewhat degenerate
    psieigb = np.dot(evecsinv, psi0)
    for i in range(len(Times)):
        
        #print("phieigb norm",psieigb.dot(np.conjugate(psieigb)))
        psiphys=np.dot(evecs, psieigb)
        
        # print("phiphys norm",psiphys.dot(np.conjugate(psiphys)))
        MITimesList[i]=Entanglement_variablehs(psiphys, sites ,dim_list)
        t=Times[i]
        psieigb = psieigb*np.exp(-1j*evals)
    return MITimesList
