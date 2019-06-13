""" Comparison of the entanglement growth S(t) following a global quench using 
the first order MPO based time evolution for the XX model with the TEBD algorithm. 

The TEBD algorithm is also used to recompress the MPS after the MPO time evolution. 
This is simpler to code but less efficient than a variational optimization. 
See arXiv:1407.1832 for details and how to extend to higher orders.

Frank Pollmann, frankp@pks.mpg.de """
import sys
sys.path.insert(0,'/Users/curtvk/Dropbox/DynalistNotes/Branches')

import numpy as np
from scipy.linalg import expm
import time
from SpinLanguageSparse import*


def initial_state_af(d,chi_max,L,dtype=complex):
    " Create an antiferromagnetc product state. "
    B_list = []
    #B[phys, Aux1, Aux2]
    # The B's here correspond to B(b^{-}=Gamma(b^{-}).Lambda(b)
    s_list = []
    for i in range(L):
        B = np.zeros((d,1,1),dtype=dtype)
        if(np.mod(i,2)==0):
            B[0,0,0] = 1.
        else:
            B[d-1,0,0]=1
        s = np.zeros(1)
        s[0] = 1.
        B_list.append(B)    
        s_list.append(s)
    s_list.append(s) #Note there are L+1 of these
    return B_list,s_list 

def initial_state_up(d,chi_max,L,dtype=complex):
    " Create an antiferromagnetc product state. "
    B_list = []
    s_list = []
    for i in range(L):
        B = np.zeros((d,1,1),dtype=dtype)
        B[0,0,0] = 1.
        s = np.zeros(1)
        s[0] = 1.
        B_list.append(B)    
        s_list.append(s)
    s_list.append(s)
    return B_list,s_list
    
def make_U_xx_bond(L,delta):
    " Create the bond evolution operator used by the TEBD algorithm."
    sp = np.array([[0.,1.],[0.,0.]])
    sm = np.array([[0.,0.],[1.,0.]])
    d = 2
    H = np.real(np.kron(sp,sm) + np.kron(sm,sp))
    u_list = (L-1)*[np.reshape(expm(-delta*H),(2,2,2,2))]
    return u_list,d    

def make_U_bond(H, L,delta):
    " Create the bond evolution operator used by the TEBD algorithm"
    " For real-time evolution, we need imaginary time "
    d = int(np.round(np.sqrt(H.shape[0])))
    u_list = (L-1)*[np.reshape(expm(-delta*H),(d,d,d,d))]
    return u_list

def make_U_heis_bond(L,delta,N):
    " Create the bond evolution operator used by the TEBD algorithm."
    s0_list, sx_list,sy_list,sz_list = gen_s0sxsysz_generalspin(1,N)
    d = int(2*N+1)
    x=np.array(sx_list[0].todense())
    y=np.array(sy_list[0].todense())
    z=np.array(sz_list[0].todense())

    H = N*(-np.kron(x,x)-np.kron(y,y)-np.kron(z,z))
    u_list = (L-1)*[np.reshape(expm(-delta*H),(d,d,d,d))]
    return u_list,d   

def make_U_xx_mpo(L,dt,dtype=float):
    " Create the MPO of the time evolution operator.  "
    s0 = np.eye(2)
    sp = np.array([[0.,1.],[0.,0.]])
    sm = np.array([[0.,0.],[1.,0.]])
    w = np.zeros((3,3,2,2),dtype=type(dt))
    w[0,:] = [s0,sp,sm]
    w[1:,0] = [-dt*sm,-dt*sp]
    w_list = [w]*L
    print(len(w_list))
    return w_list

def make_U_heis_mpo(L,dt,N,dtype=float):
    s0_list, sx_list,sy_list,sz_list = gen_s0sxsysz_generalspin(1,N)

    d=int(2*N+1)
    I =np.eye(d)
    O=1.0*np.zeros((d,d))
    x=np.array(sx_list[0].todense())
    y=np.array(sy_list[0].todense())
    z=np.array(sz_list[0].todense())

    sp = x+1j*y
    sm = x-1j*y

    #left_chi, right_chi, out_phys, in phys


    w = np.zeros((4,4,d,d),dtype=type(dt))
    w[0,:] = [I,dt*N*x,dt*N*y,dt*N*z]
    w[1,:] = [x,O,O,O]
    w[2,:] = [y,O,O,O]
    w[3,:] = [z,O,O,O]
    
    w_list = [w[0,:,:,:]]
    if(L>2):
        w_list =w_list+(L-2)*[w]
  
    w_list = w_list+[w[:,0,:,:]]
   
    #This bdry condition ensures we're dealing with 1-ij H
    # w_list[0][4,:,:,:] =[I,dt*N*0.5*sm,dt*N*0.5*sp,dt*N*z,I]

    return w_list


def one_site_rot(N,theta):
    #N is the spin value e.g., spin 1/2 is N=1/2
    s0_list, sx_list,sy_list,sz_list = gen_s0sxsysz_generalspin(1,N)
    Yprop = np.array((sy_list[0]*N).todense() ) #This matrix has evalues [-N,...,N]
    return expm(-1j*theta*Yprop)


def branch_hit(B_list,branch_sites,op):
    for i in branch_sites:
        B_list[i]=np.tensordot(op,B_list[i],axes=(1,0))
    return B_list


def apply_mpo_svd(B_list,s_list,w_list,chi_max):
    " Apply the MPO to an MPS."
    """ MPS B_list is 0-indexed list of tensors, and each one is 3-index tensor: physical, left, right
    w_list is the MPO, 4-index tensor: left_chi, right_chi, out_phys, in_phys
    s_list is list of Schmidt values, s_list is length L+1, first and last are trivial
    """
    
    #physical dimension
    d = B_list[0].shape[0]

    #MPO right bond dimension at site 0
    D = w_list[0].shape[1]

    #sites
    L = len(B_list)
    
    #bond dimension to left/right of site 0
    chi1 = B_list[0].shape[1]
    chi2 = B_list[0].shape[2]      

   
   #Fix the left boundary condition on the MPO
    # B = np.tensordot(B_list[0],w_list[0][0,:,:,:],axes=(0,1))
    B = np.tensordot(B_list[0],w_list[0][0,:,:,:],axes=(0,1))
    """ In-phys vs. out-phys ?? """
    
    B = np.reshape(np.transpose(B,(3,0,1,2)),(d,chi1,chi2*D))
    B_list[0] = B
            
    for i_site in range(1,L-1):
        chi1 = B_list[i_site].shape[1]
        chi2 = B_list[i_site].shape[2]        
        #w_list[0] = w_{a b A B}
        #contract w into B along physical index form a new tensor  
        # Bnew=B_{A a b} w_{c d A B}
        # Bnew=Bnew_{a b c d B} ....reshape to Bnew_{B a c b d}
        B = np.tensordot(B_list[i_site],w_list[i_site][:,:,:,:],axes=(0,2))
        # Bnew=Bnew_{a b c d B} ....reshape to Bnew_{B a c b d}
        B = np.reshape(np.transpose(B,(4,0,2,1,3)),(d,chi1*D,chi2*D))
        B_list[i_site] = B
        ##extend the schmidt table to reflect presence of MPO
        s_list[i_site] = np.reshape(np.tensordot(s_list[i_site],np.ones(D),axes=0),D*chi1)
    
    chi1 = B_list[L-1].shape[1]
    chi2 = B_list[L-1].shape[2]     
    #For current Heisenberg MPO the right BC is the first column(correct!)
    B = np.tensordot(B_list[L-1],w_list[L-1][:,0,:,:],axes=(0,1))


    B = np.reshape(np.transpose(B,(3,0,2,1)),(d,D*chi1,chi2))    

    #extend the schmidt table to reflect presence of MPO
    # ...here we're thinking about the L-2,L-1 bond
    s_list[L-1] = np.reshape(np.tensordot(s_list[L-1],np.ones(D),axes=0),D*chi1)
    B_list[L-1] = B
    
    #do TEBD with trivial unitary
    tebd(B_list,s_list,(L-1)*[np.reshape(np.eye(d**2),[d,d,d,d])],chi_max)

def tebd(B_list,s_list,U_list,chi_max,schmidt_tol = 10e-8):
    " Use TEBD to optmize the MPS and to project it back. "
    d = B_list[0].shape[0]
    L = len(B_list)
    for p in [0,1]: #parity of bond
        for i_bond in np.arange(p,L-1,2): 
            #ibond is the leftmost site i1 in the bond being considered
            i1=i_bond
            #i2 is the rightmost site in the bond being considered
            i2=i_bond+1
            
            #chi left of leftmost site
            chi1 = B_list[i1].shape[1]
            #chi right of rightmost site
            chi3 = B_list[i2].shape[2]
            
            # Construct theta matrix #
            #Contract B's in obvious wa
            # B = B_{j a b}
            # C = B_{j a b } B_{k b c}
            C = np.tensordot(B_list[i1],B_list[i2],axes=(2,1))  
            #C = C[phys_1,aux_l,phys_2,aux_r]
            
            

#            C = np.tensordot(C,U_list[i_bond],axes=([0,2],[0,1]))
            #C = C[phys_1,aux_l,phys_2,aux_r] U[phys_out_1,phys_out_2,phys_in_1,phys_in_2]
            #CHANGED THIS LINE...was doing U^T C rather than U C
            C = np.tensordot(C,U_list[i_bond],axes=([0,2],[2,3]))
            #C = C[aux_l,aux_r,phys_out_1,phys_out_2]
            
            
            
            #np.tranpose(C) reverses order of indices on C
            #tempC[phys_out_2,phys_out_1, aux_r,aux_l]
            #tempC[phys_out_1,aux_l,phys_out_2, aux_r]
            #theta[phys_out_1 aux_l,   phys_out_2  aux_r]
      
            theta = np.reshape(np.transpose(np.transpose(C)*s_list[i1],(1,3,0,2)),(d*chi1,d*chi3))
           
            C = np.reshape(np.transpose(C,(2,0,3,1)),(d*chi1,d*chi3))
            #C = C[phys_out_1,aux_l,phys_out_2,aux_r]

            #at this point, C is simply theta without the lambdas multipled into it


            # Schmidt decomposition #
            X, Y, Z = np.linalg.svd(theta)
            #theta = s.C = X.Y.Z
            #C = s^{-1} . X . Y . Z
            Z=Z.T

            # Think of X. Y.Z = s G s G s
            # Recall (see dynalist) that according to our MPS rules Gs . (Gs)^dagger = 1
            # So we choose to identify Z to be the last Gs
            # Therefore X.Y = sGs 1 1
        
            W = np.dot(C,Z.conj())  
            #Using the above have
            #W = s^{-1}.X.Y = G s 1 1

            #new bond dimension associated with middle bond
            #truncation happens here!
            chi2 = np.min([np.sum(Y >= schmidt_tol), chi_max])
        
            # Normalize new schmidt spectrum for middle bond
            invsq = np.sqrt(sum(Y[:chi2]**2))
            s_list[i2] = Y[:chi2]/invsq 
            # if(chi2==chi_max):
            #     print("chi is maxed out=",chi2)


            #Update the left site MPS B = s^{-1}.X.Y  = Gs 1 1 as it should!
            B_list[i1] = np.reshape(W[:,:chi2],(d,chi1,chi2))/invsq

            #update the rightmost MPS B = 1 1 G s  as discussed above.
            B_list[i2] = np.transpose(np.reshape(Z[:,:chi2],(d,chi3,chi2)),(0,2,1))

    
    # t0 = time.time()
    # for step in range(N_steps): 
    #     if(step%per_steps==0):
    #         B_list=branch_hit(B_list,branch_sites,RotY)

    #     apply_mpo_svd(B_list,s_list,w_list,chi_max) 

    #     print("chis,MPO method=",[x.size for x in s_list])
    #     s2 = np.array(s_list[int(L/2)])**2
    #     S.append(-np.sum(s2*np.log(s2))) 
    # t1 = time.time()
    # print(t1-t0)
    # pl.plot(delta*np.arange(N_steps+1),S)


def cal_entrMI_MPS(B_list,s_list,reg_set):
    #will calcualte the nth party mI or entropy of a set of n subregions given the MPS and a set of regions
    gaps = len(reg_set)+1
    GT_list = [] #These will contain the tensors that exist between the regions
    r = reg_set[0][0] #first site in the first region
    GT = np.tensordot(s_list[r],s_list[r],axes=0) #straighforward kronecker product
    GT_list.append(GT)
    d = B_list[0].shape[0] #physical dimension at leftmost site

    # all_site=[]
    # for reg in reg_set:
    #     all_site = all_site+reg
    gap_list=[]
    for n,reg in enumerate(reg_set):
        if(n==0):
            gap_list.append(list(range(0,reg[0])))
        else:
            gap_list.append(list(range(reg_set[n-1][-1]+1,reg[0])) )
    gap_list.append(list(range(reg_set[n][-1]+1,L)))  


    #will need to cycle over gaps here and deal with edge cases
    for gap in gap_list:
        for i,x in enumerate(gap):
            B = B_list[x]
            Bc= B.conj()
            E = np.tensordot(B,Bc,axes=(0,0))
                    #B[j,a,b]B*[j,c,d]
            E = np.transpose(E,axes=(0,2,1,3))
            # E = E[aL,bL,aR,bR]
            chiL = len(s_list(x[0]+1))
            chiR = len(s_list(x[-1]+1))
            E = E.reshape(chiL**2,chiR**2)
            #B[j,a,b]
            if(i==0):
                Tr=E
            else:
                Tr = np.tensordot(Tr,E,axes=(1,0))
        GT_list.append(Tr)
    #Tr at this point should have indices Tr[A B] where A is a double virtual index
    #Now we have list of gap tensors
    #We will now construct the region tensors
    rho_list=[]
    
    for r in reg_set:
        for i,x in enumerate(r):
            B = B_list[x]
            Bc= B.conj()
            E = np.tensordot(B,Bc,axes=0)
             #B[j,a,b]B*[j',c,d]
            E = np.transpose(E,axes=(0,3,1,4,2,5))
            # E = E[j,j',a,c,b,d]
            chiL = len(s_list(x))
            chiR = len(s_list(x+1))
            E = E.reshape(d,d,chiL**2,chiR**2)
             # E = E[j,j',A,B]
            E = np.transpose(E,axes=(2,0,1,3))
            # E = E[A,j,j',,B]
           
            if(i==0):
                Tr= E
            else:
                Tr = np.tensordot(Tr,E,axes=1)
                 #E[A,j,j',B] E[B,l,l',,C]

        rho_list.append(Tr)
    for i,r in enumerate(reg_set):
        
        if(gap_list[i]!=[]):
            temp = np.tensordot(GT_list[i],rho_list[i], axes=1 )
        else:
            temp = rho_list[i]
        if(i==0):
            T=temp
        else:
            T = np.tensordot(T,temp,axes=1)

    if(gap_list[-1]!=[]):
        T=np.tensordot(T,GT_list[-1])

    #T = T[A,i,i',j,j'...,B]
    shape = T.shape
    k = len(shape)
    l=[0,k-1]+list(range(1,k-1))
    T = np.reshape(T,l)
    #T = T[A,B,i,i',j,j'...]
    T = T[0]
    rhofull = T[0]
    #rhofull = rhofull[i,i',j,j',....]

    perm=[2*i for i in range(len(reg_set))] + [2*i+1 for i in range(len(reg_set))]
    rhofull = np.transpose(rhofull,axes=perm)
    #rhofull[i,j,k,ldots, i',j',k',....]
    reg_reshape=[]
    for x in reg_set:
        reg_reshape.append(d**len(x))
    reg_reshape = reg_reshape+reg_reshape
    rhofull = np.reshape(rhofull,reg_reshape)
     #rhofull[IA,IB,IC,ldots, IA',IB',IC',...]
        
    return entrMI_rhofull(rhofull)
#
######
#Given a rho in form rhofull[IA IB IC ... IA' IB' IC'], calculate all mutual info
# reg_set = [R1, R2 , ... R_{NR}] R1 etc are contiguous regions
#rho.shape = [dA dB dC....dNR dA dB  dC...dNR]



def entrMI_rhofull(rho):
    #here we're given rho as rhofull[IA IB IC ... IA' IB' IC']
    dim = int(np.sqrt(rho.size))
#     print(dim)
    rhosq=rho.reshape(dim,dim)
#     print(rhosq.shape)
    vals,_ = np.linalg.eigh(rhosq)
    vals=vals[vals>10**(-15)]
#     print(vals)
    return -np.sum(vals*np.log(vals)) 



def MI_variablehs_TEBD(rho):
    #here we're given rho as rhofull[IA IB IC ...INR.. IA' IB' IC' INR']
    #It will recursively
    #  calculate the N part mutual information SA,- SAB+ SA + SB ,  +SABC- SAB -SBC -SCA+SA + SB +SC etc 
    shape=rho.shape


    NR=int(len(rho.shape)/2)
    if(NR==0):
        return 0.0
    par_reg = 2*(NR%2) -1
    SubMI = np.sum( [ MI_variablehs_TEBD(   np.trace(rho,axis1= i,axis2=i+NR)     ) for i in range(NR) ])
    return par_reg*(entrMI_rhofull(rho)+SubMI )

        

        





