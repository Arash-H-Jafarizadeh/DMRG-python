import numpy as np
import scipy.linalg as la
from helper import *
from mps_functions import *
from mpo_functions import *
# from ed_functions import *
import ed_functions as ed

"""comment 20240813:
    so far I had no luck getting a correct result for <state|H|state> for both TN method and ED method. It works on XX terms only, but not for XX+Z terms
"""
#### ed checks
# L = 4
# ediniL=ed.Config_State([1,0,1,0])
# ediniR=ed.Config_State([1,0,0,1])
# print(ediniL)
# print(ediniR)
# allXX = np.zeros((2**L, 2**L))
# Sx = [[0, 1], [1, 0]]
# for l in range(1, L):
#     allXX += np.kron(np.eye(2**(l-1)), np.kron(Sx, np.kron(Sx, np.eye(2**(L-l-1)))))

# # allXX += -1*np.kron(Sx, np.kron( np.eye(2**(L-2)), Sx))
# fin = ediniL @ allXX @ ediniR
# print( fin )

#### tn checks
L = 4
# mps_state = MPS.zero(4)
mps_state = MPS.random_chi(4, max_bond=2)
# print(mps_state.shape)
print([a.shape for a in mps_state.left_tensors], mps_state.schmidt_matrix.shape, [a.shape for a in mps_state.right_tensors])
dens_state = MPS.to_dense(mps_state)
# print(la.norm(dens_state), ",", dens_state @ ed.Spin(1,1,L) @ np.conj(dens_state), ",", dens_state @ ed.Spin(3,1,L) @ np.conj(dens_state))
print(la.norm(dens_state), ",", [np.real( dens_state @ ed.Spin(1,jj,L) @ np.conj(dens_state) ) for jj in range(1,L+1)])
print(la.norm(dens_state), ",", [np.real( dens_state @ ed.Spin(1,jj,L) @ ed.Spin(1,jj+1,L) @ np.conj(dens_state) ) for jj in range(1,L)] ,",", np.real( dens_state @ ed.ising_ED(0,0,L) @ dens_state.conj() ) )
# print(la.norm(dens_state), ",", sum([np.real( dens_state @ ed.Spin(1,jj,L) @ np.conj(dens_state) ) for jj in range(1,L+1)]) ,",", dens_state @ ed.ising_ED(0,1,L) @ dens_state.conj())


# tnnorm = np.tensordot( np.conj( mps_state.schmidt_matrix), mps_state.schmidt_matrix, (0,0))
# print("-",tnnorm.shape)
# for tnsr in mps_state.right_tensors:
#     var = np.tensordot(tnsr.conj(), tnsr, (1,1))
#     var = var.transpose((0,2,1,3))
#     print("---", var.shape)
#     tnnorm = np.tensordot(tnnorm, var, ((0,1),(0,1)) )
#     print("--", tnnorm.shape)
# print(tnnorm)

zop=np.array([[1, 0],[0, -1]])
xop=np.array([[0, 1],[1, 0]])

X0 = np.tensordot( mps_state.schmidt_matrix, mps_state.schmidt_matrix.conj(), (0,0)) 
for el in range(L):
    # E0 = np.tensordot(X0, xop, ([0,2],[0,1]))
    E0 = np.tensordot(mps_state.right_tensors[el], mps_state.right_tensors[el].conj(), (2,2))
    # print("----", E0.shape)
    E0 = np.tensordot(E0, xop, ([1,3],[0,1]))
    # print("----", E0.shape)
    E1 = np.tensordot(E0, X0, ([0,1],[0,1]))
    print("----", E1.shape,",", np.real(E1))
    X0 = np.tensordot(X0, np.tensordot(mps_state.right_tensors[el], mps_state.right_tensors[el].conj(),(1,1)), ([0,1],[0,2]))

# E0 = np.tensordot(np.conj(mps_state.right_tensors[L-1]), mps_state.right_tensors[L-1], (2,2) )
# E0 = np.tensordot( X0, E0, ([0,1],[0,2]) )
# E0 = np.tensordot( E0, xop, ([0,1],[0,1]) )
# print("--", E0)


xxop = np.tensordot(xop,xop,0)
XX0 = np.tensordot(mps_state.schmidt_matrix, mps_state.schmidt_matrix.conj(),(0,0)) 
# print(" --",XX0.shape)
for el in range(L-1):
    E0 = np.tensordot(mps_state.right_tensors[el], mps_state.right_tensors[el+1],(2,0))
    E1 = np.tensordot(E0, E0.conj(), (3,3))
    # print(" ----",E1.shape)
    E2 = np.tensordot(E1, XX0, ([0,3],[0,1])) 
    # print(" ----",E2.shape)
    E3 = np.tensordot(E2, xxop, ([0,1,2,3],[0,2,1,3]))
    print(" ----",E3.shape,",",E3)
    XX0 = np.tensordot(XX0, np.tensordot(mps_state.right_tensors[el],mps_state.right_tensors[el].conj(),(1,1)), ([0,1],[0,2]))
    # print(" ----",XX0.shape)

# E0 = np.tensordot(mps_state.right_tensors[0], mps_state.right_tensors[1],(2,0))
# print(" --0--", E0.shape)
# E3 = np.tensordot(E0, E0.conj(), ([0,3],[0,3]))
# print(" --0--", E3.shape)
# E4 = np.tensordot(E3, xxop, ([0,1,2,3],[0,2,1,3]))
# print(" --0--", E4.shape, " , ",E4)

# print("--ED--",la.norm(dens_state), ", ",[dens_state @ Spin(3, ii, 4) @ np.conj(dens_state) for ii in range(1,5)], sum([dens_state @ Spin(3, ii, 4) @ np.conj(dens_state) for ii in range(1,5)]))



# mps_state = MPS.random_chi(4, max_bond=2)
# print([a.shape for a in mps_state.left_tensors], mps_state.schmidt_matrix.shape, [a.shape for a in mps_state.right_tensors])
# dens_state = MPS.to_dense(mps_state)
# # print(la.norm(dens_state), " , ", dens_state @ Spin(3,1,5) @ np.conj(dens_state))

# mps_ham = ising(3,-1,-.0)
# # sp1=np.reshape(np.array([[0,1],[1,0]]),(2,2,1))
# # sp2=np.reshape(np.array([[0,1],[1,0]]),(1,2,2))
# E0 = np.einsum('ijk,jlm,ilo->kmo',mps_state.right_tensors[0],mps_ham[0],np.conj(mps_state.right_tensors[0]))
# print(" --0--",E0.shape)
# E1 = np.einsum('ijk,ljmn,omp->iloknp',mps_state.right_tensors[1],mps_ham[1],np.conj(mps_state.right_tensors[1]))
# print(" --1--",E1.shape)
# E2 = np.einsum('ijk,ljm,omk->ilo',mps_state.right_tensors[2],mps_ham[2],np.conj(mps_state.right_tensors[2]))
# print(" --2--",E2.shape)
# # E3 = np.einsum('ijk,ljm,mnk->iln',mps_state.right_tensors[3],mps_ham[3],np.conj(mps_state.right_tensors[3]))
# # print(" --3--",E3.shape)
# # Ef = np.einsum('cde,cdefgh,fghijk,ijk->',E0,E1,E2,E3)
# Ef = np.einsum('abc,abcdef,def->',E0,E1,E2)
# print(" --f--",Ef.shape, " , ", Ef)
# # print(la.norm(dens_state), ", ", dens_state @ ising_ED(0.0,0,4) @ np.conj(dens_state))
# print("--ED--",la.norm(dens_state), ", ", dens_state @ Spin(1,1,4) @ Spin(1,2,4) @ np.conj(dens_state) + dens_state @ Spin(1,2,4)@Spin(1,3,4) @ np.conj(dens_state))


