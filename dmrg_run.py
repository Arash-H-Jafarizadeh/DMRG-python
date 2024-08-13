import numpy as np
import scipy.linalg as la
from helper import *
from mps_functions import *
from mpo_functions import *
from ed_functions import *

"""comment 20240813:
    so far I had no luck getting a correct result for <state|H|state> for both TN method and ED method. It works on XX terms only, but not for XX+Z terms
"""

mps_state = MPS.random_chi(4, max_bond=2)
print([a.shape for a in mps_state.left_tensors], mps_state.schmidt_matrix.shape, [a.shape for a in mps_state.right_tensors])
dens_state = MPS.to_dense(mps_state)
# print(la.norm(dens_state), " , ", dens_state @ Spin(3,1,5) @ np.conj(dens_state))


# W0 = ising(2,-1,-0.0)[0]
# for wc in range(1,2):
#     W0 = np.tensordot( W0, ising(2,-1,0.0)[wc], (-1,0))
    
# indexes = tuple(2*i for i in range(2)) + tuple(2*i+1 for i in range(2))
# W0 = np.transpose(W0, indexes)
# # W0.reshape((16,16))
# # print(ising_ED(-0,0.0,2))
# print(W0.reshape((4,4)) - ising_ED(-0.3,0.0,4))

mps_ham = ising(3,-1,-.0)
# sp1=np.reshape(np.array([[0,1],[1,0]]),(2,2,1))
# sp2=np.reshape(np.array([[0,1],[1,0]]),(1,2,2))
E0 = np.einsum('ijk,jlm,ilo->kmo',mps_state.right_tensors[0],mps_ham[0],np.conj(mps_state.right_tensors[0]))
print(" --0--",E0.shape)
E1 = np.einsum('ijk,ljmn,omp->iloknp',mps_state.right_tensors[1],mps_ham[1],np.conj(mps_state.right_tensors[1]))
print(" --1--",E1.shape)
E2 = np.einsum('ijk,ljm,omk->ilo',mps_state.right_tensors[2],mps_ham[2],np.conj(mps_state.right_tensors[2]))
print(" --2--",E2.shape)
# E3 = np.einsum('ijk,ljm,mnk->iln',mps_state.right_tensors[3],mps_ham[3],np.conj(mps_state.right_tensors[3]))
# print(" --3--",E3.shape)
# Ef = np.einsum('cde,cdefgh,fghijk,ijk->',E0,E1,E2,E3)
Ef = np.einsum('abc,abcdef,def->',E0,E1,E2)
print(" --f--",Ef.shape, " , ", Ef)
# print(la.norm(dens_state), ", ", dens_state @ ising_ED(0.0,0,4) @ np.conj(dens_state))
print("--ED--",la.norm(dens_state), ", ", dens_state @ Spin(1,1,4) @ Spin(1,2,4) @ np.conj(dens_state) + dens_state @ Spin(1,2,4)@Spin(1,3,4) @ np.conj(dens_state))

# zop=np.array([[1, 0],[0, -1]])
# E0 = np.tensordot(np.conj(mps_state.right_tensors[0]), mps_state.right_tensors[0],([0,2],[0,2]))
# E0 = np.tensordot(E0, zop,([0,1],[0,1]))
# print(" --0--", E0.shape, " , ",E0)
# E1 = np.tensordot(np.conj(mps_state.right_tensors[0]), np.conj(mps_state.right_tensors[1]),(2,0))
# E2 = np.tensordot(mps_state.right_tensors[0], mps_state.right_tensors[1],(2,0))
# E3 = np.tensordot(E1, E2,([0,1,3],[0,1,3]))
# E0 = np.tensordot(E3, zop,([0,1],[0,1]))
# print(" --1--", E0.shape," , ", E0)
# E1 = np.tensordot(np.conj(mps_state.right_tensors[0]), np.conj(mps_state.right_tensors[1]),(-1,0))
# E1 = np.tensordot(E1, np.conj(mps_state.right_tensors[2]),(-1,0))
# E2 = np.tensordot(mps_state.right_tensors[0], mps_state.right_tensors[1],(-1,0))
# E2 = np.tensordot(E2, mps_state.right_tensors[2],(-1,0))
# E3 = np.tensordot(E1, E2,([0,1,2,4],[0,1,2,4]))
# E0 = np.tensordot(E3, zop,([0,1],[0,1]))
# print(" --1--", E0.shape," , ", E0)

# print("--ED--",la.norm(dens_state), ", ",[dens_state @ Spin(3, ii, 4) @ np.conj(dens_state) for ii in range(1,5)], sum([dens_state @ Spin(3, ii, 4) @ np.conj(dens_state) for ii in range(1,5)]))

