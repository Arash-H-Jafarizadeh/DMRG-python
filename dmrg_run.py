import numpy as np
import scipy.linalg as la
from helper import *
from mps_functions import *
# from mpo_functions import *
import mpo_functions as mpo
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
print('\n')
# print("-ed -", [np.real( dens_state @ ed.Spin(3,jj,L) @ np.conj(dens_state) ) for jj in range(1,L+1)], ",", sum( [np.real( dens_state @ ed.Spin(3,jj,L) @ np.conj(dens_state) ) for jj in range(1,L+1)]))
# print("-ed -", [np.real( dens_state @ ed.Spin(1,jj,L) @ np.conj(dens_state) ) for jj in range(1,L+1)], ",", sum( [np.real( dens_state @ ed.Spin(1,jj,L) @ np.conj(dens_state) ) for jj in range(1,L+1)]))
print("-ed -", [-1.0*np.real( dens_state @ ed.Spin(1,jj,L) @ ed.Spin(1,jj+1,L) @ np.conj(dens_state) ) for jj in range(1,L)] ,",", np.real( dens_state @ ed.ising_ED(0,0,L) @ dens_state.conj() ) )
# print(la.norm(dens_state), ",", sum([np.real( dens_state @ ed.Spin(1,jj,L) @ np.conj(dens_state) ) for jj in range(1,L+1)]) ,",", dens_state @ ed.ising_ED(0,1,L) @ dens_state.conj())

## norm of the mps state
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

## one operator (here is X or Z so far)
# X_energy = []
# X0 = np.tensordot( mps_state.schmidt_matrix, mps_state.schmidt_matrix.conj(), (0,0)) 
# for el in range(L):
#     # E0 = np.tensordot(X0, xop, ([0,2],[0,1]))
#     E0 = np.tensordot(mps_state.right_tensors[el], mps_state.right_tensors[el].conj(), (2,2))
#     # print("----", E0.shape)
#     E0 = np.tensordot(E0, zop, ([1,3],[0,1]))
#     # print("----", E0.shape)
#     E1 = np.tensordot(E0, X0, ([0,1],[0,1]))
#     # print("----", E1.shape,",", np.real(E1))
#     X_energy.append(np.float64(np.real(E1)))
#     X0 = np.tensordot(X0, np.tensordot(mps_state.right_tensors[el], mps_state.right_tensors[el].conj(),(1,1)), ([0,1],[0,2]))
# print("-mps-",X_energy,",",sum(X_energy))

# ## two operators (here is XX so far)
XX_energy = []
xxop = np.tensordot(-1.0 * xop,xop,0)
XX0 = np.tensordot(mps_state.schmidt_matrix, mps_state.schmidt_matrix.conj(),(0,0)) 
for el in range(L-1):
    E0 = np.tensordot(mps_state.right_tensors[el], mps_state.right_tensors[el+1],(2,0))
    E1 = np.tensordot(E0, E0.conj(), (3,3))
    # print(" ----",E1.shape)
    E2 = np.tensordot(E1, XX0, ([0,3],[0,1])) 
    # print(" ----",E2.shape)
    E3 = np.tensordot(E2, xxop, ([0,1,2,3],[0,2,1,3]))
    # print(" ----",E3.shape,",",E3)
    XX_energy.append( np.float64(E3.real) )
    XX0 = np.tensordot(XX0, np.tensordot(mps_state.right_tensors[el],mps_state.right_tensors[el].conj(),(1,1)), ([0,1],[0,2]))
    # print(" ----",XX0.shape)
print("-mps-",XX_energy,",",sum(XX_energy))


## ~~ trying the mpo version
# mps_state = MPS.random_chi(4, max_bond=2)
# print([a.shape for a in mps_state.left_tensors], mps_state.schmidt_matrix.shape, [a.shape for a in mps_state.right_tensors])
# dens_state = MPS.to_dense(mps_state)
# # print(la.norm(dens_state), " , ", dens_state @ Spin(3,1,5) @ np.conj(dens_state))
print('\n')
mpo_ham = mpo.ising(L,1.0,0.0)
print([aa.shape for aa in mpo_ham])

var0 = np.tensordot(mps_state.schmidt_matrix, mps_state.schmidt_matrix.conj(),(0,0))
# print("- ",var0.shape)
var1 = np.tensordot(mps_state.right_tensors[0], mpo_ham[0],(1,1))
# print("- ",var1.shape)
var1 = np.tensordot(var1,mps_state.right_tensors[0].conj(),(2,1))
# print("- ",var1.shape)
var0 = np.tensordot(var0, var1, ([0,1],[0,3]))
# print("- ",var0.shape)
for el in range(1,L-1):
    # print('-- loop --', el)
    var2 = np.tensordot(mps_state.right_tensors[el], mpo_ham[el],(1,2))
    # print("-- ",var2.shape)
    var3 = np.tensordot(var2,mps_state.right_tensors[el].conj(),(3,1))
    # print("-- ",var3.shape)
    var0 = np.tensordot(var0, var3, ([0,1,2],[0,2,4]))
    # print("-- ",var0.shape)
# print('- loop -', L-1)
var4 = np.tensordot(mps_state.right_tensors[L-1], mpo_ham[L-1],(1,2))
# print("-- ",var4.shape)
var5 = np.tensordot(var4,mps_state.right_tensors[L-1].conj(),(3,1))
# print("-- ",var5.shape)
var0 = np.tensordot(var0, var5, ([0,1,2],[0,2,3]))
print("-- ",var0.shape, "-->", var0[0,0].real)
    
