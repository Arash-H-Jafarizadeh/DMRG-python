
import numpy as np
import time as tt
from scipy import linalg as la
    
"""comment 20240812:
    I think it is usefull to make this ed function as a class. This way working with them is less confusing.
    """


def Spin(a,j,L):
    
    spin = [[[0, 1], [1, 0]], [[0, - 1j],[+ 1j, 0]], [[1, 0],[0, -1]]]
    return np.kron(np.eye(2**(j-1)), np.kron(spin[abs(int(a)-1)], np.eye(2**(L-j))))


def Zero_State(L):
    return np.array( [1] + [0]*(2**L-1) )    


def Z2_State(L):
    zvar=np.zeros((2**L), dtype=np.complex64)
    NN = sum([2**ii for ii in range(1,L,2)])
    zvar[NN] += 1
    return zvar#np.array( [0]*NN + [1] + [0]*(2**L-1) )    


def State_Config(config):
    qbit=[[1.,0.],[0.,1.]]
    acc = 1; [acc := np.kron(acc,qbit[x]) for x in config]
    return  acc  
    
    
def Operator_Expt(ini_state, operator):
    return np.conj(ini_state) @ operator @ ini_state

def ising_ham(h_z,h_x,A, boundary_condition=False):
    Sx = [[0, 1], [1, 0]]
    Sy = [[0, - 1j],[+ 1j, 0]]
    Sz = [[1, 0],[0, -1]]
    H =  np.zeros((2**A, 2**A))
    for l in range(1, A):
        H += -1*np.kron(np.eye(2**(l-1)), np.kron(Sx, np.kron(Sx, np.eye(2**(A-l-1)))))
        H += h_z*np.kron(np.eye(2**(l-1)), np.kron(Sz, np.eye(2**(A-l))))
        H += h_x*np.kron(np.eye(2**(l-1)), np.kron(Sx, np.eye(2**(A-l))))
    H += h_z*np.kron(np.eye(2**(A-1), Sz)) + h_x*np.kron(np.eye(2**(A-1), Sx))
    if boundary_condition:
        H += -1*np.kron(Sx, np.kron( np.eye(2**(A-2)), Sx))
    return H

# def XX_TE_O(dt,L):
#     Sx = [[0, 1], [1, 0]]
#     Sy = [[0, - 1j],[+ 1j, 0]]
#     OddH =  np.zeros((2**L, 2**L),np.complex64)
    
#     for o in range(1,L,2):
#         OddH += 1.0*(np.kron(np.eye(2**(o-1)), np.kron(Sx, np.kron(Sx, np.eye(2**(L-o-1)))))  )
#         #OddH += 0.*(np.kron(np.eye(2**(o-1)), np.kron(Sy, np.kron(Sy, np.eye(2**(L-o-1)))))  )
    
#     EEo, VVo = np.linalg.eigh(OddH)    
    
#     return VVo @ np.diag(np.exp(-dt*1j*EEo)) @ np.conj(np.transpose(VVo))


# def XX_TE_E(dt,L):
#     Sx = [[0, 1], [1, 0]]
#     Sy = [[0, - 1j],[+ 1j, 0]]
#     EvenH =  np.zeros((2**L, 2**L),dtype=np.complex64)
    
#     for e in range(2,L,2):
#         EvenH += 1.*(np.kron(np.eye(2**(e-1)), np.kron(Sx, np.kron(Sx, np.eye(2**(L-e-1)))))  )
#         #EvenH += 0.*np.kron(np.eye(2**(e-1)), np.kron(Sy, np.kron(Sy, np.eye(2**(L-e-1)))))
    
#     EEe, VVe = np.linalg.eigh(EvenH)    
    
#     return VVe @ np.diag(np.exp(-dt*1j*EEe)) @ np.transpose(np.conj(VVe))


