import numpy as np
from scipy.linalg import expm, sinm, cosm
# from helper import *
"""comments 2024/08/09:
    I don't think defining the MPO as a class was neccessary since I wil so DMRG on the state. 
    At this point, I will use non-class MPO.  """

# class MPO:
#     def __init__(self, mpo_tensors=[]):
#         self.mpo_tensors = mpo_tensors


#     @classmethod
# def ising(cls, N, J, g, boundary_condition=False): #*for class version of code
def ising(N, J, g, boundary_condition=False):
    w_tensor = np.zeros((3,2,2,3))  #* tensor convention should be (D,d,d,D). 
    w_tensor[0,:,:,0] = np.eye(2) # 1 
    w_tensor[0,:,:,1] = np.eye(2)[[1,0]] #X
    w_tensor[0,:,:,2] = -J*g*np.array([[1,0],[0,-1]]) #-Jg.Z
    w_tensor[1,:,:,2] = -J*np.eye(2)[[1,0]] #-J.X
    w_tensor[2,:,:,2] = np.eye(2) #1
    
    op_tensors = [w_tensor for _ in range(N)]
    
    # left_tensor = np.zeros((2,2,3)) # *manually defining the left most tensor
    # left_tensor[:,:,0]=np.eye(2) #1
    # left_tensor[:,:,1]=np.eye(2)[[1,0]] #X 
    # left_tensor[:,:,2]=-J*g*np.array([[1,0],[0,-1]]) #-Jg.Z 
    
    # right_tensor = np.zeros((3,2,2)) # *manually defining the right most tensor
    # right_tensor[0,:,:] = -J*g*np.array([[1,0],[0,-1]]) #-Jg.Z
    # right_tensor[1,:,:] = -J*np.eye(2)[[1,0]] #-J.X
    # right_tensor[2,:,:] = np.eye(2) #1
    
    # op_tensors[0] = left_tensor #.reshape((1,2,2,3))
    # op_tensors[N-1] = right_tensor #.reshape((3,2,2,1))
    op_tensors[0] = np.tensordot([1, 0, 0], op_tensors[0], axes=1)
    op_tensors[-1] = np.tensordot(op_tensors[-1], [0,0,1], axes=1) 
    # return cls(mpo_tensors=op_tensors) #*for class version of code
    return op_tensors
    
    # @classmethod
# def pxp(cls, N, J, boundary_condition=False): #*for class version of code
def pxp(N, J, boundary_condition=False):
    w_tensor = np.zeros((4,2,2,4))  #*tensor convention should be (D,d,d,D). 
    w_tensor[0,:,:,0] = np.eye(2) # 1 
    w_tensor[0,:,:,1] = -J*np.array([[1, 0],[0, 0]])# J.P
    w_tensor[1,:,:,2] = np.eye(2)[[1,0]] # X
    w_tensor[2,:,:,3] = np.array([[1,0],[0,0]]) # P
    w_tensor[3,:,:,3] = np.eye(2) # 1
    
    op_tensors = [w_tensor for _ in range(N)]
    
    op_tensors[0] = np.tensordot([1, 0, 0, 0], op_tensors[0], axes=1)
    op_tensors[-1] = np.tensordot(op_tensors[-1], [0,0,0,1], axes=1)
    # return cls(mpo_tensors=op_tensors) # *for class version of code
    return op_tensors

# if __name__ == "__main__":
