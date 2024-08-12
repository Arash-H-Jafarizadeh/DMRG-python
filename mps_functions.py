import numpy as np
from scipy.linalg import expm, sinm, cosm
from helper import mps_svd

class MPS:

    def __init__(self, left_tensors=[], right_tensors=[], schmidt_matrix=np.eye(1)):
        self.left_tensors = left_tensors
        self.right_tensors = right_tensors  
        self.schmidt_matrix = schmidt_matrix

    @classmethod
    def zero(cls, N):
        # tensor convention: (v1, p, v2)

        B = np.array([1,0]).reshape((1,2,1))

        left_tensors = []
        right_tensors = [B for _ in range(N)]
        schmidt_matrix = np.eye(1)
        return cls(left_tensors=left_tensors, right_tensors=right_tensors, schmidt_matrix=schmidt_matrix)
    
    
    @classmethod
    def ZERO(cls, M, N):

        A = np.array([0,1]).reshape((1,2,1))
        B = np.array([1,0]).reshape((1,2,1))

        left_tensors = [A for _ in range(M)]
        right_tensors = [B for _ in range(N)]
        schmidt_matrix = np.eye(1)
    
        return cls(left_tensors=left_tensors, right_tensors=right_tensors, schmidt_matrix=schmidt_matrix)
    

    @classmethod
    def random_chi(cls, N):

        left_tensors = []
        right_tensors = []
        schmidt_matrix = np.eye(1)

        tensors = [np.random.randn(2,2,2) + 1j*np.random.randn(2,2,2) for _ in range(N)]  # (v1, p, v2)
        tensors[0] = tensors[0][0,:,:].reshape((1,2,2))  # (v1=1, p, v2)
        tensors[-1] = tensors[-1][:,:,0].reshape((2,2,1))  # (v1, p, v2=1)

        for ii in range(N-1, 0, -1):

            M1 = tensors[ii-1]
            M2 = tensors[ii]

            # contract M1 and M2
            M = np.tensordot(M1, M2, axes=([2],[0]))  # (v1, p1, v2)*(v1, p2, v2) -> (v1, p1, p2, v2)
            M = np.tensordot(M, schmidt_matrix, axes=([3],[0]))  # (v1, p1, p2, v2)*(v1, v2) -> (v1, p1, p2, v2)

            A, schmidt_matrix, B = mps_svd(M, chi_max=2)

            tensors[ii-1] = A
            right_tensors.insert(0,B)

        A = tensors[0]
        B = np.tensordot(A, schmidt_matrix, axes=([2],[0]))  # (v1, p1, v2)*(v1, v2) -> (v1, p1, v2)
        right_tensors.insert(0,B)

        schmidt_matrix = np.eye(1)

        return cls(left_tensors=left_tensors, right_tensors=right_tensors, schmidt_matrix=schmidt_matrix)
    


    def center_shift(self, C):

        left_tensors = self.left_tensors
        right_tensors = self.right_tensors
        schmidt_matrix = self.schmidt_matrix

        if C > 0:
            ii = 0
            while ii < C:#for ii in range(C):

                M1 = right_tensors[0] 
                M2 = right_tensors[1] 
                # contract M1 and M2
                M = np.tensordot(M1, M2, axes=([2],[0]))  
                M = np.tensordot(schmidt_matrix, M, axes=([1],[0]))
                A, schmidt_matrix, B = mps_svd(M, chi_max=4)
       
                del right_tensors[:2]
                right_tensors.insert(0,B)
                left_tensors.append(A)
                self.schmidt_matrix=schmidt_matrix

                ii += 1
            return #self.left_tensors, self.right_tensors, self.schmidt_matrix
            
        elif C < 0:
            ii = 0
            while ii < -C:

                M1 = left_tensors[-1]
                M2 = left_tensors[-2]
                M = np.tensordot(M2, M1, axes=([2],[0]))  # (v1, p1, v2)*(v1, p2, v2) -> (v1, p1, p2, v2)
                M = np.tensordot(M, schmidt_matrix, axes=([3],[0]))  # (v1, p1, p2, v2)*(v1, v2) -> (v1, p1, p2, v2)

                A, schmidt_matrix, B = mps_svd(M, chi_max=4)
                
                del left_tensors[-2:]
                left_tensors.append(A)
                right_tensors.insert(0,B)
                self.schmidt_matrix=schmidt_matrix

                ii += 1
            
            return #cls(left_tensors=left_tensors, right_tensors=right_tensors, schmidt_matrix=schmidt_matrix)
        
        elif C == 0:
            return #left_tensors, right_tensors, schmidt_matrix    
    

    # to calculate the expectation of one Pauli operator
    def Pauli_MPO(self, s):
        # left_tensors = self.left_tensors
        # right_tensors = self.right_tensors
        # schmidt_matrix = self.schmidt_matrix

        X_op = np.array([[0,1],[1,0]])
        Y_op = np.array([[0,-1j],[1j,0]])
        Z_op = np.array([[1,0],[0,-1]])

        MPS.center_shift(self, s-len(self.left_tensors)-1 )
        
        ML = np.tensordot(self.right_tensors[0], np.conjugate(self.right_tensors[0]), axes=([2],[2]))
        MZ = np.tensordot(Z_op, ML, axes=([0,1],[1,3]))
        MS = np.tensordot(self.schmidt_matrix, np.conjugate(self.schmidt_matrix), axes=([0],[0])) 
        #MS.reshape(2,2)
        z = np.tensordot(MS, MZ, axes=([0,1],[0,1])) 

        return z


    def to_dense(self):
        # convert MPS to vector 

        vector = np.array([1]).reshape((1,1))  # (p, v)

        # contract left tensors
        for ii in range(len(self.left_tensors)):

            vector = np.tensordot(vector, self.left_tensors[ii], axes=([1],[0]))  # (p1, p2, v)
            vector = vector.reshape((vector.shape[0]*vector.shape[1], vector.shape[2]))  # (p1*p2, v)

        # contract with schmidt matrix
        vector = vector @ self.schmidt_matrix  # (p, v)*(v, v) -> (p, v)

        # contract right tensors
        for ii in range(len(self.right_tensors)):

            vector = np.tensordot(vector, self.right_tensors[ii], axes=([1],[0]))  # (p1, p2, v)
            vector = vector.reshape((vector.shape[0]*vector.shape[1], vector.shape[2]))  # (p1*p2, v)

        vector = vector.reshape((vector.shape[0],))  # (p,)

        return vector

    
    def XX_TEBD_O(self,dt): # I can not do it in class
        
        # left_tensors = self.left_tensors
        # right_tensors = self.right_tensors
        # schmidt_matrix = self.schmidt_matrix
        
        #MPS.center_shift(self, -len(self.left_tensors)+1)
        self.center_shift(-len(self.left_tensors)+1)
        
        left_tensors = self.left_tensors
        right_tensors = self.right_tensors
        schmidt_matrix = self.schmidt_matrix
        
        sX = np.array([[0.,1.],[1.,0.]])
        sY = np.array([[0, -1j], [1j, 0]])
        gate0 = expm(-1j*dt*(np.kron(sX, sX)+ 0.*np.kron(sY, sY))).reshape(2, 2, 2, 2) #(np.kron(sX, sX) + 0* np.kron(sY, sY)).reshape(2, 2, 2, 2)
        
        indx = 0
        while indx < len(right_tensors) + len(left_tensors):
            print("---- ",indx," ---")
            M1 = right_tensors[0] 
            M2 = left_tensors[-1] 
            # print(schmidt_matrix)print(self.schmidt_matrix)
            M = np.tensordot(M2, self.schmidt_matrix, axes=([2],[0])) #(left_tensors[0], s_matrix, axes=([2],[0]))  
            M = np.tensordot(M, M1, axes=([2],[0]))
            #print(np.shape(M))
            #print(np.shape(gate0))
            Mf = np.tensordot(M, gate0, axes=([1,2],[2, 3]))
            Mf = np.transpose(Mf, (0, 2, 3, 1))
            #print(np.shape(Mf))
            A, schmidt_matrix, B = mps_svd(Mf, chi_max=4 , threshold=None)
            
            del right_tensors[:1]
            del left_tensors[-1:]
            left_tensors.append(A)
            right_tensors.insert(0,B)
            self.schmidt_matrix=schmidt_matrix

            #print(len(left_tensors), " loop", len(right_tensors))
            
            if indx < len(right_tensors) + len(left_tensors) - 2:
                #MPS.center_shift(self, +2)
                self.center_shift(+2)
            indx +=2
        
        return #left_tensors, right_tensors, schmidt_matrix #
    
    def XX_TEBD_E(self, dt): # I can not do it in class
        
        self.center_shift( len(self.right_tensors)-2)
        
        left_tensors = self.left_tensors
        right_tensors = self.right_tensors
        schmidt_matrix = self.schmidt_matrix
        
        # print([aa.shape for aa in left_tensors],"  ",np.shape(schmidt_matrix),"  ",[aa.shape for aa in right_tensors])
        
        sX = np.array([[0.,1.],[1.,0.]])
        sY = np.array([[0, -1j], [1j, 0]])
        gate0 = expm(-1j*dt*(np.kron(sX, sX)+ 0.*np.kron(sY, sY))).reshape(2, 2, 2, 2)#expm(1j*dt*(np.kron(sX, sX)+ 0.*np.kron(sY, sY))).reshape(2, 2, 2, 2)
        
        indx = 0
        while indx < len(right_tensors) + len(left_tensors)-2:

            print([aa.shape for aa in left_tensors],"  ",np.shape(schmidt_matrix),"  ",[aa.shape for aa in right_tensors])
            
            print("---- ",indx," ---")
            M1 = right_tensors[0] 
            M2 = left_tensors[-1]
            print(schmidt_matrix)
            print(self.schmidt_matrix)
            M = np.tensordot(M2, self.schmidt_matrix, axes=([2],[0])) #(left_tensors[0], s_matrix, axes=([2],[0]))  
            M = np.tensordot(M, M1, axes=([2],[0]))
            # print(np.shape(M))
            Mf = np.tensordot(M, gate0, axes=([1,2],[2,3]))
            Mf = np.transpose(Mf, (0, 3, 2, 1))
            print("Mf: ",np.shape(Mf))
            
            A, schmidt_matrix, B = mps_svd(Mf, chi_max=4)
            
            print("A - S - B: ",np.shape(A), np.shape(schmidt_matrix), np.shape(B)) 
            
            del right_tensors[:1]
            del left_tensors[-1:]
            left_tensors.append(A)
            right_tensors.insert(0,B)
            self.schmidt_matrix=schmidt_matrix

            #print(len(left_tensors), " loop", len(right_tensors))
            
            if indx < len(right_tensors) + len(left_tensors) - 4:
                 MPS.center_shift(self, -2)
            indx +=2
            print('\n')
        
        return #left_tensors, right_tensors, schmidt_matrix #
        
        



if __name__ == "__main__":
    "some tests"
    #mps_state = random_chi(5)
