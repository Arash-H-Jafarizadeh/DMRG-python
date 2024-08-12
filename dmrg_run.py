"""
    this is for runs/tests
"""
import numpy as np
import scipy.linalg as la
from helper import *
from mps_functions import *
from mpo_functions import *


mps_state = MPS.random_chi(5)
dens_state = MPS.to_dense(mps_state)
print(la.norm(dens_state))
