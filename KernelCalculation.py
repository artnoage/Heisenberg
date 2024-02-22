import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import *
import time
device = "cpu"

epsilon=0.1
h_value = (1-epsilon)*torch.rand(2).to(device).to(dtype=torch.float64)+epsilon
r_value =  5*torch.rand(1).to(device).to(dtype=torch.float64)  # r should be positive
t_value = 10*(torch.rand(1).to(device).to(dtype=torch.float64)-1/2) # t should be positive
point=torch.cartesian_prod(h_value,r_value,t_value)
Kernel(point)