import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import *
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device="cpu"

for i in range(20):
    s_values = torch.rand(1000,device=device,dtype=torch.float64)
    v_values = (2*torch.pi)*(2*torch.rand(1000,device=device,dtype=torch.float64)-1)  
    theta_values = torch.rand(1000,device=device,dtype=torch.float64)*2*torch.pi  # 45 degrees in radians
    r_values = torch.rand(1000,device=device,dtype=torch.float64) # r should be positive
    spherical_coords = torch.stack([s_values, v_values, theta_values, r_values],dim=1)
    test=spherical_coords[:,0]*spherical_coords[:,3]
    cartesian_coords = spherical_to_cartesian(spherical_coords)
# Convert to Cartesian coordinates
    if  True:
        norm1=norm_ccNN(cartesian_coords)
        norm2=norm_cc(cartesian_coords)
        print(torch.max(torch.abs(norm1-test)))
        

    if False:
        A=torch.tensor([0,0,0.25],device=device,dtype=torch.float64)
        print(norm_ccNN(A.unsqueeze(0))/np.sqrt(torch.pi))

    if False:
        h=0.05
        print(torch.min(torch.log(kernel(cartesian_coords,h))),torch.max(torch.log(kernel(cartesian_coords,h))))
        #print(torch.max(torch.abs(-h*torch.log(kernel(cartesian_coords,h))-norm_cc(cartesian_coords)**2))) 
    