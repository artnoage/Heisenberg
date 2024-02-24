import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import *
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device="cpu"
for i in range(20):
    epsilon=0.001
    s_values = torch.rand(6).to(device)

    v_values = (2*torch.pi)*torch.clamp(torch.rand(6).to(device)-1,min=-1+epsilon,max=1-epsilon)  

    theta_values = torch.rand(6).to(device)*2*torch.pi  # 

    r_values = torch.rand(6).to(device) # r should be positive

  

    spherical_coords = torch.stack([s_values, v_values, theta_values, r_values],dim=1)
    test=spherical_coords[:,0]*spherical_coords[:,3]
# Convert to Cartesian coordinates
    if  True:
        cartesian_coords = spherical_to_cartesian(spherical_coords)
        norm2=norm_ccNN(cartesian_coords)
        print((torch.abs(norm2-test)))


    if False:
        s_values = torch.rand(10000).to(device)
        s_values= 0.95*s_values/s_values.max()+0.005
        values=H(s_values)
        prediction2= H_inv_tensor(values)
        print(prediction2)
        print(torch.max(torch.abs(s_values-prediction2)))
    