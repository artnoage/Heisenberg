import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import *
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epsilon=0.00001
s_values = torch.rand(100).to(device)

v_values = torch.rand(100).to(device)*(2*torch.pi-epsilon) - 2*torch.pi + epsilon/2 

theta_values = torch.rand(100).to(device)*2*torch.pi  # 45 degrees in radians

r_values = torch.rand(100).to(device) # r should be positive

 # 100 random points for s, scaled for a wider range
# Create a tensor of shape [100, 4] for spherical coordinates with the same v, theta, r

spherical_coords = torch.cartesian_prod(s_values, v_values, theta_values, r_values)
print(spherical_coords.shape())
test=spherical_coords[:,3]*spherical_coords[:,4]
# Convert to Cartesian coordinates
if  True:
    cartesian_coords = spherical_to_cartesian(spherical_coords)
    norm=norm_cc(cartesian_coords)
    print(torch.max(torch.abs(norm-s_values)))


if False:
    s_values = torch.rand(10000).to(device)
    s_values= 0.95*s_values/s_values.max()+0.005
    values=H(s_values)
    prediction2= H_inv_tensor(values)
    print(prediction2)
    print(torch.max(torch.abs(s_values-prediction2)))
    