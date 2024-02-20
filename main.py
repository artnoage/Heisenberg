import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import *
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Set a fixed value for v, theta, r
v = torch.tensor(0.6).to(device)
theta = torch.tensor(torch.pi/8).to(device)  # 45 degrees in radians
r = torch.tensor(1).to(device) # r should be positive

# Randomly select 100 points for time (s)
s_values = torch.rand(1000).to(device)
s_values= s_values/s_values.max()  # 100 random points for s, scaled for a wider range
# Create a tensor of shape [100, 4] for spherical coordinates with the same v, theta, r
spherical_coords = torch.stack((s_values, torch.full_like(s_values, v), 
                                torch.full_like(s_values, theta), torch.full_like(s_values, r)), dim=1)

# Convert to Cartesian coordinates
if  False:
    cartesian_coords = spherical_to_cartesian(spherical_coords)
    norm=norm_cc(cartesian_coords)
    s_values=0.8*s_values + 0.03
    print(torch.max(torch.abs(norm-s_values)))

if True:
    s_values = torch.rand(10000).to(device)
    s_values= 0.95*s_values/s_values.max()+0.005
    values=H(s_values)
    prediction2= H_inv_tensor(values)
    print(prediction2)
    print(torch.max(torch.abs(s_values-prediction2)))
    