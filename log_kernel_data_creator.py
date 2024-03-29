import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import *
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epsilon=0.005
h_value = 2*torch.rand(10**6,device=device,dtype=torch.float64)+epsilon
r_value = 10*torch.rand(10**6,device=device,dtype=torch.float64)  # r should be positive
t_value = 10*(torch.rand(10**6,device=device,dtype=torch.float64)-1/2) # t should be positive
point=torch.stack([h_value,r_value,t_value],dim=1)
values=torch.zeros_like(h_value,device=device,dtype=torch.float64).unsqueeze(-1)
points_per_iteration=1000
for i in range(1000):
    values[i*points_per_iteration:(i+1)*points_per_iteration,:]=log_kernel_cal(point[i*points_per_iteration:(i+1)*points_per_iteration,:])
    print("you are in the", i, "collection")
    torch.save(torch.cat([point,values],dim=1),"data4.pt")