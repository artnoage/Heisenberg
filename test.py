import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import *
import time
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device="cpu"
def varadham_test(h,times):
    for i in range(times):
        s_values = torch.rand(1000,device=device,dtype=torch.float64)
        v_values = (2*torch.pi)*(2*torch.rand(1000,device=device,dtype=torch.float64)-1)  
        theta_values = torch.rand(1000,device=device,dtype=torch.float64)*2*torch.pi  # 45 degrees in radians
        r_values = 10*torch.rand(1000,device=device,dtype=torch.float64) # r should be positive
        spherical_coords = torch.stack([s_values, v_values, theta_values, r_values],dim=1)
        cartesian_coords = spherical_to_cartesian(spherical_coords)
        print(torch.min(torch.log(kernel(cartesian_coords,h))),torch.max(torch.log(kernel(cartesian_coords,h))))
    #print(torch.max(torch.abs(-h*torch.log(kernel(cartesian_coords,h))-norm_cc(cartesian_coords)**2))) 
    return

def NormNN_test(times):
    for i in range(times):
        s_values = torch.rand(1000,device=device,dtype=torch.float64)
        v_values = (2*torch.pi)*(2*torch.rand(1000,device=device,dtype=torch.float64)-1)  
        theta_values = torch.rand(1000,device=device,dtype=torch.float64)*2*torch.pi  # 45 degrees in radians
        r_values = 10*torch.rand(1000,device=device,dtype=torch.float64) # r should be positive
        spherical_coords = torch.stack([s_values, v_values, theta_values, r_values],dim=1)
        cartesian_coords = spherical_to_cartesian(spherical_coords)
        ground_truth=spherical_coords[:,0]*spherical_coords[:,3]
        norm=norm_ccNN(cartesian_coords)
        print(torch.max(torch.abs(norm-ground_truth)))



def varadham_test2(h,times):
    A=torch.zeros([times],device=device,dtype=torch.float64)
    A2=torch.zeros([times],device=device,dtype=torch.float64)
    B=torch.zeros([times],device=device,dtype=torch.float64)
    C=torch.zeros([times],device=device,dtype=torch.float64)
    C2=torch.zeros([times],device=device,dtype=torch.float64)
    for i in range(times):
        t=torch.rand(1,device=device,dtype=torch.float64)
        t2=torch.rand(1,device=device,dtype=torch.float64)
        cartesian_coords = torch.tensor([-0.00192,  0.00266,  4],device=device,dtype=torch.float64).unsqueeze(0)
        #kernel(cartesian_coords,h)
        A[i]=torch.abs(-4*h*torch.log(kernel(cartesian_coords,h)))
        #A2[i]=torch.abs(-4*2*h*torch.log(kernel(cartesian_coords,2*h))-4**2)
        B[i]=1
        C[i]=A[i]/B[i]
        #C2[i]=A2[i]/B[i]
    C=torch.mean(C)
    #C2=torch.mean(C2)
    print(A,C) 
    return

#print(norm_cc(torch.tensor([-0.0192,  0.0266,  0.3395],device=device,dtype=torch.float64).unsqueeze(0)))
#print(spherical_to_cartesian(torch.tensor([1,2*torch.pi-0.2,-torch.pi/3,1])))
#varadham_test2(0.05,1)
#NormNN_test(1)
#A=torch.tensor([-0.1297, -0.3483, -0.0536])
#dilA=dilation(0.5,A)
#B=torch.tensor([0.3646, -0.7645, -0.9866])
#dilB=dilation(0.5,B)
#print(d_cc(A,dilA)**2+d_cc(B,dilB)**2)
#print(d_cc(B,dilA)**2+d_cc(A,dilB)**2)
#print(d_ccNN(A,dilA)**2+d_ccNN(B,dilB)**2)
#print(d_ccNN(B,dilA)**2+d_ccNN(A,dilB)**2)
#NormNN_test(100)
A=torch.load("data2.pt")
for i in range(1000):
    print(A[i])        