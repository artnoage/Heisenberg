import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import *
import time
import functools

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def optimality_test_dilation(dil,point_num=3,test_points=10):
    points=torch.zeros([point_num,test_points,3])
    for i in range(point_num):
        x_values = 2*torch.rand([test_points,1],device=device,dtype=torch.float64)-1
        y_values = 2*torch.rand([test_points,1],device=device,dtype=torch.float64)-1
        t_values = 2*torch.rand([test_points,1],device=device,dtype=torch.float64)-1  # 45 degrees in radians
        points[i]=torch.cat([x_values,y_values,t_values],dim=1)
    points2=dilation(dil,points)
    left=torch.zeros(test_points)
    right=torch.zeros(test_points)
    for i in range(point_num):
        left=left+ d_cc(points[i],points2[i])**2
    for i in range(point_num-1):
        right=right + d_cc(points[i],points2[i+1])**2
    right=right+ d_cc(points[point_num-1],points2[0])**2
    diff=right-left
    print(diff.shape,points.transpose(0,1).reshape(10,6).shape)
    toprint=torch.cat([points.transpose(0,1).reshape(10,6),diff.unsqueeze(-1)],dim=1)
    return print(toprint)

def optimality_test_translation(transl,point_num=3,test_points=5000):
    points=torch.zeros([point_num,test_points,3])
    for i in range(point_num):
        x_values = 2*torch.rand([test_points,1],device=device,dtype=torch.float64)-1
        y_values = 2*torch.rand([test_points,1],device=device,dtype=torch.float64)-1
        t_values = 2*torch.rand([test_points,1],device=device,dtype=torch.float64)-1  # 45 degrees in radians
        points[i]=torch.cat([x_values,y_values,t_values],dim=1)
    A=transl
    points2=op(A,points)
    left=torch.zeros(test_points)
    right=torch.zeros(test_points)
    for i in range(point_num):
        left=left+ d_cc(points[i],points2[i])**2
    for i in range(point_num-1):
        right=right + d_cc(points[i],points2[i+1])**2
    right=right+ d_cc(points[point_num-1],points2[0])**2
    return print(torch.min(right-left))



a_values = 2*torch.rand([100,1],device=device,dtype=torch.float64)-1
b_values = 2*torch.rand([100,1],device=device,dtype=torch.float64)-1
c_values = 2*torch.rand([100,1],device=device,dtype=torch.float64)-1  # 45 degrees in radians
c_values=torch.ones_like(c_values)
transl=torch.cat([a_values,b_values,c_values],dim=1)

#for i in range(len(transl)):
#    optimality_test_translation(transl[i])

optimality_test_dilation(dil=0.5,point_num=2)