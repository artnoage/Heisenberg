import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import *
import icecream as ic
import time
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device="cpu"


def optimality_test_translation(transl,point_num=3,test_points=50000):
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




def optimality_test_dilation(dil,point_num=3,test_points=50000):
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





def varadham_test(h,test_points=100000):
    for i in range(1):
        s_values = torch.rand(test_points,device=device,dtype=torch.float64)
        v_values = (2*torch.pi)*(2*torch.rand(test_points,device=device,dtype=torch.float64)-1)  
        theta_values = torch.rand(test_points,device=device,dtype=torch.float64)*2*torch.pi  # 45 degrees in radians
        r_values = 2*torch.rand(test_points,device=device,dtype=torch.float64) # r should be positive
        spherical_coords = torch.stack([s_values, v_values, theta_values, r_values],dim=1)
        cartesian_coords = spherical_to_cartesian(spherical_coords)
        ground_truth=(spherical_coords[:,0]*spherical_coords[:,3]).unsqueeze(1)
        h=h*torch.ones_like(s_values) #smaller is 0.005
        xi=cartesian_coords[...,0]
        eta=cartesian_coords[...,1]
        rsquare= (xi**2+eta**2)
        t=cartesian_coords[...,2]
        data= torch.stack([h, rsquare, t],dim=1)
        print(torch.mean(torch.abs(log_kernel(data)-ground_truth)))
        

    return

def norm_test(test_points=50000):
    s_values = torch.rand(test_points,device=device,dtype=torch.float64)
    v_values = (2*torch.pi-0.01)*(2*torch.rand(test_points,device=device,dtype=torch.float64)-1)  
    theta_values = torch.rand(test_points,device=device,dtype=torch.float64)*2*torch.pi  # 45 degrees in radians
    r_values = 2*torch.rand(test_points,device=device,dtype=torch.float64) # r should be positive
    spherical_coords = torch.stack([s_values, v_values, theta_values, r_values],dim=1)
    cartesian_coords = spherical_to_cartesian(spherical_coords)
    ground_truth=(spherical_coords[:,0]*spherical_coords[:,3]).unsqueeze(1)
    norm1=norm_cc(cartesian_coords)
    diff1=torch.abs(norm1-ground_truth)
    norm2=norm_cc_H(cartesian_coords)
    diff2=torch.abs(norm2-ground_truth)
    norm3=cartesian_to_spherical(cartesian_coords)[:,2].unsqueeze(-1)
    diff3=torch.abs(norm3-ground_truth)
    print(torch.max(diff1),torch.max(diff2),torch.max(diff3))
    print(torch.mean(diff1),torch.mean(diff2),torch.mean(diff3))

def coordinate_change_test(test_points=50000):
    s_values = torch.rand(test_points,device=device,dtype=torch.float64)
    v_values = (2*torch.pi)*(2*torch.rand(test_points,device=device,dtype=torch.float64)-1)  
    theta_values = torch.rand(test_points,device=device,dtype=torch.float64)*2*torch.pi  # 45 degrees in radians
    r_values = 2*torch.rand(test_points,device=device,dtype=torch.float64) # r should be positive
    spherical_coords = torch.stack([s_values, v_values, theta_values, r_values],dim=1)
    cartesian_coords = spherical_to_cartesian(spherical_coords)
    ground_truth=(spherical_coords[:,0]*spherical_coords[:,3]).unsqueeze(1)
    spherical_coords_inv=cartesian_to_spherical(cartesian_coords)
    norm=spherical_coords_inv[:,2].unsqueeze(1)
    diff=torch.abs(norm-ground_truth)
    print(torch.mean(diff))


def increment_monotonicity_test(test_points=50000):
    s_values = torch.ones(test_points,device=device,dtype=torch.float64)
    v_values =(2*torch.pi-0.2)*(2*torch.rand(test_points,device=device,dtype=torch.float64)-1) 
    theta_values = torch.rand(test_points,device=device,dtype=torch.float64)*2*torch.pi  # 45 degrees in radians
    r_values = torch.rand(test_points,device=device,dtype=torch.float64) # r should be positive
    spherical_coords = torch.stack([s_values, v_values, theta_values, r_values],dim=1)
    cartesian_coords = spherical_to_cartesian(spherical_coords)
    x1=cartesian_coords[0:10000,:]
    x2=cartesian_coords[10000:20000,:]
    y1=cartesian_coords[20000:30000,:]
    y2=cartesian_coords[30000:40000,:]
    x01=x1
    y01=y1
    x02=x2
    y02=y2
    x1x2= op(-x01,x02)
    y1y2= op(-y01,y02)
    z01=op(-y01,x01)
    z12=op(-y1y2,x1x2)
    z02=op(-y02,x02)
    tuples=torch.concat([x01,x02,y01,y02],dim=1)
    tuples2=torch.concat([z02,z01,z12],dim=1)

def kernel_sum_test(h):
    gridsize=0.007
    values = torch.arange(-5, 5, gridsize,device="cpu",dtype=torch.float32)
    xi, eta, t = torch.meshgrid(values, values, values)
    xi=xi.reshape(-1)
    eta=eta.reshape(-1)
    t=t.reshape(-1)
    print(xi.shape)
    triples = torch.cat((xi.unsqueeze(-1), eta.unsqueeze(-1), t.unsqueeze(-1)), dim=-1)
    xi, eta, t = triples.unbind(dim=-1)
    xi_squared = xi ** 2
    eta_squared = eta ** 2
    first_var = xi_squared + eta_squared
    # Create a tensor of couples [x^2 + y^2, z]
    couples = torch.stack([first_var, t], dim=-1)
    h_values = h*torch.ones_like(couples[..., 0:1])
    input = torch.cat([h_values, couples], dim=-1)
    chunk_size = 500000
    num_chunks = (input.shape[0] + chunk_size - 1) // chunk_size
    value = 0
    print(num_chunks)
    for i in range(num_chunks):
        print(i)
        start = i * chunk_size
        end = min((i + 1) * chunk_size, input.shape[0])
        chunk = input[start:end].to(device=device)
        processed_chunk = kernel(chunk)
        value=value+torch.sum(processed_chunk*gridsize**3)
        print(value)
    print(value)


#norm_test()
#coordinate_change_test()
#kernel_sum_test(0.8)
varadham_test(0.3)
varadham_test(0.2)
varadham_test(0.1)
varadham_test(0.08)
varadham_test(0.008)
varadham_test(0.006)
varadham_test(0.005)
