import torch
from scipy.optimize import newton
import numpy as np
import time
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def spherical_to_cartesian(spherical_coords):
    """
    Convert spherical coordinates to Cartesian coordinates for higher-dimensional tensors.
    
    spherical_coords: A tensor where the last dimension contains:
                      s (time parameter), v (parameter), theta (angle in radians), r (radius).
    
    Returns a tensor with the same dimensionality as spherical_coords, where the last dimension
    has been replaced with Cartesian coordinates [x, y, z].
    """
    # Unpack the last dimension of the spherical coordinates
    s, v, theta, r = spherical_coords.unbind(dim=-1)
    
    # Pre-compute cos(theta) and sin(theta) for efficiency
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    
    # Compute xi, eta, t for both cases (v != 0 and v == 0)
    xi = torch.where(v != 0, 
                     ((sin_theta * (1 - torch.cos(v * s)) + cos_theta * torch.sin(v * s)) / v) * r, 
                     r * cos_theta * s)
    eta = torch.where(v != 0, 
                      ((-cos_theta * (1 - torch.cos(v * s)) + sin_theta * torch.sin(v * s)) / v )* r, 
                      r * sin_theta * s)
    t = torch.where(v != 0, 
                    (2 * (v * s - torch.sin(v * s)) / v**2 )* r**2, 
                    torch.zeros_like(s))
    
    # Stack the computed values to create the Cartesian coordinates tensor
    cartesian_coords = torch.stack((xi, eta, t), dim=-1)
    
    return cartesian_coords

def is_real_number(input_value):
    try:
        float(input_value)  # Attempt to convert to float
        return True
    except ValueError:
        return False

def can_extend_to_match(a_shape, b_shape):
    """
    Check if tensor A can be extended to match the shape of tensor B.
    :param a_shape: Shape of tensor A (as a tuple).
    :param b_shape: Shape of tensor B (as a tuple).
    :return: True if A can be extended to match B, False otherwise.
    """
    # Reverse the shapes to start comparison from the trailing dimensions
    a_shape_reversed = a_shape[::-1]
    b_shape_reversed = b_shape[::-1]
    
    # Iterate over the shapes from the end
    for a_dim, b_dim in zip(a_shape_reversed, b_shape_reversed):
        if a_dim != b_dim and a_dim != 1 and b_dim != 1:
            return False  # Cannot be extended due to incompatible dimension
    
    # If A has more dimensions than B, ensure all extra dimensions in A are 1
    if len(a_shape) > len(b_shape):
        if any(dim != 1 for dim in a_shape_reversed[len(b_shape_reversed):]):
            return False  # Extra dimensions in A are not 1, cannot be extended
    
    # If all checks pass, A can be extended to match B
    return True


def match(x,y):
    if can_extend_to_match(x.shape,y.shape):
        x = x.expand(y.shape)
        return x, y
    elif can_extend_to_match(y.shape,x.shape):
        y = y.expand(x.shape)
        return x, y
    else:
        print("Shapes do not match")
        return  

def op(x,y):
    x,y=match(x,y)
    z=torch.zeros_like(x)
    z[...,0]=x[...,0]+y[...,0]
    z[...,1]=x[...,1]+y[...,1]
    z[...,2]=x[...,2]+y[...,2] +2*(x[...,1]*y[...,0]-x[...,0]*y[...,1])
    return z


def dilation(l,x):
    if is_real_number(l):
        lten=torch.tensor([l,l,l**2])
        _, lten=match(x,lten)
        return lten*x
    elif x.shape==l.shape:
        return lten*x
    else:
        print("something wrong with dimensions")
        return 

def H(r):
    a=torch.zeros_like(r)
    H=torch.where(r !=0 , (2 * torch.pi* r - torch.sin(2 * torch.pi * r)) / (1 - torch.cos(2 * torch.pi * r)), 0)
    return H



def H_inv_tensor(data):
    if data.dim()==1:
        data=data.unsqueeze(-1)
    if data.shape[-1]!=1:
        print("input is wrong")
        return
    H_inv_model = torch.jit.load('NN/HNN.pth',map_location=data.device).to(data.dtype)
    prediction = H_inv_model(data).flatten()
    return prediction


def norm_cc(input):
    xi=input[...,0]
    eta=input[...,1]
    t=input[...,2]
    zeta = torch.complex(xi, eta)  # Construct the complex number zeta
    abs_zeta_sq = torch.abs(zeta)**2
    term1 = t * torch.sin(torch.pi * H_inv_tensor(t / abs_zeta_sq))/torch.abs(zeta)
    term2 = torch.abs(zeta) * torch.cos(torch.pi *H_inv_tensor(t / abs_zeta_sq))
    return term1  + term2

def norm_ccNN(input):
    norm_cc_model = torch.jit.load('NN/dccNN.pth',map_location=input.device).to(input.dtype)
    prediction = norm_cc_model(input).flatten()
    return prediction

def d_cc(input1,input2):
    operated=op(-input2,input1)
    return norm_cc(operated)

def kernel(data,timestep):
    timetensor=timestep*torch.ones([data.shape[0],1])
    x=data[...,0]
    y=data[...,1]
    r=(x**2+y**2).unsqueeze(-1)
    t=data[...,2].unsqueeze(-1)
    data=torch.cat([timetensor,r,t],dim=1)
    kernel_model = torch.jit.load('NN/KernelNN.pth',map_location=data.device).to(data.dtype)
    prediction = kernel_model(data).flatten()
    return prediction
    




