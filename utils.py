import torch
from scipy.optimize import newton
import numpy as np
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
    if x.shape==y.shape:
        z=torch.zeros_like(x)
        z[...,0]=x[...,0]+y[...,0]
        z[...,1]=x[...,1]+y[...,1]
        z[...,2]=x[...,2]+y[...,2] +2*(x[...,1]*y[...,0]-x[...,0]*y[...,1])
        return z
    else:
        print("shapes do not match")
        return 


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
    loaded_model = torch.jit.load('H_inv.pth',map_location=data.device).float()
    prediction = loaded_model(data).flatten()
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

def d_cc(input1,input2):
    operated=op(-input2,input1)
    return norm_cc(operated)

def Kernel_unintegrated(input_tensor):
    # Assuming the last dimension of the input_tensor is 4, in the order: tau, y, t, r
    h = input_tensor[..., 0]  # Extracts h
    rsquare = input_tensor[..., 1]    # Extracts r
    t = input_tensor[..., 2]    # Extracts t
    y = input_tensor[..., 3]   # Extracts y
   
    # Compute the expression
    part1 = (1 / (4 * torch.pi * h)) ** 2
    part2 = torch.where(y == 0, torch.tensor(1.0), (2 * y) / torch.sinh(2 * y))
    part3 = torch.cos((t * y) / 2*h)
    part4a=torch.where(y == 0, torch.tensor(1.0), (2 * y) / torch.tanh(2 * y))
    part4 = torch.exp(-((rsquare) / (4 * h)) * (part4a))

   
    result = part1 * part2 * part3 * part4
    # Ensure the last dimension is 1 by summing or averaging if needed
    # Here, the last dimension is already 1 due to the operations, so we can return the result directly
    return result

def Kernel(input_tensor,precision=4):
    original_tuples_expanded=input_tensor.unsqueeze(1)
    l=15
    B=0
    for j in range(-l,l):
        y_values = torch.linspace(j, j+1, precision).to(input_tensor.device)
        new_points_expanded = y_values.unsqueeze(0).unsqueeze(2)
        combined_tensor = torch.cat((original_tuples_expanded.expand(-1, precision, -1), new_points_expanded.expand(input_tensor.shape[0], -1, -1)), dim=2)
        torch.set_printoptions(threshold=10000)
        print(combined_tensor)
        exit()
        A=Kernel_unintegrated(A)
        A=torch.mean(A,dim=0)
        B=B+A      
    return B

