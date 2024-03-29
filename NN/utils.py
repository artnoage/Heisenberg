import torch
import numpy as np
import math
import time

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
    if x.shape[-1]==3 and is_real_number(l):
        lten=torch.tensor([l,l,l**2])
        _, lten=match(x,lten)
        return lten*x
    else:
        print("something wrong with dimensions")
        return 

def spherical_to_cartesian(spherical_coords):
    if spherical_coords.size(-1) not in [3, 4]:
        raise ValueError("The innermost dimension must be 3 or 4.")
    elif spherical_coords.size(-1) == 3:
        # Create a tensor of ones with the same size except for the last dimension
        ones = torch.ones_like(spherical_coords[..., :1])
        # Concatenate the tensor of ones to the original tensor along the last dimension
        spherical_coords = torch.cat((ones,spherical_coords), -1)
    
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
    vs=v*s
    cosvs_series = vs/2 - vs**3/24 + vs**5/720 - vs**7/40320 + vs**9/3628800 - vs**11/479001600 + vs**13/87178291200 - vs**15/20922789888000 + vs**17/6402373705728000 
    sinvs_series = 1 - vs**2/6 + vs**4/120 - vs**6/5040 + vs**8/362880 - vs**10/39916800 + vs**12/6227020800 - vs**14/1307674368000 + vs**16/355687428096000 - vs**18/121645100408832000 
    sinvssquare_series= -vs/6 + vs**3/120 - vs**5/5040 + vs**7/362880 - vs**9/39916800 + vs**11/6227020800 - vs**13/1307674368000 + vs**15/355687428096000 - vs**17/121645100408832000 
    # Compute xi, eta, t for both cases (v != 0 and v == 0)
    xi =  sin_theta * cosvs_series*s*r + cos_theta * sinvs_series*s*r 
    eta = - cos_theta * cosvs_series*s*r + sin_theta * sinvs_series*s*r
    t = -2*sinvssquare_series*s**2*r**2
    
    # Stack the computed values to create the Cartesian coordinates tensor
    cartesian_coords = torch.stack((xi, eta, t), dim=-1)
    
    return cartesian_coords

def cartesian_to_spherical(cartesian):
    cartesian_to_spherical_model = torch.jit.load('NN/coordinate_change_NN.pth',map_location=cartesian.device).to(cartesian.dtype)
    with torch.no_grad():
        prediction = cartesian_to_spherical_model(cartesian)
    return prediction





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
    H_inv_model = torch.jit.load('NN/H_inv_NN.pth',map_location=data.device).to(data.dtype)
    with torch.no_grad():
        prediction = H_inv_model(data)
    return prediction


def norm_cc_H(input):
    xi=input[...,0].unsqueeze(-1)
    eta=input[...,1].unsqueeze(-1)
    t=input[...,2].unsqueeze(-1)
    zeta = torch.complex(xi, eta)  # Construct the complex number zeta
    abs_zeta_sq = torch.abs(zeta)**2
    term1 = t * torch.sin(torch.pi * H_inv_tensor(t / abs_zeta_sq))/torch.abs(zeta)
    term2 = torch.abs(zeta) * torch.cos(torch.pi *H_inv_tensor(t / abs_zeta_sq))
    return term1  + term2

def norm_cc(input):
    norm_cc_model = torch.jit.load('NN/norm_cc_NN.pth',map_location=input.device).to(input.dtype)
    with torch.no_grad():
        prediction = norm_cc_model(input)
    return prediction

def d_cc_H(input1,input2):
    operated=op(-input2,input1)
    return norm_cc_H(operated)

def d_cc(input1,input2):
    operated=op(-input2,input1)
    return norm_cc(operated)


def kernel_unintegrated(input_tensor):
    # Assuming the last dimension of the input_tensor is 4, in the order: h, R, t, y
    h = input_tensor[..., 0]  # Extracts h
    rsquare = input_tensor[..., 1]    # Extracts R^2=\xi^2+\eta^2
    t = input_tensor[..., 2]    # Extracts t
    y = input_tensor[..., 3]   # Extracts y
   
    # Compute the expression
    part1  = (1 / (4 * torch.pi * h)) ** 2
    part2 = torch.where(y == 0, torch.tensor(1.0), (2 * y) / torch.sinh(2 * y))
    part3  = torch.cos((t * y) / (2*h))
    part4a = torch.where(y == 0, torch.tensor(1.0), (2 * y)/torch.tanh(2 * y))
    part4b  = -(rsquare / (4 * h)) * (part4a)
    part4 =    torch.exp(part4b)
    result = part1*part2 * part3 * part4
    return result


def log_kernel(input):
    logkernel_model = torch.jit.load('NN/log_kernel_NN.pth',map_location=input.device).to(input.dtype)
    with torch.no_grad():
        prediction = logkernel_model(input)
    return prediction

def kernel(input):
    v=log_kernel(input)
    h=input[...,0].unsqueeze(1)
    ker=-v/(4*h)
    ker=torch.exp(ker)
    return ker

def log_kernel_cal(data):
    precision = 10**6
    ret = torch.zeros(len(data), dtype=data.dtype, device=data.device)
    
    exp_a_initial = np.exp(20)
    exp_b_initial = np.exp(15)
    
    for element_idx, input_tensor in enumerate(data):
        start_time = time.time()
        
        a = exp_a_initial
        b = exp_b_initial
        j = 1
        h = input_tensor[0].item()
        
        while abs(h * np.log(a / b)) > 0.0015 and time.time() - start_time < 180:
            B = 0
            for i in range(int(10 * ((3.6)**j))):
                # Compute the y_values and expanded tensor outside the innermost loop if possible
                y_values = torch.linspace(i / (3**j), (i + 1) / (3**j), precision, dtype=input_tensor.dtype, device=input_tensor.device)
                expanded = input_tensor.repeat(precision, 1)
                
                # Assuming kernel_unintegrated can process batches, call it once per iteration
                combined_tensor = torch.cat((expanded, y_values.unsqueeze(1)), dim=1)
                A = kernel_unintegrated(combined_tensor)
                A_mean = torch.mean(A)
                B += A_mean.item() / (3**j)
            
            B = (2 * B)
            a = b
            b = B
            j += 1
            
            if b <=0 :
                a = exp_a_initial
                b = exp_b_initial
                
        elapsed_time = time.time() - start_time
        if (math.isnan(B) or elapsed_time > 180) and abs(h * np.log(a / b)) > 0.0015:
            ret[element_idx] = 0
            print("Time ran out")
        else:
            trials = max(j - 2, 1)
            ret[element_idx] = -4 * h * np.log(B)
            if trials>1:
                print(f"I finished return after {trials} trials, we get", -4 * h * np.log(B))
    
    return ret.unsqueeze(1)

    




