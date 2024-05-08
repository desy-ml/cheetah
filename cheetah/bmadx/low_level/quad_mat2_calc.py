import torch
    
def quad_mat2_calc(k1, length, rel_p):
    """Returns 2x2 transfer matrix elements aij and the
    coefficients to calculate the change in z position.
    Input: 
        k1_ref -- Quad strength: k1 > 0 ==> defocus
        length -- Quad length
        rel_p -- Relative momentum P/P0
    Output:
        a11, a12, a21, a22 -- transfer matrix elements
        c1, c2, c3 -- second order derivatives of z such that 
                    z = c1 * x_0^2 + c2 * x_0 * px_0 + c3* px_0^2
    **NOTE**: accumulated error due to machine epsilon. REVISIT
    """ 
    eps = 2.220446049250313e-16  # machine epsilon to double precission
    
    sqrt_k = torch.sqrt(torch.absolute(k1)+eps)
    sk_l = sqrt_k * length
    
    cx = torch.cos(sk_l) * (k1<=0) + torch.cosh(sk_l) * (k1>0) 
    sx = (torch.sin(sk_l)/(sqrt_k))*(k1<=0) + (torch.sinh(sk_l)/(sqrt_k))*(k1>0)
        
    a11 = cx
    a12 = sx / rel_p
    a21 = k1 * sx * rel_p
    a22 = cx
        
    c1 = k1 * (-cx * sx + length) / 4
    c2 = -k1 * sx**2 / (2 * rel_p)
    c3 = -(cx * sx + length) / (4 * rel_p**2)

    return [[a11, a12], [a21, a22]], [c1, c2, c3]