import torch

from cheetah.bmadx.structures import Particle
from cheetah.bmadx.low_level.sqrt_one import sqrt_one
    
def track_a_drift(p_in, drift):
    """Tracks the incoming Particle p_in though drift element
    and returns the outgoing particle. 
    See Bmad manual section 24.9 
    """
    L = drift.L
    
    s = p_in.s
    p0c = p_in.p0c
    mc2 = p_in.mc2
    
    x, px, y, py, z, pz = p_in.x, p_in.px, p_in.y, p_in.py, p_in.z, p_in.pz
    
    P = 1 + pz            # Particle's total momentum over p0
    Px = px / P           # Particle's 'x' momentum over p0
    Py = py / P           # Particle's 'y' momentum over p0
    Pxy2 = Px**2 + Py**2  # Particle's transverse mometum^2 over p0^2
    Pl = torch.sqrt(1-Pxy2)     # Particle's longitudinal momentum over p0
    
    x = x + L * Px / Pl
    y = y + L * Py / Pl
    
    # z = z + L * ( beta/beta_ref - 1.0/Pl ) but numerically accurate:
    dz = L * (sqrt_one((mc2**2 * (2*pz+pz**2))/((p0c*P)**2 + mc2**2))
                + sqrt_one(-Pxy2)/Pl)
    z = z + dz
    s = s + L

    return Particle(x, px, y, py, z, pz, s, p0c, mc2)