from bmadx.structures import Particle
from bmadx.low_level.sqrt_one import make_sqrt_one

def make_apply_energy_kick(lib):
    """Makes function given library lib."""
    sqrt = lib.sqrt
    sqrt_one = make_sqrt_one(lib)
    
    def apply_energy_kick(dE, p_in):
        """Changes the energy of a particle by dE."""
        z, pz, p0c, mc2 = p_in.z, p_in.pz, p_in.p0c, p_in.mc2
        
        pc = (1 + pz) * p0c
        beta_old = (1+pz) * p0c / sqrt(((1+pz)*p0c)**2 + mc2**2)
        E_old =  pc / beta_old
        
        E_new = E_old + dE
        
        pz = pz + (1 + pz) * sqrt_one((2*E_old*dE + dE**2)/pc**2)
        pc_new = p0c * (1 + pz)
        beta_new = pc_new / E_new
        z = z * beta_new / beta_old
        
        return Particle(p_in.x, p_in.px, p_in.y, p_in.py, z, pz,
                        p_in.s, p0c, mc2)
    
    return apply_energy_kick