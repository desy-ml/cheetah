from cheetah.bmadx.structures import Particle

from cheetah.bmadx.low_level.offset_particle import offset_particle_set, offset_particle_unset
from cheetah.bmadx.low_level.low_energy_z_correction import low_energy_z_correction
from cheetah.bmadx.low_level.quad_mat2_calc import quad_mat2_calc
    
def track_a_quadrupole(p_in, quad):
    """Tracks the incoming Particle p_in though quad element and
    returns the outgoing particle.
    See Bmad manual section 24.15
    """
    l = quad.L
    k1 = quad.K1
    n_step = quad.NUM_STEPS  # number of divisions
    step_len = l / n_step  # length of division
    
    x_off = quad.X_OFFSET
    y_off = quad.Y_OFFSET
    tilt = quad.TILT
    
    b1 = k1 * l
    
    s = p_in.s
    p0c = p_in.p0c
    mc2 = p_in.mc2
    
    # --- TRACKING --- :
    
    par = offset_particle_set(x_off, y_off, tilt, p_in)
    x, px, y, py, z, pz = par.x, par.px, par.y, par.py, par.z, par.pz
    
    for i in range(n_step):
        rel_p = 1 + pz  # Particle's relative momentum (P/P0)
        k1 = b1/(l*rel_p)
        
        tx, dzx = quad_mat2_calc(-k1, step_len, rel_p)
        ty, dzy = quad_mat2_calc( k1, step_len, rel_p)
        
        z = ( z
                + dzx[0] * x**2 + dzx[1] * x * px + dzx[2] * px**2
                + dzy[0] * y**2 + dzy[1] * y * py + dzy[2] * py**2 )
        
        x_next = tx[0][0] * x + tx[0][1] * px
        px_next = tx[1][0] * x + tx[1][1] * px
        y_next = ty[0][0] * y + ty[0][1] * py
        py_next = ty[1][0] * y + ty[1][1] * py
        
        x, px, y, py = x_next, px_next, y_next, py_next
        
        z = z + low_energy_z_correction(pz, p0c, mc2, step_len)
    
    s = s + l
    
    par = offset_particle_unset(x_off, y_off, tilt,
                                Particle(x, px, y, py, z, pz, s, p0c, mc2))
    
    return par