from bmadx.constants import C_LIGHT

def make_particle_rf_time(lib):
    """Makes function given library lib."""
    sqrt = lib.sqrt
    
    def particle_rf_time(p):
        """Returns rf time of Particle p."""
        beta = (1+p.pz) * p.p0c / sqrt(((1+p.pz)*p.p0c)**2 + p.mc2**2)
        time = - p.z / (beta * C_LIGHT)
        
        return time
    
    return particle_rf_time