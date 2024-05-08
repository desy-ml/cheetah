import torch
from cheetah.bmadx.structures import Particle
    
def offset_particle_set(x_offset, y_offset, tilt, p_lab):
    """Transform from lab to element coords.
    See Bmad manual (2022-11-06) sections 5.6.1, 15.3.1 and 24.2
    **NOTE**: transverse only as of now.
    """
    s = torch.sin(tilt)
    c = torch.cos(tilt)
    x_ele_int = p_lab.x - x_offset
    y_ele_int = p_lab.y - y_offset
    x_ele = x_ele_int*c + y_ele_int*s
    y_ele = -x_ele_int*s + y_ele_int*c
    px_ele = p_lab.px*c + p_lab.py*s
    py_ele = -p_lab.px*s + p_lab.py*c
    
    return Particle(x_ele, px_ele, y_ele, py_ele, p_lab.z, p_lab.pz,
                    p_lab.s, p_lab.p0c, p_lab.mc2)

def offset_particle_unset(x_offset, y_offset, tilt, p_ele):
    """Transforms from element body to lab coords.
    See Bmad manual (2022-11-06) sections 5.6.1, 15.3.1 and 24.2
    **NOTE**: transverse only as of now.
    """
    s = torch.sin(tilt)
    c = torch.cos(tilt)
    x_lab_int = p_ele.x*c - p_ele.y*s
    y_lab_int = p_ele.x*s + p_ele.y*c
    x_lab = x_lab_int + x_offset
    y_lab = y_lab_int + y_offset
    px_lab = p_ele.px*c - p_ele.py*s
    py_lab = p_ele.px*s + p_ele.py*c
    
    return Particle(x_lab, px_lab, y_lab, py_lab, p_ele.z, p_ele.pz,
                    p_ele.s, p_ele.p0c, p_ele.mc2)