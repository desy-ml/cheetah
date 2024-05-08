import torch
from cheetah.particles import ParticleBeam
from cheetah.bmadx import (
    M_ELECTRON,
    Particle,
    Quadrupole,
    track_a_quadrupole
)


def cheetah_to_bmad_coords(cheetah_coords: torch.Tensor, p0c:torch.Tensor) -> torch.Tensor :
    """Transforms Cheetah coordinates to Bmad coordinates.
    :param cheetah_coords: 7-dimensional particle vectors in Cheetah coordinates.
    :param p0c: reference momentum in eV/c. 
    """
    bmad_coords = torch.zeros_like(cheetah_coords)
    
    # Cheetah coords:
    x = cheetah_coords[..., 0]
    xp = cheetah_coords[..., 1]
    y = cheetah_coords[..., 2]
    yp = cheetah_coords[..., 3]
    s = cheetah_coords[..., 4]
    p = cheetah_coords[..., 5]

    # intermediate calcs:
    delta_E = p * p0c
    mc2 = torch.ones_like(p0c) * M_ELECTRON
    E0 = torch.sqrt(mc2**2 + p0c**2)
    E = delta_E + E0
    P = torch.sqrt( E**2 - mc2**2 )
    pz = (P - p0c) / p0c
    px = xp * torch.sqrt( (1+pz)**2 / (1+xp**2+yp**2) )
    py = yp * torch.sqrt( (1+pz)**2 / (1+xp**2+yp**2) )
    z = s

    # final Bmad coords: 
    bmad_coords[..., 0] = x
    bmad_coords[..., 1] = px
    bmad_coords[..., 2] = y
    bmad_coords[..., 3] = py
    bmad_coords[..., 4] = z
    bmad_coords[..., 5] = pz
    
    return bmad_coords


def bmad_to_cheetah_coords(bmad_coords: torch.Tensor, p0c:torch.Tensor) -> torch.Tensor :
    """Transforms Bmad coordinates to Cheetah coordinates.
    :param bmad_coords: 6-dimensional particle vectors in Bmad coordinates.
    :param p0c: reference momentum in eV/c. 
    """
    cheetah_coords = torch.ones(
        (*bmad_coords.shape[:-1], 7), 
        dtype=bmad_coords.dtype, 
        device=bmad_coords.device
    )
    
    # Bmad coords:
    x = bmad_coords[..., 0]
    px = bmad_coords[..., 1]
    y = bmad_coords[..., 2]
    py = bmad_coords[..., 3]
    z = bmad_coords[..., 4]
    pz = bmad_coords[..., 5]

    # intermediate calcs:
    xp = px / torch.sqrt( (1+pz)**2 - px**2 - py**2 )
    yp = py / torch.sqrt( (1+pz)**2 - px**2 - py**2 )
    s = z
    mc2 = torch.ones_like(p0c) * M_ELECTRON
    E0 = torch.sqrt(mc2**2 + p0c**2)
    P = (1+pz)*p0c
    E = torch.sqrt(P**2 + mc2**2)
    p = (E - E0) / p0c 

    # final Cheetah coords:
    cheetah_coords[..., 0] = x
    cheetah_coords[..., 1] = xp
    cheetah_coords[..., 2] = y
    cheetah_coords[..., 3] = yp
    cheetah_coords[..., 4] = s
    cheetah_coords[..., 5] = p
    
    return cheetah_coords


def cheetah_particle_beam_to_bmadx_particle(cheetah_beam: ParticleBeam) -> Particle :
    """"Converts a Cheetah ParticleBeam to a BmadX Particle.
    :param cheetah_beam: Cheetah ParticleBeam.
    """
    mc2 = torch.ones_like(cheetah_beam.energy) * M_ELECTRON
    p0c = torch.sqrt(cheetah_beam.energy**2 - mc2**2)

    bmad_coords = cheetah_to_bmad_coords(cheetah_beam.particles, p0c)
    
    bmadx_particle = Particle(
        x = bmad_coords[..., 0],
        px = bmad_coords[..., 1],
        y = bmad_coords[..., 2],
        py = bmad_coords[..., 3],
        z = bmad_coords[..., 4],
        pz = bmad_coords[..., 5],
        s = torch.zeros_like(p0c),
        p0c = p0c,
        mc2 = mc2
    )

    return bmadx_particle


def bmadx_particle_to_cheetah_beam(bmadx_particle: Particle) -> ParticleBeam :
    """Converts a BmadX Particle to a Cheetah ParticleBeam.
    :param bmadx_particle: BmadX Particle.
    """
    mc2 = torch.ones_like(bmadx_particle.p0c) * M_ELECTRON
    energy = torch.sqrt(bmadx_particle.p0c**2 + mc2**2)

    cheetah_coords = bmad_to_cheetah_coords(
        torch.stack(
            (
                bmadx_particle.x, 
                bmadx_particle.px, 
                bmadx_particle.y, 
                bmadx_particle.py, 
                bmadx_particle.z, 
                bmadx_particle.pz
            ),
            dim = -1
        ), 
        bmadx_particle.p0c
    )

    cheetah_beam = ParticleBeam(
        cheetah_coords,
        energy,
        device = bmadx_particle.x.device,
        dtype = bmadx_particle.x.dtype
    )

    return cheetah_beam


def cheetah_to_bmad_quad(cheetah_quad) -> Quadrupole :
    """Converts a Cheetah Quadrupole to a BmadX Quadrupole."""
    bmadx_quad = Quadrupole(
        L = cheetah_quad.length,
        K1 = cheetah_quad.k1,
        NUM_STEPS = cheetah_quad.num_steps,
        X_OFFSET = cheetah_quad.misalignment[...,0],
        Y_OFFSET = cheetah_quad.misalignment[...,1],
        TILT = cheetah_quad.tilt
    )
    return bmadx_quad


def track_bmadx_quad(
        incoming_cheetah_beam: ParticleBeam, 
        cheetah_quad
) -> ParticleBeam:
    """Tracks a Cheetah beam through a BmadX quadrupole."""
    incoming_bmadx_particle = cheetah_particle_beam_to_bmadx_particle(incoming_cheetah_beam)
    bmadx_quad = cheetah_to_bmad_quad(cheetah_quad)
    outgoing_bmadx_particle = track_a_quadrupole(incoming_bmadx_particle, bmadx_quad)
    outgoing_cheetah_beam = bmadx_particle_to_cheetah_beam(outgoing_bmadx_particle)
    return outgoing_cheetah_beam