import pytest
import torch
import cheetah

def test_cold_uniform_beam_expansion():
    """
    Test that that a cold uniform beam doubles in size in all dimensions when traveling through a drift section having an analytically-predicted length, with space_charge. (cf ImpactX test)
    """

    # Simulation parameters
    num_particles = 10000
    total_charge=torch.tensor(1e-9)
    R0 = torch.tensor(0.001)
    energy=torch.tensor(2.5e8)
    gamma = energy/cheetah.rest_energy
    beta = torch.sqrt(1-1/gamma**2)
    incoming = cheetah.ParticleBeam.uniform_3d_ellispoid(
        num_particles=torch.tensor(num_particles),
        total_charge=total_charge,
        energy = energy,
        radius_x = R0,
        radius_y = R0,
        radius_s = R0/gamma,  # radius of the beam in s direction, in the lab frame
        sigma_xp = torch.tensor(1e-15),
        sigma_yp = torch.tensor(1e-15),
        sigma_p = torch.tensor(1e-15),
    )

    # Initial beam properties
    sig_xi = incoming.sigma_x
    sig_yi = incoming.sigma_y
    sig_si = incoming.sigma_s

    # Compute section lenght
    kappa = 1+(torch.sqrt(torch.tensor(2))/4)*torch.log(3+2*torch.sqrt(torch.tensor(2)))
    Nb = total_charge/cheetah.elementary_charge
    L=beta*gamma*kappa*torch.sqrt(R0**3/(Nb*cheetah.electron_radius))

    segment_space_charge = cheetah.Segment(
    elements=[
        cheetah.Drift(L/6),
        cheetah.SpaceChargeKick(L/3),
        cheetah.Drift(L/3),
        cheetah.SpaceChargeKick(L/3),
        cheetah.Drift(L/3),
        cheetah.SpaceChargeKick(L/3),
        cheetah.Drift(L/6)
    ]
    )
    outgoing_beam = segment_space_charge.track(incoming) 

    # Final beam properties
    sig_xo = outgoing_beam.sigma_x
    sig_yo = outgoing_beam.sigma_y
    sig_so = outgoing_beam.sigma_s
    
    torch.set_printoptions(precision=16)
    assert torch.isclose(sig_xo,2*sig_xi,rtol=2e-2,atol=0.0)
    assert torch.isclose(sig_yo,2*sig_yi,rtol=2e-2,atol=0.0)
    assert torch.isclose(sig_so,2*sig_si,rtol=2e-2,atol=0.0)