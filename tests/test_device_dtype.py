import pytest
import torch

import cheetah
from cheetah.utils import is_mps_available_and_functional


@pytest.mark.parametrize(
    "target_device",
    [
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
        pytest.param(
            torch.device("mps"),
            marks=pytest.mark.skipif(
                not is_mps_available_and_functional(), reason="MPS not available"
            ),
        ),
    ],
)
def test_move_quadrupole_to_device(target_device: torch.device):
    """Test that a quadrupole magnet can be successfully moved to a different device."""
    quad = cheetah.Quadrupole(
        length=torch.tensor(0.2), k1=torch.tensor(4.2), name="my_quad"
    )

    # Test that by default the quadrupole is on the CPU
    assert quad.length.device.type == "cpu"
    assert quad.k1.device.type == "cpu"
    assert quad.misalignment.device.type == "cpu"
    assert quad.tilt.device.type == "cpu"

    # Move the quadrupole to the target device
    quad.to(target_device)

    # Test that the quadrupole is now on the target device
    assert quad.length.device.type == target_device.type
    assert quad.k1.device.type == target_device.type
    assert quad.misalignment.device.type == target_device.type
    assert quad.tilt.device.type == target_device.type


@pytest.mark.parametrize(
    "ElementClass",
    [
        cheetah.Cavity,
        cheetah.Dipole,
        cheetah.Drift,
        cheetah.HorizontalCorrector,
        cheetah.Quadrupole,
        cheetah.RBend,
        cheetah.Sextupole,
        cheetah.Solenoid,
        cheetah.TransverseDeflectingCavity,
        cheetah.Undulator,
        cheetah.VerticalCorrector,
    ],
)
def test_forced_element_dtype(ElementClass):
    """
    Test that the dtype is properly overridden for all element classes.
    """
    element = ElementClass(
        length=torch.tensor(1.0, dtype=torch.float64), dtype=torch.float16
    )

    for buffer in element.buffers():
        assert buffer.dtype == torch.float16


@pytest.mark.parametrize(
    "ElementClass",
    [
        cheetah.Cavity,
        cheetah.Dipole,
        cheetah.Drift,
        cheetah.HorizontalCorrector,
        cheetah.Quadrupole,
        cheetah.RBend,
        cheetah.Sextupole,
        cheetah.Solenoid,
        cheetah.TransverseDeflectingCavity,
        cheetah.Undulator,
        cheetah.VerticalCorrector,
    ],
)
def test_infer_element_dtype(ElementClass):
    """
    Test that the dtype is properly inferred for all element classes.
    """
    element = ElementClass(length=torch.tensor(1.0, dtype=torch.float64))

    for buffer in element.buffers():
        assert buffer.dtype == torch.float64


def test_conflicting_quadrupole_dtype():
    """
    Test that creating a quadrupole with conflicting argument dtypes fails.
    """
    with pytest.raises(AssertionError):
        cheetah.Quadrupole(
            length=torch.tensor(1.0, dtype=torch.float32),
            k1=torch.tensor(10.0, dtype=torch.float64),
        )

    # Ensure that the conflict can be solved by explicit dtype selection
    quad = cheetah.Quadrupole(
        length=torch.tensor(1.0, dtype=torch.float32),
        k1=torch.tensor(10.0, dtype=torch.float64),
        dtype=torch.float16,
    )
    assert quad.length.dtype == torch.float16


def test_change_quadrupole_dtype():
    """
    Test that a quadrupole magnet can be successfully changed to a different dtype.
    """
    quad = cheetah.Quadrupole(
        length=torch.tensor(0.2),
        k1=torch.tensor(4.2),
        name="my_quad",
    )

    # Test that by default the quadrupole is of dtype float32
    assert quad.length.dtype == torch.float32
    assert quad.k1.dtype == torch.float32
    assert quad.misalignment.dtype == torch.float32
    assert quad.tilt.dtype == torch.float32

    # Change the dtype of the quadrupole
    quad.to(torch.float64)

    # Test that the quadrupole is now of dtype float64
    assert quad.length.dtype == torch.float64
    assert quad.k1.dtype == torch.float64
    assert quad.misalignment.dtype == torch.float64
    assert quad.tilt.dtype == torch.float64


@pytest.mark.parametrize(
    "target_device",
    [
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
        pytest.param(
            torch.device("mps"),
            marks=pytest.mark.skipif(
                not is_mps_available_and_functional(), reason="MPS not available"
            ),
        ),
    ],
)
def test_move_particlebeam_to_device(target_device: torch.device):
    """Test that a particle beam can be successfully moved to a different device."""
    beam = cheetah.ParticleBeam.from_parameters(num_particles=100_000)

    # Test that by default the particle beam is on the CPU
    assert beam.particles.device.type == "cpu"
    assert beam.energy.device.type == "cpu"
    assert beam.total_charge.device.type == "cpu"

    beam.to(target_device)

    # Test that the particle beam is now on the target device
    assert beam.particles.device.type == target_device.type
    assert beam.energy.device.type == target_device.type
    assert beam.total_charge.device.type == target_device.type


@pytest.mark.parametrize(
    "BeamClass",
    [
        cheetah.ParameterBeam,
        cheetah.ParticleBeam,
    ],
)
def test_forced_beam_dtype(BeamClass):
    """
    Test that the dtype is properly overriden on beam creation.
    """
    beam = BeamClass.from_parameters(
        mu_x=torch.tensor(1e-5, dtype=torch.float32), dtype=torch.float64
    )
    for buffer in beam.buffers():
        assert buffer.dtype == torch.float64

    beam = BeamClass.from_twiss(
        beta_x=torch.tensor(1.0, dtype=torch.float16),
        beta_y=torch.tensor(2.0, dtype=torch.float64),
        dtype=torch.float32,
    )
    for buffer in beam.buffers():
        assert buffer.dtype == torch.float32


@pytest.mark.parametrize(
    "BeamClass",
    [
        cheetah.ParameterBeam,
        cheetah.ParticleBeam,
    ],
)
def test_infer_beam_dtype(BeamClass):
    """
    Test that the dtype is properly inferred on beam creation.
    """
    beam = BeamClass.from_parameters(mu_x=torch.tensor(1e-5, dtype=torch.float64))
    for buffer in beam.buffers():
        assert buffer.dtype == torch.float64

    beam = BeamClass.from_twiss(
        beta_x=torch.tensor(1.0, dtype=torch.float64),
        beta_y=torch.tensor(2.0, dtype=torch.float64),
    )
    for buffer in beam.buffers():
        assert buffer.dtype == torch.float64


def test_conflicting_particlebeam_dtype():
    """
    Test if creating a ParticleBeam with conflicting argument dtypes fails.
    """
    with pytest.raises(AssertionError):
        cheetah.ParticleBeam.from_twiss(
            beta_x=torch.tensor(1.0, dtype=torch.float32),
            beta_y=torch.tensor(2.0, dtype=torch.float64),
        )


def test_change_particlebeam_dtype():
    """
    Test that a particle beam can be successfully changed to a different dtype.
    """
    beam = cheetah.ParticleBeam.from_parameters(num_particles=100_000)

    # Test that by default the particle beam is of dtype float32
    assert beam.particles.dtype == torch.float32
    assert beam.energy.dtype == torch.float32
    assert beam.total_charge.dtype == torch.float32

    beam.to(torch.float64)

    # Test that the particle beam is now of dtype float64
    assert beam.particles.dtype == torch.float64
    assert beam.energy.dtype == torch.float64
    assert beam.total_charge.dtype == torch.float64


@pytest.mark.parametrize(
    "BeamClass",
    [
        cheetah.ParameterBeam,
        cheetah.ParticleBeam,
    ],
)
def test_transformed_beam_dtype(BeamClass):
    """
    Test that Beam.transformed_to keeps the dtype by default.
    """
    beam = BeamClass.from_parameters(mu_x=torch.tensor(1e-5), dtype=torch.float64)

    # Verify the dtype is kept by default
    transformed_beam = beam.transformed_to(mu_x=torch.tensor(-2e-5))
    assert transformed_beam.mu_x.dtype == torch.float64

    # Check that the manual dtype selection works
    transformed_beam = beam.transformed_to(
        mu_x=torch.tensor(-2e-5), dtype=torch.float32
    )
    assert transformed_beam.mu_x.dtype == torch.float32
