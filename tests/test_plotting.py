import pytest
import torch

import cheetah

from .resources import ARESlatticeStage3v1_9 as ares


@pytest.mark.filterwarnings("ignore::cheetah.utils.DefaultParameterWarning")
def test_twiss_plot():
    """
    Test that the Twiss plot does not raise an exception using the ARES EA as an
    example.
    """
    cell = cheetah.converters.ocelot.subcell_of_ocelot(
        ares.cell, "AREASOLA1", "AREABSCR1"
    )
    ares.areamqzm1.k1 = 5.0
    ares.areamqzm2.k1 = -5.0
    ares.areamcvm1.k1 = 1e-3
    ares.areamqzm3.k1 = 5.0
    ares.areamchm1.k1 = -2e-3

    incoming_beam = cheetah.ParticleBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )
    segment = cheetah.Segment.from_ocelot(cell)

    # Run the plotting to see if it raises an exception
    segment.plot_twiss(incoming_beam)


def test_mean_and_std_particle_plot():
    """
    Test that the mean and standard deviation particle plot does not raise an exception
    using the example from the `simple.ipynb` example notebook from the documentation.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.BPM(name="BPM1SMATCH"),
            cheetah.Drift(length=torch.tensor(1.0)),
            cheetah.BPM(name="BPM6SMATCH"),
            cheetah.Drift(length=torch.tensor(1.0)),
            cheetah.VerticalCorrector(length=torch.tensor(0.3), name="V7SMATCH"),
            cheetah.Drift(length=torch.tensor(0.2)),
            cheetah.HorizontalCorrector(length=torch.tensor(0.3), name="H10SMATCH"),
            cheetah.Drift(length=torch.tensor(7.0)),
            cheetah.HorizontalCorrector(length=torch.tensor(0.3), name="H12SMATCH"),
            cheetah.Drift(length=torch.tensor(0.05)),
            cheetah.BPM(name="BPM13SMATCH"),
        ]
    )

    segment.V7SMATCH.angle = torch.tensor(3.142e-3)

    incoming = cheetah.ParticleBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )

    # Run the plotting to see if it raises an exception
    segment.plot_overview(incoming=incoming)


@pytest.mark.filterwarnings("ignore::cheetah.utils.DefaultParameterWarning")
def test_twiss_plot_vectorized_2d():
    """
    Test that the Twiss plot does not raise an exception using the ARES EA as an example
    and when the model has two vector dimensions.
    """
    segment = cheetah.Segment.from_ocelot(ares.cell).subcell("AREASOLA1", "AREABSCR1")
    segment.AREAMQZM1.k1 = torch.tensor(5.0)
    segment.AREAMQZM2.k1 = torch.tensor([[-5.0, -2.0, -1.0], [1.0, 2.0, 5.0]])
    segment.AREAMCVM1.k1 = torch.tensor(1e-3)
    segment.AREAMQZM3.k1 = torch.tensor(5.0)
    segment.AREAMCHM1.k1 = torch.tensor(-2e-3)
    segment.Drift_AREAMCHM1.length = (
        torch.FloatTensor(2, 3).uniform_(0.9, 1.1) * segment.Drift_AREAMCHM1.length
    )

    incoming = cheetah.ParticleBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )

    # Run the plotting to see if it raises an exception
    segment.plot_twiss(incoming=incoming, vector_idx=(0, 2))


@pytest.mark.filterwarnings("ignore::cheetah.utils.DefaultParameterWarning")
def test_reference_particle_plot_vectorized_2d():
    """
    Test that the Twiss plot does not raise an exception using the ARES EA as an example
    and when the model has two vector dimensions.
    """
    segment = cheetah.Segment.from_ocelot(ares.cell).subcell("AREASOLA1", "AREABSCR1")
    segment.AREAMQZM1.k1 = torch.tensor(5.0)
    segment.AREAMQZM2.k1 = torch.tensor([1.0, 2.0, 5.0])
    segment.AREAMCVM1.k1 = torch.tensor(1e-3)
    segment.AREAMQZM3.k1 = torch.tensor(5.0)
    segment.AREAMCHM1.k1 = torch.tensor(-2e-3)
    segment.Drift_AREAMCHM1.length = (
        torch.FloatTensor(2, 1).uniform_(0.9, 1.1) * segment.Drift_AREAMCHM1.length
    )

    incoming = cheetah.ParticleBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )

    # Run the plotting to see if it raises an exception
    segment.plot_overview(incoming=incoming, resolution=0.1, vector_idx=(0, 2))


def test_plotting_with_nonleaf_tensors():
    """Test that the plotting routines can handle elements with non-leaf tensors."""
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(1.0, requires_grad=True)),
            cheetah.BPM(is_active=True),
        ]
    )

    incoming = cheetah.ParticleBeam.from_astra(
        "tests/resources/ACHIP_EA1_2021.1351.001"
    )

    # Prepopulate the segment
    segment.track(incoming)

    # Test that plotting does not raise an exception
    segment.plot_overview(incoming=incoming)
    segment.plot_twiss(incoming=incoming)


def test_plotting_with_gradients():
    """
    Test that plotting doesn't raise an exception for segments that contain tensors
    that require gradients.
    """
    segment = cheetah.Segment(
        elements=[cheetah.Drift(length=torch.tensor(1.0, requires_grad=True))]
    )
    beam = cheetah.ParameterBeam.from_parameters()

    segment.plot_overview(incoming=beam)
    segment.plot_twiss(incoming=beam)


@pytest.mark.parametrize("style", ["histogram", "contour"])
def test_plot_6d_particle_beam_distribution(style):
    """Test that the 6D `ParticleBeam` distribution plot does not raise an exception."""
    beam = cheetah.ParticleBeam.from_astra("tests/resources/ACHIP_EA1_2021.1351.001")

    # Run the plotting to see if it raises an exception
    _ = beam.plot_distribution(bin_ranges="unit_same", plot_2d_kws={"style": style})


def test_plot_particle_beam_point_cloud():
    """Test that the `ParticleBeam`'s point cloud plot does not raise an exception."""
    beam = cheetah.ParticleBeam.from_astra("tests/resources/ACHIP_EA1_2021.1351.001")

    # Run the plotting to see if it raises an exception
    _ = beam.plot_point_cloud()


def test_plot_floor_space_basic():
    """
    Test that `Segment.plot_floor_space` renders basic elements without raising an
    exception.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(1.0)),
            cheetah.Quadrupole(length=torch.tensor(0.3), name="quad1"),
            cheetah.Drift(length=torch.tensor(0.5)),
            cheetah.Cavity(length=torch.tensor(0.8), name="cav1"),
            cheetah.BPM(name="bpm1"),
            cheetah.Screen(name="screen1"),
            cheetah.Aperture(name="aper1"),
            cheetah.Marker(name="mark1"),
            cheetah.HorizontalCorrector(length=torch.tensor(0.2), name="hcor1"),
            cheetah.VerticalCorrector(length=torch.tensor(0.2), name="vcor1"),
            cheetah.Solenoid(length=torch.tensor(0.4), name="sol1"),
            cheetah.Undulator(length=torch.tensor(1.0), name="und1"),
        ]
    )

    ax = segment.plot_floor_space()
    assert ax is not None


def test_plot_floor_space_dipoles():
    """
    Test that `Segment.plot_floor_space` handles bending dipoles correctly without
    raising an exception.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(1.0)),
            cheetah.Dipole(
                length=torch.tensor(1.0),
                angle=torch.tensor(0.5),
                name="bend1",
            ),
            cheetah.Drift(length=torch.tensor(0.5)),
            cheetah.RBend(
                length=torch.tensor(1.0),
                angle=torch.tensor(-0.5),
                name="rbend1",
            ),
            cheetah.Dipole(
                length=torch.tensor(0.5),
                angle=torch.tensor(0.0),
                name="zero_bend",
            ),
            cheetah.Drift(length=torch.tensor(1.0)),
        ]
    )

    ax = segment.plot_floor_space()
    assert ax is not None


def test_plot_floor_space_vectorized():
    """
    Test that `Segment.plot_floor_space` handles vectorized parameters with
    `vector_idx`.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor([1.0, 2.0])),
            cheetah.Quadrupole(
                length=torch.tensor([0.3, 0.4]),
                k1=torch.tensor([5.0, -5.0]),
            ),
            cheetah.Dipole(
                length=torch.tensor([1.0, 1.0]),
                angle=torch.tensor([0.2, -0.2]),
            ),
        ]
    )

    ax = segment.plot_floor_space(vector_idx=(1,))
    assert ax is not None


def test_plot_floor_space_nested():
    """Test that `Segment.plot_floor_space` handles nested subsegments."""
    subsegment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(0.5)),
            cheetah.Quadrupole(length=torch.tensor(0.2), name="nested_quad"),
            cheetah.Drift(length=torch.tensor(0.5)),
        ],
        name="sub",
    )
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(1.0)),
            subsegment,
            cheetah.Drift(length=torch.tensor(1.0)),
        ]
    )

    ax = segment.plot_floor_space()
    assert ax is not None


def test_plot_floor_space_empty():
    """Test that `Segment.plot_floor_space` handles an empty segment gracefully."""
    segment = cheetah.Segment(elements=[])
    ax = segment.plot_floor_space()
    assert ax is not None


@pytest.mark.filterwarnings("ignore::cheetah.utils.DefaultParameterWarning")
def test_plot_floor_space_ares():
    """Test that `Segment.plot_floor_space` handles the full ARES lattice."""
    segment = cheetah.Segment.from_ocelot(ares.cell)
    ax = segment.plot_floor_space()
    assert ax is not None


def test_plot_floor_space_options_and_alias():
    """
    Test `Segment.plot_floor_space` with custom options including labels, legend,
    custom widths, custom colors, and the `plot_floor_plan` alias.
    """
    segment = cheetah.Segment(
        elements=[
            cheetah.Drift(length=torch.tensor(1.0)),
            cheetah.Quadrupole(length=torch.tensor(0.4), name="Q1"),
            cheetah.Dipole(
                length=torch.tensor(0.8),
                angle=torch.tensor(0.2),
                name="D1",
            ),
            cheetah.BPM(name="B1"),
        ]
    )

    # Test options
    ax = segment.plot_floor_space(
        show_labels=True,
        show_legend=True,
        element_width={"Q1": 0.5, "Dipole": 0.6},
        element_color={"Q1": "purple", "D1": "navy"},
        start_position=(1.0, 2.0),
        start_angle=0.1,
    )
    assert ax is not None

    # Test plot_floor_plan alias
    ax_alias = segment.plot_floor_plan()
    assert ax_alias is not None


def test_plot_floor_space_invalid_style():
    """
    Test that `Segment.plot_floor_space` raises appropriate errors for unimplemented
    or invalid styles.
    """
    segment = cheetah.Segment(elements=[cheetah.Drift(length=torch.tensor(1.0))])

    with pytest.raises(NotImplementedError):
        segment.plot_floor_space(style="sprites")

    with pytest.raises(ValueError):
        segment.plot_floor_space(style="invalid")
