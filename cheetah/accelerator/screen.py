from copy import deepcopy
from typing import Optional, Union

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle
from scipy import constants
from scipy.constants import physical_constants
from torch import Size, nn
from torch.distributions import MultivariateNormal

from cheetah.particles import Beam, ParameterBeam, ParticleBeam
from cheetah.utils import UniqueNameGenerator

from .element import Element

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")

rest_energy = torch.tensor(
    constants.electron_mass
    * constants.speed_of_light**2
    / constants.elementary_charge  # electron mass
)
electron_mass_eV = torch.tensor(
    physical_constants["electron mass energy equivalent in MeV"][0] * 1e6
)


class Screen(Element):
    """
    Diagnostic screen in a particle accelerator.

    :param resolution: Resolution of the camera sensor looking at the screen given as
        Tensor `(width, height)`.
    :param pixel_size: Size of a pixel on the screen in meters given as a Tensor
        `(width, height)`.
    :param binning: Binning used by the camera.
    :param misalignment: Misalignment of the screen in meters given as a Tensor
        `(x, y)`.
    :param is_active: If `True` the screen is active and will record the beam's
        distribution. If `False` the screen is inactive and will not record the beam's
        distribution.
    :param name: Unique identifier of the element.
    """

    def __init__(
        self,
        resolution: Optional[Union[torch.Tensor, nn.Parameter]] = None,
        pixel_size: Optional[Union[torch.Tensor, nn.Parameter]] = None,
        binning: Optional[Union[torch.Tensor, nn.Parameter]] = None,
        misalignment: Optional[Union[torch.Tensor, nn.Parameter]] = None,
        is_active: bool = False,
        name: Optional[str] = None,
        device=None,
        dtype=torch.float32,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(name=name)

        self.resolution = (
            torch.as_tensor(resolution, device=device)
            if resolution is not None
            else torch.tensor((1024, 1024), device=device)
        )
        self.pixel_size = (
            torch.as_tensor(pixel_size, **factory_kwargs)
            if pixel_size is not None
            else torch.tensor((1e-3, 1e-3), **factory_kwargs)
        )
        self.binning = (
            torch.as_tensor(binning, device=device)
            if binning is not None
            else torch.tensor(1, device=device)
        )
        self.misalignment = (
            torch.as_tensor(misalignment, **factory_kwargs)
            if misalignment is not None
            else torch.tensor([(0.0, 0.0)], **factory_kwargs)
        )
        self.length = torch.zeros(self.misalignment.shape[:-1], **factory_kwargs)
        self.is_active = is_active

        self.set_read_beam(None)
        self.cached_reading = None

    @property
    def is_skippable(self) -> bool:
        return not self.is_active

    @property
    def effective_resolution(self) -> torch.Tensor:
        return self.resolution / self.binning

    @property
    def effective_pixel_size(self) -> torch.Tensor:
        return self.pixel_size * self.binning

    @property
    def extent(self) -> torch.Tensor:
        return torch.stack(
            [
                -self.resolution[0] * self.pixel_size[0] / 2,
                self.resolution[0] * self.pixel_size[0] / 2,
                -self.resolution[1] * self.pixel_size[1] / 2,
                self.resolution[1] * self.pixel_size[1] / 2,
            ]
        )

    @property
    def pixel_bin_edges(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.linspace(
                -self.resolution[0] * self.pixel_size[0] / 2,
                self.resolution[0] * self.pixel_size[0] / 2,
                int(self.effective_resolution[0]) + 1,
            ),
            torch.linspace(
                -self.resolution[1] * self.pixel_size[1] / 2,
                self.resolution[1] * self.pixel_size[1] / 2,
                int(self.effective_resolution[1]) + 1,
            ),
        )

    def transfer_map(self, energy: torch.Tensor) -> torch.Tensor:
        device = self.misalignment.device
        dtype = self.misalignment.dtype

        return torch.eye(7, device=device, dtype=dtype).repeat((*energy.shape, 1, 1))

    def track(self, incoming: Beam) -> Beam:
        if self.is_active:
            copy_of_incoming = deepcopy(incoming)

            if isinstance(incoming, ParameterBeam):
                copy_of_incoming._mu[:, 0] -= self.misalignment[:, 0]
                copy_of_incoming._mu[:, 2] -= self.misalignment[:, 1]
            elif isinstance(incoming, ParticleBeam):
                copy_of_incoming.particles[:, :, 0] -= self.misalignment[:, 0]
                copy_of_incoming.particles[:, :, 1] -= self.misalignment[:, 1]

            self.set_read_beam(copy_of_incoming)

            return Beam.empty
        else:
            return incoming

    @property
    def reading(self) -> torch.Tensor:
        if self.cached_reading is not None:
            return self.cached_reading

        read_beam = self.get_read_beam()
        if read_beam is Beam.empty or read_beam is None:
            image = torch.zeros(
                (
                    *self.misalignment.shape[:-1],
                    int(self.effective_resolution[1]),
                    int(self.effective_resolution[0]),
                )
            )
        elif isinstance(read_beam, ParameterBeam):
            transverse_mu = torch.stack(
                [read_beam._mu[..., 0], read_beam._mu[..., 2]], dim=-1
            )
            transverse_cov = torch.stack(
                [
                    torch.stack(
                        [read_beam._cov[..., 0, 0], read_beam._cov[..., 0, 2]], dim=-1
                    ),
                    torch.stack(
                        [read_beam._cov[..., 2, 0], read_beam._cov[..., 2, 2]], dim=-1
                    ),
                ],
                dim=-1,
            )
            dist = [
                MultivariateNormal(
                    loc=transverse_mu_sample, covariance_matrix=transverse_cov_sample
                )
                for transverse_mu_sample, transverse_cov_sample in zip(
                    transverse_mu.cpu(), transverse_cov.cpu()
                )
            ]

            left = self.extent[0]
            right = self.extent[1]
            hstep = self.pixel_size[0] * self.binning
            bottom = self.extent[2]
            top = self.extent[3]
            vstep = self.pixel_size[1] * self.binning
            x, y = torch.meshgrid(
                torch.arange(left, right, hstep),
                torch.arange(bottom, top, vstep),
                indexing="ij",
            )
            pos = torch.dstack((x, y))
            image = torch.stack(
                [dist_sample.log_prob(pos).exp() for dist_sample in dist]
            )
            image = torch.flip(image, dims=[1])
        elif isinstance(read_beam, ParticleBeam):
            image = torch.zeros(
                (
                    *self.misalignment.shape[:-1],
                    int(self.effective_resolution[1]),
                    int(self.effective_resolution[0]),
                )
            )
            for i, (xs_sample, ys_sample) in enumerate(zip(read_beam.xs, read_beam.ys)):
                image_sample, _ = torch.histogramdd(
                    torch.stack((xs_sample, ys_sample)).T.cpu(),
                    bins=self.pixel_bin_edges,
                )
                image_sample = torch.flipud(image_sample.T)
                image_sample = image_sample.cpu()

                image[i] = image_sample
        else:
            raise TypeError(f"Read beam is of invalid type {type(read_beam)}")

        self.cached_reading = image
        return image

    def get_read_beam(self) -> Beam:
        # Using these get and set methods instead of Python's property decorator to
        # prevent `nn.Module` from intercepting the read beam, which is itself an
        # `nn.Module`, and registering it as a submodule of the screen.
        return self._read_beam[0] if self._read_beam is not None else None

    def set_read_beam(self, value: Beam) -> None:
        # Using these get and set methods instead of Python's property decorator to
        # prevent `nn.Module` from intercepting the read beam, which is itself an
        # `nn.Module`, and registering it as a submodule of the screen.
        self._read_beam = [value]
        self.cached_reading = None

    def broadcast(self, shape: Size) -> Element:
        new_screen = self.__class__(
            resolution=self.resolution,
            pixel_size=self.pixel_size,
            binning=self.binning,
            misalignment=self.misalignment.repeat((*shape, 1)),
            is_active=self.is_active,
            name=self.name,
        )
        new_screen.length = self.length.repeat(shape)
        return new_screen

    def split(self, resolution: torch.Tensor) -> list[Element]:
        return [self]

    def plot(self, ax: plt.Axes, s: float) -> None:
        alpha = 1 if self.is_active else 0.2
        patch = Rectangle(
            (s, -0.6), 0, 0.6 * 2, color="tab:green", alpha=alpha, zorder=2
        )
        ax.add_patch(patch)

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + [
            "resolution",
            "pixel_size",
            "binning",
            "misalignment",
            "is_active",
        ]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(resolution={repr(self.resolution)}, "
            + f"pixel_size={repr(self.pixel_size)}, "
            + f"binning={repr(self.binning)}, "
            + f"misalignment={repr(self.misalignment)}, "
            + f"is_active={repr(self.is_active)}, "
            + f"name={repr(self.name)})"
        )
