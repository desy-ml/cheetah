from typing import Literal, Optional, Union

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle
from torch import nn
from torch.distributions import MultivariateNormal

from cheetah.accelerator.element import Element
from cheetah.particles import Beam, ParameterBeam, ParticleBeam
from cheetah.utils import UniqueNameGenerator, kde_histogram_2d

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")


class Screen(Element):
    """
    Diagnostic screen in a particle accelerator.

    :param resolution: Resolution of the camera sensor looking at the screen given as
        Tensor `(width, height)` in pixels.
    :param pixel_size: Size of a pixel on the screen in meters given as a Tensor
        `(width, height)`.
    :param binning: Binning used by the camera.
    :param misalignment: Misalignment of the screen in meters given as a Tensor
        `(x, y)`.
    :param method: Method used to generate the screen's reading. Can be either
        "histogram" or "kde", defaults to "histogram". KDE will be slower but allows
        backward differentiation.
    :param kde_bandwidth: Bandwidth used for the kernel density estimation in meters.
        Controls the smoothness of the distribution.
    :param is_blocking: If `True` the screen is blocking and will stop the beam.
    :param is_active: If `True` the screen is active and will record the beam's
        distribution. If `False` the screen is inactive and will not record the beam's
        distribution.
    :param name: Unique identifier of the element.

    NOTE: `method='histogram'` currently does not support vectorisation. Please use
        `method=`kde` instead. Similarly, `ParameterBeam` can also not be vectorised.
        Please use `ParticleBeam` instead.
    """

    def __init__(
        self,
        resolution: Optional[Union[torch.Tensor, nn.Parameter]] = None,
        pixel_size: Optional[Union[torch.Tensor, nn.Parameter]] = None,
        binning: Optional[Union[torch.Tensor, nn.Parameter]] = None,
        misalignment: Optional[Union[torch.Tensor, nn.Parameter]] = None,
        method: Literal["histogram", "kde"] = "histogram",
        kde_bandwidth: Optional[Union[torch.Tensor, nn.Parameter]] = None,
        is_blocking: bool = False,
        is_active: bool = False,
        name: Optional[str] = None,
        device=None,
        dtype=torch.float32,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(name=name)

        self.register_buffer(
            "resolution",
            (
                torch.as_tensor(resolution, **factory_kwargs)
                if resolution is not None
                else torch.tensor((1024, 1024), **factory_kwargs)
            ),
        )
        self.register_buffer(
            "pixel_size",
            (
                torch.as_tensor(pixel_size, **factory_kwargs)
                if pixel_size is not None
                else torch.tensor((1e-3, 1e-3), **factory_kwargs)
            ),
        )
        self.register_buffer(
            "binning",
            (
                torch.as_tensor(binning, **factory_kwargs)
                if binning is not None
                else torch.tensor(1, **factory_kwargs)
            ),
        )
        self.register_buffer(
            "misalignment",
            (
                torch.as_tensor(misalignment, **factory_kwargs)
                if misalignment is not None
                else torch.tensor((0.0, 0.0), **factory_kwargs)
            ),
        )
        self.register_buffer(
            "length",
            torch.zeros(self.misalignment.shape[:-1], **factory_kwargs),
        )
        assert method in [
            "histogram",
            "kde",
        ], f"Invalid method {method}. Must be either 'histogram' or 'kde'."
        self.method = method
        self.register_buffer(
            "kde_bandwidth",
            (
                torch.as_tensor(kde_bandwidth, **factory_kwargs)
                if kde_bandwidth is not None
                else torch.clone(self.pixel_size[0])
            ),
        )
        self.is_blocking = is_blocking
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

    @property
    def pixel_bin_centers(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            (self.pixel_bin_edges[0][1:] + self.pixel_bin_edges[0][:-1]) / 2,
            (self.pixel_bin_edges[1][1:] + self.pixel_bin_edges[1][:-1]) / 2,
        )

    def transfer_map(self, energy: torch.Tensor) -> torch.Tensor:
        device = self.misalignment.device
        dtype = self.misalignment.dtype

        return torch.eye(7, device=device, dtype=dtype).repeat((*energy.shape, 1, 1))

    def track(self, incoming: Beam) -> Beam:
        if self.is_active:
            copy_of_incoming = incoming.clone()

            if isinstance(incoming, ParameterBeam):
                copy_of_incoming._mu, _ = torch.broadcast_tensors(
                    copy_of_incoming._mu, self.misalignment[..., 0]
                )
                copy_of_incoming._mu = copy_of_incoming._mu.clone()

                copy_of_incoming._mu[..., 0] -= self.misalignment[..., 0]
                copy_of_incoming._mu[..., 2] -= self.misalignment[..., 1]
            elif isinstance(incoming, ParticleBeam):
                copy_of_incoming.particles, _ = torch.broadcast_tensors(
                    copy_of_incoming.particles,
                    self.misalignment[..., 0].unsqueeze(-1).unsqueeze(-1),
                )
                copy_of_incoming.particles = copy_of_incoming.particles.clone()

                copy_of_incoming.particles[..., 0] -= self.misalignment[
                    ..., 0
                ].unsqueeze(-1)
                copy_of_incoming.particles[..., 1] -= self.misalignment[
                    ..., 1
                ].unsqueeze(-1)

            self.set_read_beam(copy_of_incoming)

        return Beam.empty if self.is_blocking else incoming

    @property
    def reading(self) -> torch.Tensor:
        image = None
        if self.cached_reading is not None:
            return self.cached_reading

        read_beam = self.get_read_beam()
        if read_beam is Beam.empty or read_beam is None:
            image = torch.zeros(
                (int(self.effective_resolution[1]), int(self.effective_resolution[0]))
            )
        elif isinstance(read_beam, ParameterBeam):
            if torch.numel(read_beam._mu[..., 0]) > 1:
                raise NotImplementedError(
                    "`Screen` does not support vectorization of `ParameterBeam`. "
                    "Please use `ParticleBeam` instead. If this is a feature you would "
                    "like to see, please open an issue on GitHub."
                )

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
            dist = MultivariateNormal(
                loc=transverse_mu, covariance_matrix=transverse_cov
            )

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
            image = dist.log_prob(pos).exp()
            image = torch.flip(image, dims=[1])
        elif isinstance(read_beam, ParticleBeam):
            if self.method == "histogram":
                # Catch vectorisation, which is currently not supported by "histogram"
                if (
                    len(read_beam.particles.shape) > 2
                    or len(read_beam.particle_charges.shape) > 1
                    or len(read_beam.energy.shape) > 0
                ):
                    raise NotImplementedError(
                        "The `'histogram'` method of `Screen` does not support "
                        "vectorization. Use `'kde'` instead. If this is a feature you "
                        "would like to see, please open an issue on GitHub."
                    )

                image, _ = torch.histogramdd(
                    torch.stack((read_beam.x, read_beam.y)).T, bins=self.pixel_bin_edges
                )
                image = torch.flipud(image.T)
            elif self.method == "kde":
                image = kde_histogram_2d(
                    x1=read_beam.x,
                    x2=read_beam.y,
                    bins1=self.pixel_bin_centers[0],
                    bins2=self.pixel_bin_centers[1],
                    bandwidth=self.kde_bandwidth,
                )
                # Change the x, y positions
                image = torch.transpose(image, -2, -1)
                # Flip up and down, now row 0 corresponds to the top
                image = torch.flip(image, dims=[-2])
        else:
            raise TypeError(f"Read beam is of invalid type {type(read_beam)}")

        self.cached_reading = image
        return image

    def get_read_beam(self) -> Beam:
        # Using these get and set methods instead of Python's property decorator to
        # prevent `nn.Module` from intercepting the read beam, which is itself an
        # `nn.Module`, and registering it as a submodule of the screen.
        return self._read_beam

    def set_read_beam(self, value: Beam) -> None:
        # Using these get and set methods instead of Python's property decorator to
        # prevent `nn.Module` from intercepting the read beam, which is itself an
        # `nn.Module`, and registering it as a submodule of the screen.
        self._read_beam = value
        self.cached_reading = None

    def split(self, resolution: torch.Tensor) -> list[Element]:
        return [self]

    def plot(self, ax: plt.Axes, s: float, vector_idx: Optional[tuple] = None) -> None:
        plot_s = s[vector_idx] if s.dim() > 0 else s

        alpha = 1 if self.is_active else 0.2

        patch = Rectangle(
            (plot_s, -0.6), 0, 0.6 * 2, color="tab:green", alpha=alpha, zorder=2
        )
        ax.add_patch(patch)

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + [
            "resolution",
            "pixel_size",
            "binning",
            "misalignment",
            "method",
            "kde_bandwidth",
            "is_active",
        ]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(resolution={repr(self.resolution)}, "
            + f"pixel_size={repr(self.pixel_size)}, "
            + f"binning={repr(self.binning)}, "
            + f"misalignment={repr(self.misalignment)}, "
            + f"method={repr(self.method)}, "
            + f"kde_bandwidth={repr(self.kde_bandwidth)}, "
            + f"is_active={repr(self.is_active)}, "
            + f"name={repr(self.name)})"
        )
