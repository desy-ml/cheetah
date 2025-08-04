from typing import Literal

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle

from cheetah.accelerator.element import Element
from cheetah.particles import Beam, ParticleBeam, Species
from cheetah.track_methods import base_rmatrix, misalignment_matrix
from cheetah.utils import (
    UniqueNameGenerator,
    bmadx,
    squash_index_for_unavailable_dims,
    verify_device_and_dtype,
)

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")


class Quadrupole(Element):
    """
    Quadrupole magnet in a particle accelerator.

    :param length: Length in meters.
    :param k1: Strength of the quadrupole in 1/m^-2.
    :param misalignment: Misalignment vector of the quadrupole in x- and y-directions.
    :param tilt: Tilt angle of the quadrupole in x-y plane in radians. pi/4 for
        skew-quadrupole.
    :param num_steps: Number of drift-kick-drift steps to use for tracking through the
        element when tracking method is set to `"bmadx"`.
    :param tracking_method: Method to use for tracking through the element.
    :param name: Unique identifier of the element.
    :param sanitize_name: Whether to sanitise the name to be a valid Python
        variable name. This is needed if you want to use the `segment.element_name`
        syntax to access the element in a segment.
    """

    def __init__(
        self,
        length: torch.Tensor,
        k1: torch.Tensor | None = None,
        misalignment: torch.Tensor | None = None,
        tilt: torch.Tensor | None = None,
        num_steps: int = 1,
        tracking_method: Literal["cheetah", "bmadx"] = "cheetah",
        name: str | None = None,
        sanitize_name: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        device, dtype = verify_device_and_dtype(
            [length, k1, misalignment, tilt], device, dtype
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(name=name, sanitize_name=sanitize_name, **factory_kwargs)

        self.length = torch.as_tensor(length, **factory_kwargs)

        self.register_buffer_or_parameter(
            "k1", torch.as_tensor(k1 if k1 is not None else 0.0, **factory_kwargs)
        )
        self.register_buffer_or_parameter(
            "misalignment",
            torch.as_tensor(
                misalignment if misalignment is not None else (0.0, 0.0),
                **factory_kwargs,
            ),
        )
        self.register_buffer_or_parameter(
            "tilt", torch.as_tensor(tilt if tilt is not None else 0.0, **factory_kwargs)
        )

        self.num_steps = num_steps
        self.tracking_method = tracking_method

    def transfer_map(self, energy: torch.Tensor, species: Species) -> torch.Tensor:
        R = base_rmatrix(
            length=self.length,
            k1=self.k1,
            hx=torch.zeros_like(self.length),
            species=species,
            tilt=self.tilt,
            energy=energy,
        )

        if torch.all(self.misalignment == 0):
            return R
        else:
            R_entry, R_exit = misalignment_matrix(self.misalignment)
            R = torch.einsum("...ij,...jk,...kl->...il", R_exit, R, R_entry)
            return R

    def track(self, incoming: Beam) -> Beam:
        """
        Track particles through the quadrupole element.

        :param incoming: Beam entering the element.
        :return: Beam exiting the element.
        """
        if self.tracking_method == "cheetah":
            return super().track(incoming)
        elif self.tracking_method == "bmadx":
            assert isinstance(
                incoming, ParticleBeam
            ), "Bmad-X tracking is currently only supported for `ParticleBeam`."
            return self._track_bmadx(incoming)
        else:
            raise ValueError(
                f"Invalid tracking method {self.tracking_method}. "
                + "Supported methods are 'cheetah' and 'bmadx'."
            )

    def _track_bmadx(self, incoming: ParticleBeam) -> ParticleBeam:
        """
        Track particles through the quadrupole element using the Bmad-X tracking method.

        :param incoming: Beam entering the element. Currently only supports
            `ParticleBeam`.
        :return: Beam exiting the element.
        """
        # Compute Bmad coordinates and p0c
        x = incoming.x
        px = incoming.px
        y = incoming.y
        py = incoming.py
        tau = incoming.tau
        delta = incoming.p
        mc2 = incoming.species.mass_eV

        z, pz, p0c = bmadx.cheetah_to_bmad_z_pz(tau, delta, incoming.energy, mc2)

        x_offset = self.misalignment[..., 0]
        y_offset = self.misalignment[..., 1]

        step_length = self.length / self.num_steps
        b1 = self.k1 * self.length

        # Begin Bmad-X tracking
        x, px, y, py = bmadx.offset_particle_set(
            x_offset, y_offset, self.tilt, x, px, y, py
        )

        for _ in range(self.num_steps):
            rel_p = 1 + pz  # Particle's relative momentum (P/P0)
            k1 = b1.unsqueeze(-1) / (self.length.unsqueeze(-1) * rel_p)

            tx, dzx = bmadx.calculate_quadrupole_coefficients(-k1, step_length, rel_p)
            ty, dzy = bmadx.calculate_quadrupole_coefficients(k1, step_length, rel_p)

            z = (
                z
                + dzx[0] * x**2
                + dzx[1] * x * px
                + dzx[2] * px**2
                + dzy[0] * y**2
                + dzy[1] * y * py
                + dzy[2] * py**2
            )

            x_next = tx[0][0] * x + tx[0][1] * px
            px_next = tx[1][0] * x + tx[1][1] * px
            y_next = ty[0][0] * y + ty[0][1] * py
            py_next = ty[1][0] * y + ty[1][1] * py

            x, px, y, py = x_next, px_next, y_next, py_next

            z = z + bmadx.low_energy_z_correction(pz, p0c, mc2, step_length)

        # s = s + l
        x, px, y, py = bmadx.offset_particle_unset(
            x_offset, y_offset, self.tilt, x, px, y, py
        )

        # pz is unaffected by tracking, therefore needs to match vector dimensions
        pz = pz * torch.ones_like(x)
        # End of Bmad-X tracking

        # Convert back to Cheetah coordinates
        tau, delta, ref_energy = bmadx.bmad_to_cheetah_z_pz(z, pz, p0c, mc2)

        outgoing_beam = ParticleBeam(
            particles=torch.stack(
                (x, px, y, py, tau, delta, torch.ones_like(x)), dim=-1
            ),
            energy=ref_energy,
            particle_charges=incoming.particle_charges,
            survival_probabilities=incoming.survival_probabilities,
            s=incoming.s + self.length,
            species=incoming.species,
        )
        return outgoing_beam

    @property
    def is_skippable(self) -> bool:
        return self.tracking_method == "cheetah"

    @property
    def is_active(self) -> bool:
        return torch.any(self.k1 != 0).item()

    def split(self, resolution: torch.Tensor) -> list[Element]:
        num_splits = (self.length.abs().max() / resolution).ceil().int()
        return [
            Quadrupole(
                self.length / num_splits,
                self.k1,
                misalignment=self.misalignment,
                tilt=self.tilt,
                num_steps=self.num_steps,
                tracking_method=self.tracking_method,
                dtype=self.length.dtype,
                device=self.length.device,
            )
            for i in range(num_splits)
        ]

    def plot(
        self, s: float, vector_idx: tuple | None = None, ax: plt.Axes | None = None
    ) -> plt.Axes:
        ax = ax or plt.subplot(111)

        plot_k1 = (
            self.k1[squash_index_for_unavailable_dims(vector_idx, self.k1.shape)]
            if len(self.k1.shape) > 0
            else self.k1
        )

        plot_s = (
            s[squash_index_for_unavailable_dims(vector_idx, s.shape)]
            if len(s.shape) > 0
            else s
        )
        plot_length = (
            self.length[squash_index_for_unavailable_dims(vector_idx, s.shape)]
            if len(self.length.shape) > 0
            else self.length
        )

        alpha = 1 if self.is_active else 0.2
        height = 0.8 * (torch.sign(plot_k1) if self.is_active else 1)
        patch = Rectangle(
            (plot_s, 0), plot_length, height, color="tab:red", alpha=alpha, zorder=2
        )
        ax.add_patch(patch)

    def to_mesh(
        self, cuteness: float | dict = 1.0, show_download_progress: bool = True
    ) -> "tuple[trimesh.Trimesh | None, np.ndarray]":  # noqa: F821 # type: ignore
        # Import only here because most people will not need it
        import trimesh

        mesh, output_transform = super().to_mesh(
            cuteness=cuteness, show_download_progress=show_download_progress
        )

        # NOTE: The quadrupole's tilt and misalignment have no effect on where the next
        # element is placed.

        # Rotate according to tilt
        rotation_matrix = trimesh.transformations.rotation_matrix(
            self.tilt.item(), [0, 0, 1], [0, 0, 0]
        )
        mesh.apply_transform(rotation_matrix)

        # Apply misalignment in x and y directions
        mesh.apply_translation(
            [self.misalignment[0].item(), self.misalignment[1].item(), 0.0]
        )

        return mesh, output_transform

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + [
            "length",
            "k1",
            "misalignment",
            "tilt",
            "tracking_method",
        ]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(length={repr(self.length)}, "
            + f"k1={repr(self.k1)}, "
            + f"misalignment={repr(self.misalignment)}, "
            + f"tilt={repr(self.tilt)}, "
            + f"num_steps={repr(self.num_steps)}, "
            + f"tracking_method={repr(self.tracking_method)}, "
            + f"name={repr(self.name)})"
        )
