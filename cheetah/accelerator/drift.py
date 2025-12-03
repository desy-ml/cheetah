from typing import Literal

import matplotlib.pyplot as plt
import torch

from cheetah.accelerator.element import Element
from cheetah.particles import Beam, ParticleBeam, Species
from cheetah.track_methods import base_ttensor, drift_matrix
from cheetah.utils import UniqueNameGenerator, bmadx, cache_transfer_map

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")


class Drift(Element):
    """
    Drift section in a particle accelerator.

    :param length: Length in meters.
    :param tracking_method: Method to use for tracking through the element.
    :param name: Unique identifier of the element.
    :param sanitize_name: Whether to sanitise the name to be a valid Python variable
        name. This is needed if you want to use the `segment.element_name` syntax to
        access the element in a segment.
    """

    supported_tracking_methods = ["linear", "second_order", "drift_kick_drift"]

    def __init__(
        self,
        length: torch.Tensor,
        tracking_method: Literal[
            "linear", "second_order", "drift_kick_drift"
        ] = "linear",
        name: str | None = None,
        sanitize_name: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(name=name, sanitize_name=sanitize_name, **factory_kwargs)

        self.length = length

        self.tracking_method = tracking_method

    @cache_transfer_map
    def first_order_transfer_map(
        self, energy: torch.Tensor, species: Species
    ) -> torch.Tensor:
        return drift_matrix(length=self.length, energy=energy, species=species)

    @cache_transfer_map
    def second_order_transfer_map(
        self, energy: torch.Tensor, species: Species
    ) -> torch.Tensor:
        zero = self.length.new_zeros(())

        T = base_ttensor(
            self.length, k1=zero, k2=zero, hx=zero, energy=energy, species=species
        )

        # Fill the first-order transfer map into the second-order transfer map
        T[..., :, 6, :] = drift_matrix(
            length=self.length, energy=energy, species=species
        )

        return T

    def track(self, incoming: Beam) -> Beam:
        """
        Track particles through the dipole element.

        :param incoming: Beam entering the element.
        :return: Beam exiting the element.
        """
        if self.tracking_method == "linear":
            return super()._track_first_order(incoming)
        elif self.tracking_method == "cheetah":
            raise NotImplementedError(
                "The 'cheetah' tracking method has been deprecated and is no longer"
                " supported. Please use 'linear' instead."
            )
        elif self.tracking_method == "second_order":
            return super()._track_second_order(incoming)
        elif self.tracking_method == "drift_kick_drift":
            return self._track_drift_kick_drift(incoming)
        elif self.tracking_method == "bmadx":
            raise NotImplementedError(
                "The 'bmadx' tracking method has been deprecated and is no longer"
                " supported. Please use 'drift_kick_drift' instead."
            )
        else:
            raise ValueError(
                f"Invalid tracking method {self.tracking_method}. For element of"
                f" type {self.__class__.__name__}, supported methods are "
                f"{self.supported_tracking_methods}."
            )

    def _track_drift_kick_drift(self, incoming: ParticleBeam) -> ParticleBeam:
        """
        Track particles through the dipole element using the Drift_kick_drift method.

        :param incoming: Beam entering the element. Currently only supports
            `ParticleBeam`.
        :return: Beam exiting the element.
        """
        assert isinstance(
            incoming, ParticleBeam
        ), "Drift-kick-drift tracking is currently only supported for `ParticleBeam`."

        # Compute Bmad coordinates and p0c
        x = incoming.x
        px = incoming.px
        y = incoming.y
        py = incoming.py
        tau = incoming.tau
        delta = incoming.p

        z, pz, p0c = bmadx.cheetah_to_bmad_z_pz(
            tau, delta, incoming.energy, incoming.species.mass_eV
        )

        # Begin Bmad-X tracking
        x, y, z = bmadx.track_a_drift(
            self.length, x, px, y, py, z, pz, p0c, incoming.species.mass_eV
        )
        # End of Bmad-X tracking

        # Convert back to Cheetah coordinates
        tau, delta, ref_energy = bmadx.bmad_to_cheetah_z_pz(
            z, pz, p0c, incoming.species.mass_eV
        )

        # Broadcast to align their shapes so that they can be stacked
        x, px, y, py, tau, delta = torch.broadcast_tensors(x, px, y, py, tau, delta)

        outgoing_beam = ParticleBeam(
            particles=torch.stack(
                [x, px, y, py, tau, delta, torch.ones_like(x)], dim=-1
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
        return self.tracking_method == "linear"

    def split(self, resolution: torch.Tensor) -> list[Element]:
        num_splits = (self.length.abs().max() / resolution).ceil().int()
        return [
            Drift(
                self.length / num_splits,
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
        # This does nothing on purpose, because drift sections are visualised as gaps.
        return ax

    def to_mesh(
        self, cuteness: float | dict = 1.0, show_download_progress: bool = True
    ) -> "tuple[trimesh.Trimesh | None, np.ndarray]":  # noqa: F821 # type: ignore
        # Override to return None for the mesh, as drift sections do not have a 3D mesh
        # representation on purpose. If this override were not present, Cheetah would
        # throw a warning that no mesh is available for `Drift` elements.

        # Import only here because most people will not need it
        import trimesh

        # Compute transformation matrix needed for next mesh to align to output
        output_transform = trimesh.transformations.translation_matrix(
            [0.0, 0.0, self.length.item()]
        )

        return None, output_transform

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + ["length"]
