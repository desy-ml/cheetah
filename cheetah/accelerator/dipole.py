from typing import Literal

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle

from cheetah.accelerator.element import Element
from cheetah.particles import Beam, ParticleBeam, Species
from cheetah.track_methods import base_rmatrix, rotation_matrix
from cheetah.utils import UniqueNameGenerator, bmadx, verify_device_and_dtype

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")


class Dipole(Element):
    """
    Dipole magnet (by default a sector bending magnet).

    :param length: Length in meters.
    :param angle: Deflection angle in rad.
    :param k1: Focussing strength in 1/m^-2. Only used with `"cheetah"` tracking method.
    :param dipole_e1: The angle of inclination of the entrance face in rad.
    :param dipole_e2: The angle of inclination of the exit face in rad.
    :param tilt: Tilt of the magnet in x-y plane [rad].
    :param gap: The magnet gap in meters. Note that in MAD and ELEGANT: HGAP = gap/2.
    :param gap_exit: The magnet gap at the exit in meters. Note that in MAD and
        ELEGANT: HGAP = gap/2. Only set if different from `gap`. Only used with
        `"bmadx"` tracking method.
    :param fringe_integral: Fringe field integral (of the enterance face).
    :param fringe_integral_exit: Fringe field integral of the exit face. Only set if
        different from `fringe_integral`. Only used with `"bmadx"` tracking method.
    :param fringe_at: Where to apply the fringe fields for `"bmadx"` tracking. The
        available options are:
        - "neither": Do not apply fringe fields.
        - "entrance": Apply fringe fields at the entrance end.
        - "exit": Apply fringe fields at the exit end.
        - "both": Apply fringe fields at both ends.
    :param fringe_type: Type of fringe field for `"bmadx"` tracking. Currently only
        supports `"linear_edge"`.
    :param name: Unique identifier of the element.
    :param sanitize_name: Whether to sanitise the name to be a valid Python
        variable name. This is needed if you want to use the `segment.element_name`
        syntax to access the element in a segment.
    """

    def __init__(
        self,
        length: torch.Tensor,
        angle: torch.Tensor | None = None,
        k1: torch.Tensor | None = None,
        dipole_e1: torch.Tensor | None = None,
        dipole_e2: torch.Tensor | None = None,
        tilt: torch.Tensor | None = None,
        gap: torch.Tensor | None = None,
        gap_exit: torch.Tensor | None = None,
        fringe_integral: torch.Tensor | None = None,
        fringe_integral_exit: torch.Tensor | None = None,
        fringe_at: Literal["neither", "entrance", "exit", "both"] = "both",
        fringe_type: Literal["linear_edge"] = "linear_edge",
        tracking_method: Literal["cheetah", "bmadx"] = "cheetah",
        name: str | None = None,
        sanitize_name: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        device, dtype = verify_device_and_dtype(
            [
                length,
                angle,
                k1,
                dipole_e1,
                dipole_e2,
                tilt,
                gap,
                gap_exit,
                fringe_integral,
                fringe_integral_exit,
            ],
            device,
            dtype,
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(name=name, sanitize_name=sanitize_name, **factory_kwargs)

        self.length = torch.as_tensor(length, **factory_kwargs)

        self.register_buffer_or_parameter(
            "angle",
            torch.as_tensor(angle if angle is not None else 0.0, **factory_kwargs),
        )
        self.register_buffer_or_parameter(
            "k1", torch.as_tensor(k1 if k1 is not None else 0.0, **factory_kwargs)
        )
        self.register_buffer_or_parameter(
            "_e1",
            torch.as_tensor(
                dipole_e1 if dipole_e1 is not None else 0.0, **factory_kwargs
            ),
        )
        self.register_buffer_or_parameter(
            "_e2",
            torch.as_tensor(
                dipole_e2 if dipole_e2 is not None else 0.0, **factory_kwargs
            ),
        )
        self.register_buffer_or_parameter(
            "fringe_integral",
            torch.as_tensor(
                fringe_integral if fringe_integral is not None else 0.0,
                **factory_kwargs,
            ),
        )
        self.register_buffer_or_parameter(
            "fringe_integral_exit",
            torch.as_tensor(
                (
                    fringe_integral_exit
                    if fringe_integral_exit is not None
                    else self.fringe_integral
                ),
                **factory_kwargs,
            ),
        )

        self.register_buffer_or_parameter(
            "gap", torch.as_tensor(gap if gap is not None else 0.0, **factory_kwargs)
        )
        self.register_buffer_or_parameter(
            "gap_exit",
            torch.as_tensor(
                gap_exit if gap_exit is not None else self.gap, **factory_kwargs
            ),
        )
        self.register_buffer_or_parameter(
            "tilt", torch.as_tensor(tilt if tilt is not None else 0.0, **factory_kwargs)
        )

        self.fringe_at = fringe_at
        self.fringe_type = fringe_type
        self.tracking_method = tracking_method

    @property
    def hx(self) -> torch.Tensor:
        return torch.where(self.length == 0.0, 0.0, self.angle / self.length)

    @property
    def dipole_e1(self) -> torch.Tensor:
        return self._e1

    @dipole_e1.setter
    def dipole_e1(self, value: torch.Tensor):
        self._e1 = value

    @property
    def dipole_e2(self) -> torch.Tensor:
        return self._e2

    @dipole_e2.setter
    def dipole_e2(self, value: torch.Tensor):
        self._e2 = value

    @property
    def is_skippable(self) -> bool:
        return self.tracking_method == "cheetah"

    @property
    def is_active(self) -> bool:
        return torch.any(self.angle != 0).item()

    def track(self, incoming: Beam) -> Beam:
        """
        Track particles through the dipole element.

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
        # TODO: The renaming of the compinents of `incoming` to just the component name
        # makes things hard to read. The resuse and overwriting of those component names
        # throughout the function makes it even hard, is bad practice and should really
        # be fixed!

        # Compute Bmad coordinates and p0c
        x = incoming.x
        px = incoming.px
        y = incoming.y
        py = incoming.py
        tau = incoming.tau
        delta = incoming.p
        mc2 = incoming.species.mass_eV

        z, pz, p0c = bmadx.cheetah_to_bmad_z_pz(tau, delta, incoming.energy, mc2)

        # Begin Bmad-X tracking
        x, px, y, py = bmadx.offset_particle_set(
            torch.zeros_like(self.tilt),
            torch.zeros_like(self.tilt),
            self.tilt,
            x,
            px,
            y,
            py,
        )

        if self.fringe_at == "entrance" or self.fringe_at == "both":
            px, py = self._bmadx_fringe_linear("entrance", x, px, y, py)
        x, px, y, py, z, pz = self._bmadx_body(x, px, y, py, z, pz, p0c, mc2)
        if self.fringe_at == "exit" or self.fringe_at == "both":
            px, py = self._bmadx_fringe_linear("exit", x, px, y, py)

        x, px, y, py = bmadx.offset_particle_unset(
            torch.zeros_like(self.tilt),
            torch.zeros_like(self.tilt),
            self.tilt,
            x,
            px,
            y,
            py,
        )
        # End of Bmad-X tracking

        # Convert back to Cheetah coordinates
        tau, delta, ref_energy = bmadx.bmad_to_cheetah_z_pz(z, pz, p0c, mc2)

        # Broadcast to align their shapes so that they can be stacked
        x, px, y, py, tau, delta = torch.broadcast_tensors(x, px, y, py, tau, delta)

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

    def _bmadx_body(
        self,
        x: torch.Tensor,
        px: torch.Tensor,
        y: torch.Tensor,
        py: torch.Tensor,
        z: torch.Tensor,
        pz: torch.Tensor,
        p0c: torch.Tensor,
        mc2: float,
    ) -> list[torch.Tensor]:
        """
        Track particle coordinates through bend body.

        :param x: Initial x coordinate [m].
        :param px: Initial Bmad cannonical px coordinate.
        :param y: Initial y coordinate [m].
        :param py: Initial Bmad cannonical py coordinate.
        :param z: Initial Bmad cannonical z coordinate [m].
        :param pz: Initial Bmad cannonical pz coordinate.
        :param p0c: Reference momentum [eV/c].
        :param mc2: Particle mass [eV/c^2].
        :return: x, px, y, py, z, pz final Bmad cannonical coordinates.
        """
        px_norm = torch.sqrt((1 + pz) ** 2 - py**2)  # For simplicity
        phi1 = torch.arcsin(px / px_norm)
        g = self.angle / self.length
        gp = g.unsqueeze(-1) / px_norm
        gp_safe = torch.where(
            gp != 0, gp, torch.tensor(1e-12, dtype=gp.dtype, device=gp.device)
        )

        alpha = (
            2
            * (1 + g.unsqueeze(-1) * x)
            * torch.sin(self.angle.unsqueeze(-1) + phi1)
            * self.length.unsqueeze(-1)
            * bmadx.sinc(self.angle).unsqueeze(-1)
            - gp
            * (
                (1 + g.unsqueeze(-1) * x)
                * self.length.unsqueeze(-1)
                * bmadx.sinc(self.angle).unsqueeze(-1)
            )
            ** 2
        )

        x2_t1 = x * torch.cos(self.angle.unsqueeze(-1)) + self.length.unsqueeze(
            -1
        ) ** 2 * g.unsqueeze(-1) * bmadx.cosc(self.angle.unsqueeze(-1))

        x2_t2 = torch.sqrt(
            (torch.cos(self.angle.unsqueeze(-1) + phi1) ** 2) + gp * alpha
        )
        x2_t3 = torch.cos(self.angle.unsqueeze(-1) + phi1)

        c1 = x2_t1 + alpha / (x2_t2 + x2_t3)
        c2 = x2_t1 + (x2_t2 - x2_t3) / gp_safe
        temp = torch.abs(self.angle.unsqueeze(-1) + phi1)
        x2 = c1 * (temp < torch.pi / 2) + c2 * (temp >= torch.pi / 2)

        Lcu = (
            x2
            - self.length.unsqueeze(-1) ** 2
            * g.unsqueeze(-1)
            * bmadx.cosc(self.angle.unsqueeze(-1))
            - x * torch.cos(self.angle.unsqueeze(-1))
        )

        Lcv = -self.length.unsqueeze(-1) * bmadx.sinc(
            self.angle.unsqueeze(-1)
        ) - x * torch.sin(self.angle.unsqueeze(-1))

        theta_p = 2 * (
            self.angle.unsqueeze(-1) + phi1 - torch.pi / 2 - torch.arctan2(Lcv, Lcu)
        )

        Lc = torch.sqrt(Lcu**2 + Lcv**2)
        Lp = Lc / bmadx.sinc(theta_p / 2)

        P = p0c.unsqueeze(-1) * (1 + pz)  # In eV
        E = torch.sqrt(P**2 + mc2**2)  # In eV
        E0 = torch.sqrt(p0c**2 + mc2**2)  # In eV
        beta = P / E
        beta0 = p0c / E0

        x_f = x2
        px_f = px_norm * torch.sin(self.angle.unsqueeze(-1) + phi1 - theta_p)
        y_f = y + py * Lp / px_norm
        z_f = (
            z
            + (beta * self.length.unsqueeze(-1) / beta0.unsqueeze(-1))
            - ((1 + pz) * Lp / px_norm)
        )

        return x_f, px_f, y_f, py, z_f, pz

    def _bmadx_fringe_linear(
        self,
        location: Literal["entrance", "exit"],
        x: torch.Tensor,
        px: torch.Tensor,
        y: torch.Tensor,
        py: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Tracks linear fringe.

        :param location: "entrance" or "exit".
        :param x: Initial x coordinate [m].
        :param px: Initial Bmad cannonical px coordinate.
        :param y: Initial y coordinate [m].
        :param py: Initial Bmad cannonical py coordinate.
        :return: px, py final Bmad cannonical coordinates.
        """
        g = self.angle / self.length
        e = self._e1 * (location == "entrance") + self._e2 * (location == "exit")
        f_int = self.fringe_integral * (
            location == "entrance"
        ) + self.fringe_integral_exit * (location == "exit")
        h_gap = 0.5 * (
            self.gap * (location == "entrance") + self.gap_exit * (location == "exit")
        )

        hx = g * torch.tan(e)
        hy = -g * torch.tan(
            e - 2 * f_int * h_gap * g * (1 + torch.sin(e) ** 2) / torch.cos(e)
        )
        px_f = px + x * hx.unsqueeze(-1)
        py_f = py + y * hy.unsqueeze(-1)

        return px_f, py_f

    def transfer_map(self, energy: torch.Tensor, species: Species) -> torch.Tensor:
        device = self.length.device
        dtype = self.length.dtype

        R_enter = self._transfer_map_enter()
        R_exit = self._transfer_map_exit()

        if torch.any(self.length != 0.0):  # Bending magnet with finite length
            R = base_rmatrix(
                length=self.length,
                k1=self.k1,
                hx=self.hx,
                species=species,
                tilt=torch.zeros_like(self.length),
                energy=energy,
            )  # Tilt is applied after adding edges
        else:  # Reduce to Thin-Corrector
            R = torch.eye(7, device=device, dtype=dtype).repeat(
                (*self.length.shape, 1, 1)
            )
            R[..., 0, 1] = self.length
            R[..., 2, 6] = self.angle
            R[..., 2, 3] = self.length

        # Apply fringe fields
        R = R_exit @ R @ R_enter

        # Apply rotation for tilted magnets
        if torch.any(self.tilt != 0):
            rotation = rotation_matrix(self.tilt)
            R = rotation.transpose(-1, -2) @ R @ rotation

        return R

    def _transfer_map_enter(self) -> torch.Tensor:
        """Linear transfer map for the entrance face of the dipole magnet."""
        device = self.length.device
        dtype = self.length.dtype

        sec_e = 1.0 / torch.cos(self._e1)
        phi = (
            self.fringe_integral
            * self.hx
            * self.gap
            * sec_e
            * (1 + torch.sin(self._e1) ** 2)
        )

        tm = torch.eye(7, device=device, dtype=dtype).repeat(*phi.shape, 1, 1)
        tm[..., 1, 0] = self.hx * torch.tan(self._e1)
        tm[..., 3, 2] = -self.hx * torch.tan(self._e1 - phi)

        return tm

    def _transfer_map_exit(self) -> torch.Tensor:
        """Linear transfer map for the exit face of the dipole magnet."""
        device = self.length.device
        dtype = self.length.dtype

        sec_e = 1.0 / torch.cos(self._e2)
        phi = (
            self.fringe_integral_exit
            * self.hx
            * self.gap
            * sec_e
            * (1 + torch.sin(self._e2) ** 2)
        )

        tm = torch.eye(7, device=device, dtype=dtype).repeat(*phi.shape, 1, 1)
        tm[..., 1, 0] = self.hx * torch.tan(self._e2)
        tm[..., 3, 2] = -self.hx * torch.tan(self._e2 - phi)

        return tm

    def plot(
        self, s: float, vector_idx: tuple | None = None, ax: plt.Axes | None = None
    ) -> plt.Axes:
        ax = ax or plt.subplot(111)

        plot_s = s[vector_idx] if s.dim() > 0 else s
        plot_length = self.length[vector_idx] if self.length.dim() > 0 else self.length
        plot_angle = self.angle[vector_idx] if self.angle.dim() > 0 else self.angle

        alpha = 1 if self.is_active else 0.2
        height = 0.8 * (torch.sign(plot_angle) if self.is_active else 1)

        patch = Rectangle(
            (plot_s, 0), plot_length, height, color="tab:green", alpha=alpha, zorder=2
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

        # Rotate the mesh by half the bending angle
        mesh_rotation = trimesh.transformations.rotation_matrix(
            self.angle.item() / 2.0, [0, 1, 0], [0, 0, 0]
        )
        mesh.apply_transform(mesh_rotation)

        # Rotate the output transform by the full bending angle
        output_transform = (
            trimesh.transformations.rotation_matrix(
                self.angle.item(), [0, 1, 0], [0, 0, 0]
            )
            @ output_transform
        )

        return mesh, output_transform

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + [
            "length",
            "angle",
            "k1",
            "dipole_e1",
            "dipole_e2",
            "tilt",
            "gap",
            "gap_exit",
            "fringe_integral",
            "fringe_integral_exit",
            "fringe_at",
            "fringe_type",
            "tracking_method",
        ]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(length={repr(self.length)}, "
            + f"angle={repr(self.angle)}, "
            + f"k1={repr(self.k1)}, "
            + f"dipole_e1={repr(self.dipole_e1)},"
            + f"dipole_e2={repr(self.dipole_e2)},"
            + f"tilt={repr(self.tilt)},"
            + f"gap={repr(self.gap)},"
            + f"gap_exit={repr(self.gap_exit)},"
            + f"fringe_integral={repr(self.fringe_integral)},"
            + f"fringe_integral_exit={repr(self.fringe_integral_exit)},"
            + f"fringe_at={repr(self.fringe_at)},"
            + f"fringe_type={repr(self.fringe_type)},"
            + f"tracking_method={repr(self.tracking_method)}, "
            + f"name={repr(self.name)})"
        )
