from typing import Literal

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle

from cheetah.accelerator.element import Element
from cheetah.particles import Beam, ParticleBeam
from cheetah.utils import UniqueNameGenerator, verify_device_and_dtype

generate_unique_name = UniqueNameGenerator(prefix="unnamed_element")


class Multipole(Element):
    """
    Multipole magnet in a particle accelerator.

    :param length: Length in meters.
    :param polynom_a: Coefficients for skew multipole components (A_n) in 1/m^(n+1)
    :param polynom_b: Coefficients for normal multipole components (B_n) in 1/m^(n+1)
    :param max_order: Maximum order of the multipole field.
    :param misalignment: Misalignment vector of the element in x- and y-directions.
    :param tilt: Tilt angle of the element in x-y plane [rad].
    :param num_steps: Number of integration steps.
    :param fringe_quad_entrance: Whether to apply quadrupole fringe field at entrance
        (0=no fringe, 1=Lee-Whiting, 2=Lee-Whiting+Elegant).
    :param fringe_quad_exit: Whether to apply quadrupole fringe field at exit
        (0=no fringe, 1=Lee-Whiting, 2=Lee-Whiting+Elegant).
    :param fringe_int_m0: Fringe field integrals for entrance
        (I0m/K1, I1m/K1, I2m/K1, I3m/K1, Lambda2m/K1).
    :param fringe_int_p0: Fringe field integrals for exit
        (I0p/K1, I1p/K1, I2p/K1, I3p/K1, Lambda2p/K1).
    :param tracking_method: Method to use for tracking through the element.
    :param name: Unique identifier of the element.
    """

    def __init__(
        self,
        length: torch.Tensor,
        polynom_a: torch.Tensor | None = None,
        polynom_b: torch.Tensor | None = None,
        max_order: int = 1,
        misalignment: torch.Tensor | None = None,
        tilt: torch.Tensor | None = None,
        num_steps: int = 1,
        fringe_quad_entrance: int = 0,
        fringe_quad_exit: int = 0,
        fringe_int_m0: torch.Tensor | None = None,
        fringe_int_p0: torch.Tensor | None = None,
        tracking_method: Literal["symplectic4", "symplectic4_rad"] = "symplectic4",
        name: str | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        device, dtype = verify_device_and_dtype(
            [
                length,
                polynom_a,
                polynom_b,
                misalignment,
                tilt,
                fringe_int_m0,
                fringe_int_p0,
            ],
            device,
            dtype,
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(name=name, **factory_kwargs)

        self.length = torch.as_tensor(length, **factory_kwargs)

        # Set up multipole coefficients
        if polynom_a is None:
            # Initialize to zeros with size max_order+1
            polynom_a = torch.zeros(max_order + 1, **factory_kwargs)
        else:
            polynom_a = torch.as_tensor(polynom_a, **factory_kwargs)
            if polynom_a.size(0) <= max_order:
                # Pad with zeros if needed
                padding = torch.zeros(
                    max_order + 1 - polynom_a.size(0), **factory_kwargs
                )
                polynom_a = torch.cat([polynom_a, padding])
            else:
                polynom_a = polynom_a[: max_order + 1]

        if polynom_b is None:
            # Initialize to zeros with size max_order+1
            polynom_b = torch.zeros(max_order + 1, **factory_kwargs)
        else:
            polynom_b = torch.as_tensor(polynom_b, **factory_kwargs)
            if polynom_b.size(0) <= max_order:
                # Pad with zeros if needed
                padding = torch.zeros(
                    max_order + 1 - polynom_b.size(0), **factory_kwargs
                )
                polynom_b = torch.cat([polynom_b, padding])
            else:
                polynom_b = polynom_b[: max_order + 1]

        self.register_buffer_or_parameter("polynom_a", polynom_a)
        self.register_buffer_or_parameter("polynom_b", polynom_b)

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

        # Fringe field parameters
        self.fringe_quad_entrance = fringe_quad_entrance
        self.fringe_quad_exit = fringe_quad_exit

        # Default fringe integrals if not provided
        default_fringe_ints = torch.tensor([0.0, 0.5, 0.0, 0.0, 0.0], **factory_kwargs)

        if fringe_int_m0 is not None:
            self.register_buffer_or_parameter(
                "fringe_int_m0", torch.as_tensor(fringe_int_m0, **factory_kwargs)
            )
        else:
            self.register_buffer_or_parameter("fringe_int_m0", default_fringe_ints)

        if fringe_int_p0 is not None:
            self.register_buffer_or_parameter(
                "fringe_int_p0", torch.as_tensor(fringe_int_p0, **factory_kwargs)
            )
        else:
            self.register_buffer_or_parameter("fringe_int_p0", default_fringe_ints)

        self.max_order = max_order
        self.num_steps = num_steps
        self.tracking_method = tracking_method

    def track(self, incoming: Beam) -> Beam:
        """
        Track particles through the multipole element.

        :param incoming: Beam entering the element.
        :return: Beam exiting the element.
        """
        assert isinstance(
            incoming, ParticleBeam
        ), "Tracking is only supported for `ParticleBeam`."

        if self.tracking_method == "symplectic4":
            return self._track_symplectic4(incoming)
        elif self.tracking_method == "symplectic4_rad":
            return self._track_symplectic4_rad(incoming)
        else:
            raise ValueError(
                f"Invalid tracking method {self.tracking_method}. "
                + "Supported methods are 'symplectic4' and 'symplectic4_rad'."
            )

    def _track_symplectic4(self, incoming: ParticleBeam) -> ParticleBeam:
        """
        Track particles through the multipole element using the 4th order symplectic
        integration.

        :param incoming: ParticleBeam entering the element.
        :return: ParticleBeam exiting the element.
        """
        # Constants for 4th order symplectic integration
        DRIFT1 = 0.6756035959798286638
        DRIFT2 = -0.1756035959798286639
        KICK1 = 1.351207191959657328
        KICK2 = -1.702414383919314656

        # Get particle coordinates
        x = incoming.x
        px = incoming.px
        y = incoming.y
        py = incoming.py
        tau = incoming.tau
        delta = incoming.p

        # Apply misalignment at entrance if needed
        if torch.any(self.misalignment != 0) or torch.any(self.tilt != 0):
            # For now, let's ignore tilt and just apply simple offset
            x_offset = self.misalignment[..., 0]
            y_offset = self.misalignment[..., 1]
            x = x - x_offset
            y = y - y_offset

        # Check if quadrupole component exists (b2 != 0)
        b2 = 0.0
        if self.polynom_b.size(0) > 1:
            b2 = self.polynom_b[1].item()

        # Apply fringe field at entrance if enabled and we have a quadrupole component
        if self.fringe_quad_entrance and b2 != 0:
            if self.fringe_quad_entrance == 1:
                # Apply simple Lee-Whiting fringe field
                self._quad_fringe_pass_p(x, px, y, py, tau, delta, b2)
            elif self.fringe_quad_entrance == 2:
                # Apply Elegant-style fringe field
                self._apply_linear_quad_fringe_entrance(x, px, y, py, tau, delta, b2)

        # Prepare for tracking
        SL = self.length / self.num_steps
        L1 = SL * DRIFT1
        L2 = SL * DRIFT2
        K1 = SL * KICK1
        K2 = SL * KICK2

        # Symplectic integration loop
        for _ in range(self.num_steps):
            # First drift
            norm = 1.0 / (1.0 + delta)
            norm_l1 = L1 * norm
            x += norm_l1 * px
            y += norm_l1 * py
            tau += norm_l1 * (px * px + py * py) / (2.0 * (1.0 + delta))

            # First kick
            self._apply_kick(x, y, px, py, K1)

            # Second drift
            norm_l2 = L2 * norm
            x += norm_l2 * px
            y += norm_l2 * py
            tau += norm_l2 * (px * px + py * py) / (2.0 * (1.0 + delta))

            # Second kick
            self._apply_kick(x, y, px, py, K2)

            # Third drift
            x += norm_l2 * px
            y += norm_l2 * py
            tau += norm_l2 * (px * px + py * py) / (2.0 * (1.0 + delta))

            # Third kick
            self._apply_kick(x, y, px, py, K1)

            # Fourth drift
            x += norm_l1 * px
            y += norm_l1 * py
            tau += norm_l1 * (px * px + py * py) / (2.0 * (1.0 + delta))

        # Apply fringe field at exit if enabled and we have a quadrupole component
        if self.fringe_quad_exit and b2 != 0:
            if self.fringe_quad_exit == 1:
                # Apply simple Lee-Whiting fringe field
                self._quad_fringe_pass_n(x, px, y, py, tau, delta, b2)
            elif self.fringe_quad_exit == 2:
                # Apply Elegant-style fringe field
                self._apply_linear_quad_fringe_exit(x, px, y, py, tau, delta, b2)

        # Apply misalignment at exit if needed
        if torch.any(self.misalignment != 0) or torch.any(self.tilt != 0):
            # Restore coordinates to laboratory frame
            x = x + x_offset
            y = y + y_offset

        # Create output beam
        outgoing_beam = ParticleBeam(
            particles=torch.stack(
                (x, px, y, py, tau, delta, torch.ones_like(x)), dim=-1
            ),
            energy=incoming.energy,
            particle_charges=incoming.particle_charges,
            survival_probabilities=incoming.survival_probabilities,
            species=incoming.species,
        )

        return outgoing_beam

    def _track_symplectic4_rad(self, incoming: ParticleBeam) -> ParticleBeam:
        """
        Track particles through the multipole element using the 4th order symplectic
        integration with radiation effects included.

        :param incoming: ParticleBeam entering the element.
        :return: ParticleBeam exiting the element.
        """
        # Constants for 4th order symplectic integration
        DRIFT1 = 0.6756035959798286638
        DRIFT2 = -0.1756035959798286639
        KICK1 = 1.351207191959657328
        KICK2 = -1.702414383919314656

        # Get particle coordinates
        x = incoming.x
        px = incoming.px
        y = incoming.y
        py = incoming.py
        tau = incoming.tau
        delta = incoming.p

        # Apply misalignment at entrance if needed
        if torch.any(self.misalignment != 0) or torch.any(self.tilt != 0):
            # For now, let's ignore tilt and just apply simple offset
            x_offset = self.misalignment[..., 0]
            y_offset = self.misalignment[..., 1]
            x = x - x_offset
            y = y - y_offset

        # Check if quadrupole component exists (b2 != 0)
        b2 = 0.0
        if self.polynom_b.size(0) > 1:
            b2 = self.polynom_b[1].item()

        # Apply fringe field at entrance if enabled and we have a quadrupole component
        if self.fringe_quad_entrance and b2 != 0:
            if self.fringe_quad_entrance == 1:
                # Apply simple Lee-Whiting fringe field
                self._quad_fringe_pass_p(x, px, y, py, tau, delta, b2)
            elif self.fringe_quad_entrance == 2:
                # Apply Elegant-style fringe field
                self._apply_linear_quad_fringe_entrance(x, px, y, py, tau, delta, b2)

        # Prepare for tracking
        SL = self.length / self.num_steps
        L1 = SL * DRIFT1
        L2 = SL * DRIFT2
        K1 = SL * KICK1
        K2 = SL * KICK2

        # Get the beam energy (needed for radiation effects)
        energy = incoming.energy

        # Symplectic integration loop
        for _ in range(self.num_steps):
            # First drift
            self._drift6(x, px, y, py, tau, delta, L1)

            # First kick with radiation
            self._apply_kick_rad(x, y, px, py, delta, K1, energy)

            # Second drift
            self._drift6(x, px, y, py, tau, delta, L2)

            # Second kick with radiation
            self._apply_kick_rad(x, y, px, py, delta, K2, energy)

            # Third drift
            self._drift6(x, px, y, py, tau, delta, L2)

            # Third kick with radiation
            self._apply_kick_rad(x, y, px, py, delta, K1, energy)

            # Fourth drift
            self._drift6(x, px, y, py, tau, delta, L1)

        # Apply fringe field at exit if enabled and we have a quadrupole component
        if self.fringe_quad_exit and b2 != 0:
            if self.fringe_quad_exit == 1:
                # Apply simple Lee-Whiting fringe field
                self._quad_fringe_pass_n(x, px, y, py, tau, delta, b2)
            elif self.fringe_quad_exit == 2:
                # Apply Elegant-style fringe field
                self._apply_linear_quad_fringe_exit(x, px, y, py, tau, delta, b2)

        # Apply misalignment at exit if needed
        if torch.any(self.misalignment != 0) or torch.any(self.tilt != 0):
            # Restore coordinates to laboratory frame
            x = x + x_offset
            y = y + y_offset

        # Create output beam
        outgoing_beam = ParticleBeam(
            particles=torch.stack(
                (x, px, y, py, tau, delta, torch.ones_like(x)), dim=-1
            ),
            energy=incoming.energy,
            particle_charges=incoming.particle_charges,
            survival_probabilities=incoming.survival_probabilities,
            species=incoming.species,
        )

        return outgoing_beam

    def _drift6(
        self,
        x: torch.Tensor,
        px: torch.Tensor,
        y: torch.Tensor,
        py: torch.Tensor,
        tau: torch.Tensor,
        delta: torch.Tensor,
        length: float,
    ) -> None:
        """
        Apply a drift to the 6D phase space coordinates.

        This method implements a drift space transformation in full 6D phase space.
        The longitudinal coordinate (tau) update accounts for path length differences
        due to transverse motion, which is essential for maintaining symplecticity
        in the tracking. The implementation follows the approach used in PyAT.

        :param x: Horizontal position
        :param px: Horizontal momentum
        :param y: Vertical position
        :param py: Vertical momentum
        :param tau: Longitudinal position
        :param delta: Relative momentum deviation
        :param length: Length of the drift
        """
        # Equivalent to ATdrift6/fastdrift in the C code
        # Apply drift to positions
        norm = 1.0 / (1.0 + delta)
        norm_length = length * norm

        x += norm_length * px
        y += norm_length * py

        # Update longitudinal coordinate - matching PyAT's implementation
        # In PyAT, r[5] (tau) += NormL*(r[1]*r[1]+r[3]*r[3])/(2*(1+r[4])) where r[4] is
        # delta
        tau += norm_length * (px * px + py * py) / (2.0 * (1.0 + delta))

    def _apply_kick_rad(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        px: torch.Tensor,
        py: torch.Tensor,
        delta: torch.Tensor,
        kick_strength: float,
        energy: torch.Tensor,
    ) -> None:
        """
        Apply multipole kick to particles with radiation effects.

        This method calculates the magnetic field components as in _apply_kick,
        but also includes synchrotron radiation effects. The radiation calculation
        follows the model described by M. Sands, which accounts for energy loss
        due to photon emission in the magnetic field.

        :param x: Horizontal position
        :param y: Vertical position
        :param px: Horizontal momentum
        :param py: Vertical momentum
        :param delta: Relative momentum deviation
        :param kick_strength: Strength of the kick (includes step length factor)
        :param energy: Beam energy in GeV
        """
        # BUGFIX: Properly implement PyAT's field calculation algorithm
        # Always start with the value at max_order index, even if it's zero
        # This is critical for consistent results regardless of max_order
        ReSum = self.polynom_b[self.max_order].expand_as(x)
        ImSum = self.polynom_a[self.max_order].expand_as(y)

        # Apply the recursive algorithm exactly as in PyAT
        for i in range(self.max_order - 1, -1, -1):
            ReSumTemp = ReSum * x - ImSum * y + self.polynom_b[i]
            ImSum = ImSum * x + ReSum * y + self.polynom_a[i]
            ReSum = ReSumTemp

        # Constants for radiation calculation
        CGAMMA = 8.846e-5  # Radiation constant for electrons
        TWOPI = 2 * torch.pi

        # Store original delta for use in kick calculation (as in PyAT)
        delta_orig = delta.clone()

        # Calculate normalized velocities as in PyAT
        p_norm = 1.0 / (1.0 + delta)
        xpr = px * p_norm
        ypr = py * p_norm

        # Calculate B2P (perpendicular B-field squared) as in PyAT's StrB2perp
        v_norm2 = 1.0 / (1.0 + xpr**2 + ypr**2)
        bx = ImSum  # In PyAT, bx = ImSum
        by = ReSum  # In PyAT, by = ReSum
        B2P = (by**2 + bx**2 + (bx * ypr - by * xpr) ** 2) * v_norm2

        # Calculate CRAD according to M.Sands (4.1) as in PyAT
        CRAD = CGAMMA * energy**3 / (TWOPI * 1e27)  # [m]/[GeV^3]

        # For straight elements, irho = 0
        irho = 0.0

        # Calculate and apply energy loss as in PyAT
        delta.sub_(
            CRAD
            * (1.0 + delta) ** 2
            * B2P
            * (1.0 + (xpr**2 + ypr**2) / 2.0)
            * kick_strength
        )

        # Recalculate momenta from angles after energy loss
        p_norm = 1.0 / (1.0 + delta)
        px.copy_(xpr / p_norm)
        py.copy_(ypr / p_norm)

        # Apply kicks using the original delta value as in PyAT
        px.sub_(kick_strength * (ReSum - (delta_orig - x * irho) * irho))
        py.add_(kick_strength * ImSum)

    def _apply_kick(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        px: torch.Tensor,
        py: torch.Tensor,
        kick_strength: float,
    ) -> None:
        """
        Apply multipole kick to particles.

        This method calculates the magnetic field components from the multipole
        expansion and applies the resulting momentum kicks to the particles. The
        algorithm recursively computes the field components following the same approach
        as in PyAT.

        :param x: Horizontal position
        :param px: Horizontal momentum
        :param y: Vertical position
        :param py: Vertical momentum
        :param kick_strength: Strength of the kick (includes step length factor)
        """
        # BUGFIX: Properly implement PyAT's field calculation algorithm
        # Always start with the value at max_order index, even if it's zero
        # This is critical for consistent results regardless of max_order
        ReSum = self.polynom_b[self.max_order].expand_as(x)
        ImSum = self.polynom_a[self.max_order].expand_as(y)

        # Apply the recursive algorithm exactly as in PyAT
        for i in range(self.max_order - 1, -1, -1):
            ReSumTemp = ReSum * x - ImSum * y + self.polynom_b[i]
            ImSum = ImSum * x + ReSum * y + self.polynom_a[i]
            ReSum = ReSumTemp

        # Apply the kick
        px -= kick_strength * ReSum
        py += kick_strength * ImSum

    @property
    def is_skippable(self) -> bool:
        """
        Check if the element can be skipped during tracking.
        For multipoles, it's skippable if all multipole coefficients are zero.
        """
        return (
            torch.all(self.polynom_a == 0).item()
            and torch.all(self.polynom_b == 0).item()
        )

    def split(self, resolution: torch.Tensor) -> list[Element]:
        """
        Split the multipole element into smaller pieces for higher-resolution tracking.

        :param resolution: The desired resolution for splitting in meters.
        :return: List of Multipole elements that together make up the original element.
        """
        num_splits = torch.ceil(torch.max(self.length) / resolution).int()

        # For splits, only apply fringe at the very beginning and end
        split_elements = []
        for i in range(num_splits):
            # Only apply fringe at first and last element
            fringe_entrance = self.fringe_quad_entrance if i == 0 else 0
            fringe_exit = self.fringe_quad_exit if i == num_splits - 1 else 0

            split_elements.append(
                Multipole(
                    self.length / num_splits,
                    polynom_a=self.polynom_a,
                    polynom_b=self.polynom_b,
                    max_order=self.max_order,
                    misalignment=self.misalignment,
                    tilt=self.tilt,
                    num_steps=self.num_steps,
                    fringe_quad_entrance=fringe_entrance,
                    fringe_quad_exit=fringe_exit,
                    fringe_int_m0=self.fringe_int_m0,
                    fringe_int_p0=self.fringe_int_p0,
                    tracking_method=self.tracking_method,
                    dtype=self.length.dtype,
                    device=self.length.device,
                )
            )

        return split_elements

    @property
    def is_active(self) -> bool:
        """Check if the element has any non-zero multipole coefficients."""
        return (torch.any(self.polynom_a != 0) or torch.any(self.polynom_b != 0)).item()

    def plot(self, ax: plt.Axes, s: float, vector_idx: tuple | None = None) -> None:
        """Plot the multipole element."""
        plot_s = s[vector_idx] if s.dim() > 0 else s
        plot_length = self.length[vector_idx] if self.length.dim() > 0 else self.length

        alpha = 1 if self.is_active else 0.2
        # For general multipole, use a different color
        patch = Rectangle(
            (plot_s, 0), plot_length, 0.8, color="tab:purple", alpha=alpha, zorder=2
        )
        ax.add_patch(patch)

    @property
    def defining_features(self) -> list[str]:
        return super().defining_features + [
            "length",
            "polynom_a",
            "polynom_b",
            "max_order",
            "misalignment",
            "tilt",
            "fringe_quad_entrance",
            "fringe_quad_exit",
            "fringe_int_m0",
            "fringe_int_p0",
        ]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(length={repr(self.length)}, "
            + f"polynom_a={repr(self.polynom_a)}, "
            + f"polynom_b={repr(self.polynom_b)}, "
            + f"max_order={repr(self.max_order)}, "
            + f"misalignment={repr(self.misalignment)}, "
            + f"tilt={repr(self.tilt)}, "
            + f"num_steps={repr(self.num_steps)}, "
            + f"fringe_quad_entrance={repr(self.fringe_quad_entrance)}, "
            + f"fringe_quad_exit={repr(self.fringe_quad_exit)}, "
            + f"tracking_method={repr(self.tracking_method)}, "
            + f"name={repr(self.name)})"
        )

    def _quad_fringe_pass_p(
        self,
        x: torch.Tensor,
        px: torch.Tensor,
        y: torch.Tensor,
        py: torch.Tensor,
        tau: torch.Tensor,
        delta: torch.Tensor,
        b2: float,
    ) -> None:
        """
        Apply quadrupole fringe field effect at entrance using Lee-Whiting's formula.

        This is a PyTorch implementation of QuadFringePassP from quadfringe.c

        :param x: Horizontal position
        :param px: Horizontal momentum
        :param y: Vertical position
        :param py: Vertical momentum
        :param tau: Longitudinal position
        :param delta: Relative momentum deviation
        :param b2: Quadrupole strength coefficient (polynom_b[1])
        """
        # Calculate fringe field parameters
        u = b2 / (12.0 * (1.0 + delta))
        x2 = x * x
        y2 = y * y
        xy = x * y

        # Calculate displacements
        gx = u * (x2 + 3 * y2) * x
        gy = u * (y2 + 3 * x2) * y

        # Apply position changes
        x.add_(gx)
        y.sub_(gy)

        # Calculate momentum changes
        px_tmp = 3 * u * (2 * xy * py - (x2 + y2) * px)
        py_tmp = 3 * u * (2 * xy * px - (x2 + y2) * py)

        # Update longitudinal coordinate
        tau.sub_((gy * py - gx * px) / (1.0 + delta))

        # Apply momentum changes
        px.add_(px_tmp)
        py.sub_(py_tmp)

    def _quad_fringe_pass_n(
        self,
        x: torch.Tensor,
        px: torch.Tensor,
        y: torch.Tensor,
        py: torch.Tensor,
        tau: torch.Tensor,
        delta: torch.Tensor,
        b2: float,
    ) -> None:
        """
        Apply quadrupole fringe field effect at exit using Lee-Whiting's formula.

        This is a PyTorch implementation of QuadFringePassN from quadfringe.c

        :param x: Horizontal position
        :param px: Horizontal momentum
        :param y: Vertical position
        :param py: Vertical momentum
        :param tau: Longitudinal position
        :param delta: Relative momentum deviation
        :param b2: Quadrupole strength coefficient (polynom_b[1])
        """
        # Calculate fringe field parameters
        u = b2 / (12.0 * (1.0 + delta))
        x2 = x * x
        y2 = y * y
        xy = x * y

        # Calculate displacements
        gx = u * (x2 + 3 * y2) * x
        gy = u * (y2 + 3 * x2) * y

        # Apply position changes (opposite signs from entrance)
        x.sub_(gx)
        y.add_(gy)

        # Calculate momentum changes
        px_tmp = 3 * u * (2 * xy * py - (x2 + y2) * px)
        py_tmp = 3 * u * (2 * xy * px - (x2 + y2) * py)

        # Update longitudinal coordinate
        tau.add_((gy * py - gx * px) / (1.0 + delta))

        # Apply momentum changes (opposite signs from entrance)
        px.sub_(px_tmp)
        py.add_(py_tmp)

    def _quad_partial_fringe_matrix(
        self, K1: torch.Tensor, in_fringe: float, fringe_int: torch.Tensor, part: int
    ) -> torch.Tensor:
        """
        Generate partial fringe matrix for quadrupole fringe effects (Elegant-style).

        :param K1: Quadrupole strength divided by (1+delta)
        :param in_fringe: Fringe direction (-1 for entrance, 1 for exit)
        :param fringe_int: Fringe integrals
        :param part: Part number (1 or 2)
        :return: 6x6 transfer matrix
        """
        # Initialize 6x6 identity matrix
        R = torch.eye(6, device=K1.device, dtype=K1.dtype).expand(K1.shape[0], 6, 6)

        # Square of K1
        K1sqr = K1 * K1

        # Calculate J parameters according to part number
        if part == 1:
            J1x = in_fringe * (K1 * fringe_int[1] - 2 * K1sqr * fringe_int[3] / 3.0)
            J2x = in_fringe * (K1 * fringe_int[2])
            J3x = in_fringe * (K1sqr * (fringe_int[2] + fringe_int[4]))

            # For y-plane, use negative K1
            K1_y = -K1
            J1y = in_fringe * (
                K1_y * fringe_int[1] - 2 * K1_y * K1_y * fringe_int[3] / 3.0
            )
            J2y = -J2x
            J3y = J3x
        else:  # part == 2
            J1x = in_fringe * (
                K1 * fringe_int[1] + K1sqr * fringe_int[0] * fringe_int[2] / 2
            )
            J2x = in_fringe * (K1 * fringe_int[2])
            J3x = in_fringe * (K1sqr * (fringe_int[4] - fringe_int[0] * fringe_int[1]))

            # For y-plane, use negative K1
            K1_y = -K1
            J1y = in_fringe * (
                K1_y * fringe_int[1] + K1_y * K1_y * fringe_int[0] * fringe_int[2]
            )
            J2y = -J2x
            J3y = J3x

        # Calculate matrix elements
        exp_J1x = torch.exp(J1x)
        R[:, 0, 0] = exp_J1x
        R[:, 0, 1] = J2x / exp_J1x
        R[:, 1, 0] = exp_J1x * J3x
        R[:, 1, 1] = (1.0 + J2x * J3x) / exp_J1x

        exp_J1y = torch.exp(J1y)
        R[:, 2, 2] = exp_J1y
        R[:, 2, 3] = J2y / exp_J1y
        R[:, 3, 2] = exp_J1y * J3y
        R[:, 3, 3] = (1.0 + J2y * J3y) / exp_J1y

        return R

    def _apply_linear_quad_fringe_entrance(
        self,
        x: torch.Tensor,
        px: torch.Tensor,
        y: torch.Tensor,
        py: torch.Tensor,
        tau: torch.Tensor,
        delta: torch.Tensor,
        b2: float,
    ) -> None:
        """
        Apply quadrupole fringe field at entrance including linear Elegant-style
        effects.

        :param x: Horizontal position
        :param px: Horizontal momentum
        :param y: Vertical position
        :param py: Vertical momentum
        :param tau: Longitudinal position
        :param delta: Relative momentum deviation
        :param b2: Quadrupole strength coefficient (polynom_b[1])
        """
        # Elegant-style fringe field entrance (linear part)
        in_fringe = -1.0
        k1 = b2 / (1.0 + delta)

        # First linear matrix
        R = self._quad_partial_fringe_matrix(k1, in_fringe, self.fringe_int_p0, 1)

        # Apply first matrix
        x_new = R[:, 0, 0] * x + R[:, 0, 1] * px
        px_new = R[:, 1, 0] * x + R[:, 1, 1] * px
        y_new = R[:, 2, 2] * y + R[:, 2, 3] * py
        py_new = R[:, 3, 2] * y + R[:, 3, 3] * py

        x.copy_(x_new)
        px.copy_(px_new)
        y.copy_(y_new)
        py.copy_(py_new)

        # Apply nonlinear fringe field (AT code)
        self._quad_fringe_pass_p(x, px, y, py, tau, delta, b2)

        # Second linear matrix
        R = self._quad_partial_fringe_matrix(k1, in_fringe, self.fringe_int_m0, 2)

        # Apply second matrix
        x_new = R[:, 0, 0] * x + R[:, 0, 1] * px
        px_new = R[:, 1, 0] * x + R[:, 1, 1] * px
        y_new = R[:, 2, 2] * y + R[:, 2, 3] * py
        py_new = R[:, 3, 2] * y + R[:, 3, 3] * py

        x.copy_(x_new)
        px.copy_(px_new)
        y.copy_(y_new)
        py.copy_(py_new)

    def _apply_linear_quad_fringe_exit(
        self,
        x: torch.Tensor,
        px: torch.Tensor,
        y: torch.Tensor,
        py: torch.Tensor,
        tau: torch.Tensor,
        delta: torch.Tensor,
        b2: float,
    ) -> None:
        """
        Apply quadrupole fringe field at exit including linear Elegant-style effects.

        :param x: Horizontal position
        :param px: Horizontal momentum
        :param y: Vertical position
        :param py: Vertical momentum
        :param tau: Longitudinal position
        :param delta: Relative momentum deviation
        :param b2: Quadrupole strength coefficient (polynom_b[1])
        """
        # Elegant-style fringe field exit (linear part)
        in_fringe = 1.0
        k1 = b2 / (1.0 + delta)

        # First linear matrix
        R = self._quad_partial_fringe_matrix(k1, in_fringe, self.fringe_int_m0, 1)

        # Apply first matrix
        x_new = R[:, 0, 0] * x + R[:, 0, 1] * px
        px_new = R[:, 1, 0] * x + R[:, 1, 1] * px
        y_new = R[:, 2, 2] * y + R[:, 2, 3] * py
        py_new = R[:, 3, 2] * y + R[:, 3, 3] * py

        x.copy_(x_new)
        px.copy_(px_new)
        y.copy_(y_new)
        py.copy_(py_new)
