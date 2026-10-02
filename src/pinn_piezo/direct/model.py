"""Network architecture for the direct PINN."""

from __future__ import annotations

from collections import OrderedDict
import math

import numpy as np
import torch
import torch.nn.init as init
from torch import nn

from .. import materials
from ..config import CENTER, HEIGHT, REFERENCE_FORCE, WIDTH
from ..indirect.model import SplitMixedFieldNet, _phase_enriched_outputs
from ..scaling import direct_scales


def init_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)


def init_weights_siren(m, omega_0: float = 30.0):
    if isinstance(m, nn.Linear):
        num_input = m.weight.size(-1)
        with torch.no_grad():
            m.weight.uniform_(-np.sqrt(6 / num_input) / omega_0,
                              np.sqrt(6 / num_input) / omega_0)


class SinActivation(nn.Module):
    """Wrapper to use ``torch.sin`` inside an ``nn.Sequential``."""

    def forward(self, x):
        return torch.sin(x)


class BendingEnrichedSplitMixedFieldNet(SplitMixedFieldNet):
    """Two-trunk network with a zero-initialized slender-bending skip path.

    The basis contains no load or material constants and therefore does not
    prescribe a solution.  It merely exposes the low-order polynomial modes
    that represent cantilever curvature directly to the optimizer, alongside
    the unrestricted nonlinear primary and flux trunks.
    """

    def __init__(self, input_size, hidden_sizes, activation):
        super().__init__(input_size, hidden_sizes, activation)
        self.primary_bending = nn.Linear(7, 3, bias=False)
        init.zeros_(self.primary_bending.weight)

    def forward(self, inputs):
        outputs = super().forward(inputs)
        x_norm = inputs[:, 0:1]
        y_centered = inputs[:, 1:2] - 0.5
        basis = torch.cat((
            torch.ones_like(x_norm),
            x_norm,
            x_norm**2,
            x_norm**3,
            y_centered * x_norm,
            y_centered * x_norm**2,
            y_centered * x_norm**3,
        ), dim=1)
        primary = outputs[:, 0:3] + self.primary_bending(basis)
        return torch.cat((primary, outputs[:, 3:8]), dim=1)


class FCN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation=nn.Tanh,
                 reference_force: float = REFERENCE_FORCE,
                 normalization_force: float | None = None,
                 legacy: bool = False,
                 hard_natural_bcs: bool = False,
                 hard_floating_electrode: bool = False,
                 slender_warping: bool = False,
                 phase_enriched: bool = False,
                 split_trunks: bool = False,
                 constitutive_bridge: bool = False,
                 hard_axial_constitutive: bool = False,
                 bending_basis: bool = False,
                 beam_kinematics: bool = False,
                 bridge_correction_limit: float = 0.1,
                 beam_static_lift: bool = False,
                 traction_profile: str = "uniform",
                 point_load_width_ratio: float = 1.0 / 16.0):
        super().__init__()
        self.reference_force = float(reference_force)
        self.normalization_force = (
            abs(self.reference_force)
            if normalization_force is None else float(normalization_force)
        )
        if self.normalization_force <= 0.0:
            raise ValueError("normalization_force must be positive")
        self.scales = direct_scales(self.normalization_force)
        self.legacy = bool(legacy)
        self.hard_natural_bcs = bool(hard_natural_bcs)
        self.hard_floating_electrode = bool(hard_floating_electrode)
        self.slender_warping = bool(slender_warping)
        self.phase_enriched = bool(phase_enriched)
        self.split_trunks = bool(split_trunks)
        self.constitutive_bridge = bool(constitutive_bridge)
        self.hard_axial_constitutive = bool(hard_axial_constitutive)
        self.bending_basis = bool(bending_basis)
        self.beam_kinematics = bool(beam_kinematics)
        self.bridge_correction_limit = float(bridge_correction_limit)
        self.beam_static_lift = bool(beam_static_lift)
        self.traction_profile = str(traction_profile)
        self.point_load_width_ratio = float(point_load_width_ratio)
        if not 0.0 < self.bridge_correction_limit <= 1.0:
            raise ValueError("bridge_correction_limit must lie in (0, 1]")
        if self.legacy and self.phase_enriched:
            raise ValueError("legacy and phase_enriched modes are incompatible")
        if self.legacy and self.split_trunks:
            raise ValueError("legacy and split_trunks modes are incompatible")
        if self.constitutive_bridge and not self.split_trunks:
            raise ValueError("constitutive_bridge requires split_trunks=True")
        if self.bending_basis and not self.split_trunks:
            raise ValueError("bending_basis requires split_trunks=True")
        if self.beam_kinematics and not self.split_trunks:
            raise ValueError("beam_kinematics requires split_trunks=True")
        if self.beam_kinematics and self.legacy:
            raise ValueError("beam_kinematics is incompatible with legacy=True")
        if self.beam_static_lift and not self.constitutive_bridge:
            raise ValueError("beam_static_lift requires constitutive_bridge=True")
        if self.traction_profile not in ("uniform", "parabolic", "point"):
            raise ValueError(
                "traction_profile must be 'uniform', 'parabolic', or 'point'"
            )
        if not 0.0 < self.point_load_width_ratio <= 1.0:
            raise ValueError("point_load_width_ratio must lie in (0, 1]")
        if self.hard_floating_electrode:
            # Dimensionless voltage of the conducting floating electrode.
            self.floating_voltage = nn.Parameter(torch.zeros(1))

        network_input_size = input_size + 2 * int(self.phase_enriched)
        layers = [
            ('input', nn.Linear(network_input_size, hidden_sizes[0])),
            ('act0', activation()),
        ]
        for i in range(1, len(hidden_sizes)):
            layers.append((f'hidden_{i - 1}',
                           nn.Linear(hidden_sizes[i - 1], hidden_sizes[i])))
            layers.append((f'act_{i}', activation()))
        layers.append(('output', nn.Linear(hidden_sizes[-1], output_size)))

        if self.split_trunks:
            split_net = (
                BendingEnrichedSplitMixedFieldNet
                if self.bending_basis else SplitMixedFieldNet
            )
            self.net = split_net(
                network_input_size, list(hidden_sizes), activation,
            )
        else:
            self.net = nn.Sequential(OrderedDict(layers))

    def _material_coefficients(self, x, phase=None):
        """Return differentiable-device material columns for ``x``.

        The constants do not require gradients, but building them here lets
        the constitutive bridge work on CPU, CUDA, and MPS without consulting
        NumPy or the collocation dataset.
        """
        n = x.shape[0]

        def constant(value):
            return torch.full(
                (n, 1), float(value), dtype=x.dtype, device=x.device,
            )

        if phase is None:
            top = x[:, 1:2] >= CENTER
        else:
            top = phase.to(device=x.device) > 0
        e31 = torch.where(
            top, constant(materials.e31_top),
            constant(materials.e31_bottom),
        )
        e33 = torch.where(
            top, constant(materials.e33_top),
            constant(materials.e33_bottom),
        )
        return (
            constant(materials.C11), constant(materials.C12),
            constant(materials.C22), constant(materials.G),
            constant(materials.epsilon_1),
            constant(materials.epsilon_2), e31, e33,
        )

    def _constitutive_flux(self, x, primary, phase=None):
        """Reconstruct the stress-charge flux from ``(u,v,phi)``."""
        gradients = [
            torch.autograd.grad(
                primary[:, index].sum(), x,
                create_graph=True, retain_graph=True,
            )[0]
            for index in range(3)
        ]
        ux, uy = gradients[0][:, 0:1], gradients[0][:, 1:2]
        vx, vy = gradients[1][:, 0:1], gradients[1][:, 1:2]
        phix, phiy = gradients[2][:, 0:1], gradients[2][:, 1:2]
        C11, C12, C22, G, k1, k2, e31, e33 = (
            self._material_coefficients(x, phase)
        )
        Ex, Ey = -phix, -phiy
        return torch.cat((
            C11 * ux + C12 * vy - e31 * Ey,
            C12 * ux + C22 * vy - e33 * Ey,
            G * (uy + vx),
            k1 * Ex,
            e31 * ux + e33 * vy + k2 * Ey,
        ), dim=1)

    def _forward_impl(self, x, phase=None):
        if self.legacy:
            outputs = self.net(x)
            return torch.cat([
                x[:, 0:1] * outputs[:, 0:1],
                x[:, 0:1] * outputs[:, 1:2],
                (x[:, 1:2] / HEIGHT) * outputs[:, 2:3],
                outputs[:, 3:],
            ], dim=1)
        # Resolve the 100:1 geometric aspect ratio before the first layer.
        coordinates = torch.cat(
            (x[:, 0:1] / WIDTH, x[:, 1:2] / HEIGHT), dim=1,
        )
        if self.phase_enriched:
            outputs = _phase_enriched_outputs(self.net, x, phase)
        else:
            outputs = self.net(coordinates)
        phi = outputs[:, 2:3]

        x_norm = coordinates[:, 0:1]
        y_norm = coordinates[:, 1:2]
        u_dimensionless = outputs[:, 0:1]
        if self.slender_warping or self.beam_kinematics:
            center_coordinates = torch.cat(
                (coordinates[:, 0:1],
                 torch.full_like(coordinates[:, 1:2], 0.5)), dim=1,
            )
            if self.phase_enriched:
                center_outputs = _phase_enriched_outputs(
                    self.net, x, phase, center=True,
                )
            else:
                center_outputs = self.net(center_coordinates)
        if self.beam_kinematics:
            # Timoshenko-type decomposition of the dominant slender-beam mode.
            # The first two primary outputs represent shear deflection and
            # bending deflection.  With theta=v_b' and gamma=v_s', the large
            # bending terms cancel analytically instead of being learned as a
            # fragile u_y+v_x subtraction.
            slenderness = HEIGHT / WIDTH
            shear_deflection = (
                self.scales.shear / float(materials.G)
                * WIDTH * x_norm
                * center_outputs[:, 0:1]
            )
            bending_deflection = (
                self.scales.v * x_norm**2 * center_outputs[:, 1:2]
            )
            theta = torch.autograd.grad(
                bending_deflection.sum(), x,
                create_graph=True, retain_graph=True,
            )[0][:, 0:1]
            average_shear = torch.autograd.grad(
                shear_deflection.sum(), x,
                create_graph=True, retain_graph=True,
            )[0][:, 0:1]
            centerline_v = bending_deflection + shear_deflection
            # f'(s)=6s(1-s)-1 gives gamma_xy=6s(1-s)(w'-theta),
            # hence zero shear traction on the horizontal faces.
            shear_warping = (
                3.0 * y_norm**2 - 2.0 * y_norm**3 - y_norm
            )
            clamp_release = 1.0 - torch.exp(-x_norm / slenderness)
            curvature = torch.autograd.grad(
                theta.sum(), x, create_graph=True, retain_graph=True,
            )[0][:, 0:1]
            u_modified = (
                -(x[:, 1:2] - CENTER) * theta
                + HEIGHT * clamp_release * shear_warping * average_shear
            )
            poisson_contraction = (
                0.5 * float(materials.NU) * clamp_release
                * (x[:, 1:2] - CENTER) ** 2 * curvature
            )
            v_modified = centerline_v + poisson_contraction
        elif self.slender_warping:
            slenderness = HEIGHT / WIDTH
            v_dimensionless = (
                center_outputs[:, 1:2]
                + slenderness**2
                * (outputs[:, 1:2] - center_outputs[:, 1:2])
            )
            u_modified = self.scales.u * x_norm * u_dimensionless
            v_modified = self.scales.v * x_norm * v_dimensionless
        else:
            v_dimensionless = outputs[:, 1:2]
            u_modified = self.scales.u * x_norm * u_dimensionless
            v_modified = self.scales.v * x_norm * v_dimensionless
        if self.hard_floating_electrode:
            phi_modified = self.scales.phi * y_norm * (
                self.floating_voltage + (1.0 - y_norm) * phi
            )
        else:
            phi_modified = self.scales.phi * y_norm * phi

        external_tip_shear = None
        if self.hard_natural_bcs:
            dimensionless_load = (
                self.reference_force / (HEIGHT * self.scales.shear)
            )
            face_factor = y_norm * (1.0 - y_norm)
            tip_distance = (1.0 - x_norm) ** 2
            interior_tip_lift = face_factor / (face_factor + tip_distance + 1e-30)
            on_tip = torch.isclose(x_norm, torch.ones_like(x_norm))
            away_from_corners = face_factor > 0.0
            # Transfinite extension of the discontinuous corner data: uniform
            # shear on the open right face, zero shear on the open horizontal
            # faces.  The two corner values are assigned to the horizontal
            # condition; they have zero measure in the traction resultant.
            tip_lift = torch.where(
                on_tip,
                torch.where(away_from_corners,
                            torch.ones_like(face_factor),
                            torch.zeros_like(face_factor)),
                interior_tip_lift,
            )
            if self.traction_profile == "uniform":
                dimensionless_tip_traction = torch.full_like(
                    y_norm, dimensionless_load,
                )
            elif self.traction_profile == "parabolic":
                dimensionless_tip_traction = (
                    dimensionless_load * 6.0 * y_norm * (1.0 - y_norm)
                )
            else:
                width = self.point_load_width_ratio
                normalization = (
                    width * math.sqrt(math.pi / 2.0)
                    * math.erf(1.0 / (math.sqrt(2.0) * width))
                )
                dimensionless_tip_traction = (
                    dimensionless_load
                    * torch.exp(-0.5 * ((1.0 - y_norm) / width) ** 2)
                    / normalization
                )
            stress = torch.cat([
                self.scales.stress_x * (1.0 - x_norm) * outputs[:, 3:4],
                self.scales.stress_y * face_factor
                * outputs[:, 4:5],
                self.scales.shear * (
                    dimensionless_tip_traction * tip_lift
                    + face_factor * (1.0 - x_norm) * outputs[:, 5:6]
                ),
            ], dim=1)
            external_tip_shear = (
                self.scales.shear
                * dimensionless_tip_traction * tip_lift
            )
            electric_displacement = torch.cat([
                self.scales.electric_displacement
                * x_norm * (1.0 - x_norm) * outputs[:, 6:7],
                self.scales.electric_displacement * outputs[:, 7:8],
            ], dim=1)
        else:
            stress = torch.cat([
                self.scales.stress_x * outputs[:, 3:4],
                self.scales.stress_y * outputs[:, 4:5],
                self.scales.shear * outputs[:, 5:6],
            ], dim=1)
            electric_displacement = (
                self.scales.electric_displacement * outputs[:, 6:8]
            )

        primary = torch.cat(
            (u_modified, v_modified, phi_modified), dim=1,
        )
        learned_flux = torch.cat((stress, electric_displacement), dim=1)
        if self.constitutive_bridge:
            # The second trunk represents a mixed-field correction around the
            # constitutive flux.  Its trace is zero wherever an external flux
            # is prescribed.  Consequently the applied load and open-circuit
            # conditions cannot be satisfied by the flux trunk while leaving
            # (u,v,phi) at zero: boundary gradients must enter the primary
            # trunk from the first cold-training step.  The correction remains
            # free in the interior, where it retains the first-order mixed
            # approximation space used for equilibrium and Gauss' law.
            horizontal_bubble = 4.0 * y_norm * (1.0 - y_norm)
            right_factor = 1.0 - x_norm
            side_bubble = 4.0 * x_norm * (1.0 - x_norm)
            relative_correction = torch.tanh(torch.cat((
                learned_flux[:, 0:1] / self.scales.stress_x,
                learned_flux[:, 1:2] / self.scales.stress_y,
                learned_flux[:, 2:3] / self.scales.shear,
                learned_flux[:, 3:4]
                / self.scales.electric_displacement,
                learned_flux[:, 4:5]
                / self.scales.electric_displacement,
            ), dim=1))
            correction_factor = torch.cat((
                right_factor * relative_correction[:, 0:1],
                horizontal_bubble * relative_correction[:, 1:2],
                right_factor * horizontal_bubble
                * relative_correction[:, 2:3],
                side_bubble * relative_correction[:, 3:4],
                (1.0 - y_norm) * relative_correction[:, 4:5],
            ), dim=1)
            constitutive_flux = self._constitutive_flux(x, primary, phase)
            # A bounded relative correction cannot create a loaded flux field
            # when the primary fields are zero.  This rules out the observed
            # boundary-layer shortcut while retaining an independently
            # parameterized mixed correction once the primal solution grows.
            bridge_reference = constitutive_flux
            if self.hard_natural_bcs:
                # Couple the hard boundary load directly to the constitutive
                # primary branch while leaving the interior flux correction
                # trainable.  These masks impose only the actual boundary
                # tractions: unlike the optional beam_static_lift below, they
                # do not prescribe an equilibrated stress field in the beam.
                bridge_reference = torch.cat((
                    right_factor * constitutive_flux[:, 0:1],
                    horizontal_bubble * constitutive_flux[:, 1:2],
                    right_factor * horizontal_bubble
                    * constitutive_flux[:, 2:3] + external_tip_shear,
                    side_bubble * constitutive_flux[:, 3:4],
                    constitutive_flux[:, 4:5],
                ), dim=1)
            if self.beam_static_lift:
                force = self.reference_force
                sigma_x_lift = (
                    6.0 * force * (WIDTH - x[:, 0:1])
                    * (HEIGHT - 2.0 * x[:, 1:2]) / HEIGHT**3
                )
                tau_lift = (
                    6.0 * force * x[:, 1:2]
                    * (HEIGHT - x[:, 1:2]) / HEIGHT**3
                )
                bridge_reference = torch.cat((
                    sigma_x_lift,
                    torch.zeros_like(sigma_x_lift),
                    tau_lift,
                    constitutive_flux[:, 3:5],
                ), dim=1)
            learned_flux = bridge_reference * (
                1.0 + self.bridge_correction_limit * correction_factor
            )
        if self.hard_axial_constitutive:
            # Close only the axial stress exactly.  This removes the observed
            # mixed-field escape sigma_xx^NN != sigma_xx(u,v,phi) while
            # retaining sigma_yy, tau and D as independent first-order fields.
            # The free right-face value is consequently enforced by the
            # existing traction loss instead of the independent-output mask.
            constitutive_flux = self._constitutive_flux(x, primary, phase)
            learned_flux = torch.cat(
                (constitutive_flux[:, 0:1], learned_flux[:, 1:5]), dim=1,
            )
        return torch.cat((primary, learned_flux), dim=1)

    def forward(self, x, phase=None):
        if not (
            self.constitutive_bridge
            or self.beam_kinematics
            or self.hard_axial_constitutive
        ):
            return self._forward_impl(x, phase)
        # Boundary/evaluation tensors historically did not require coordinate
        # gradients.  The bridge needs first derivatives of the primary fields
        # even there, so create a local differentiable view when necessary.
        with torch.enable_grad():
            differentiable_x = x
            if not differentiable_x.requires_grad:
                differentiable_x = x.detach().requires_grad_(True)
            return self._forward_impl(differentiable_x, phase)

    def forward_trace(self, x, phase):
        """Evaluate a specified one-sided material trace at the interface."""
        if not self.phase_enriched:
            raise RuntimeError("one-sided traces require phase_enriched=True")
        return self.forward(x, phase=phase)


def build_default_model(device=None,
                        input_size: int = 2,
                        output_size: int = 8,
                        hidden_sizes=(100, 250),
                        reference_force: float = REFERENCE_FORCE,
                        normalization_force: float | None = None,
                        legacy: bool = False,
                        hard_natural_bcs: bool = False,
                        hard_floating_electrode: bool = False,
                        slender_warping: bool = False,
                        phase_enriched: bool = False,
                        split_trunks: bool = False,
                        constitutive_bridge: bool = False,
                        hard_axial_constitutive: bool = False,
                        bending_basis: bool = False,
                        beam_kinematics: bool = False,
                        bridge_correction_limit: float = 0.1,
                        beam_static_lift: bool = False,
                        traction_profile: str = "uniform",
                        point_load_width_ratio: float = 1.0 / 16.0,
                        output_init_gain: float = 0.0,
                        activation=nn.Tanh):
    """Reproduce the model build sequence used in PINN_pz_v3_directo.ipynb."""
    model = FCN(input_size, list(hidden_sizes), output_size,
                activation=activation,
                reference_force=reference_force,
                normalization_force=normalization_force,
                legacy=legacy,
                hard_natural_bcs=hard_natural_bcs,
                hard_floating_electrode=hard_floating_electrode,
                slender_warping=slender_warping,
                phase_enriched=phase_enriched,
                split_trunks=split_trunks,
                constitutive_bridge=constitutive_bridge,
                hard_axial_constitutive=hard_axial_constitutive,
                bending_basis=bending_basis,
                beam_kinematics=beam_kinematics,
                bridge_correction_limit=bridge_correction_limit,
                beam_static_lift=beam_static_lift,
                traction_profile=traction_profile,
                point_load_width_ratio=point_load_width_ratio)
    model.apply(init_weights)
    # A random O(1) thickness variation in v would imply strains O((L/H)^2).
    # The default therefore preserves the historical zero start.  A small
    # Xavier gain is exposed as a controlled diagnostic: it gives the trunk a
    # gradient on the first step without injecting O(1) transverse strain.
    if output_init_gain < 0.0:
        raise ValueError("output_init_gain must be non-negative")
    output_layers = (
        (model.net.primary.output, model.net.flux.output)
        if split_trunks else (model.net.output,)
    )
    for output_layer in output_layers:
        if output_init_gain == 0.0:
            init.zeros_(output_layer.weight)
        else:
            init.xavier_normal_(output_layer.weight, gain=output_init_gain)
        init.zeros_(output_layer.bias)
    if bending_basis:
        init.zeros_(model.net.primary_bending.weight)
    if device is not None:
        model.to(device)
    return model
