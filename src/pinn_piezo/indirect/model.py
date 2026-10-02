"""Network architectures for the indirect PINN."""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.init as init
from torch import nn

from .. import config, materials
from ..config import CENTER, HEIGHT, REFERENCE_VOLTAGE, WIDTH
from ..scaling import indirect_scales


def init_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)


def material_phase(x):
    """Return the known layer indicator without introducing a second net.

    A smooth MLP fed only ``(x, y)`` cannot represent the admissible jumps of
    ``sigma_xx`` and ``D_x`` at the bimorph interface.  Supplying the known
    phase as a fixed feature gives one shared network two one-sided traces.
    Derivatives of this indicator are zero inside each material, exactly where
    the strong-form residual is evaluated; interface transmission conditions
    are imposed separately in :mod:`pinn_piezo.indirect.losses`.
    """
    return torch.where(
        x[:, 1:2] >= CENTER,
        torch.ones_like(x[:, 1:2]),
        -torch.ones_like(x[:, 1:2]),
    )


def _network_input(x, phase=None, *, phase_enriched=False):
    coordinates = torch.cat(
        (x[:, 0:1] / WIDTH, x[:, 1:2] / HEIGHT), dim=1,
    )
    if not phase_enriched:
        return coordinates
    phase = material_phase(x) if phase is None else phase
    cusp = torch.abs(coordinates[:, 1:2] - 0.5)
    return torch.cat((
        coordinates,
        cusp,
        phase.to(dtype=x.dtype, device=x.device),
    ), dim=1)


def _center_network_input(x, phase=None, *, phase_enriched=False):
    coordinates = torch.cat(
        (x[:, 0:1] / WIDTH, torch.full_like(x[:, 1:2], 0.5)), dim=1,
    )
    if not phase_enriched:
        return coordinates
    phase = material_phase(x) if phase is None else phase
    cusp = torch.zeros_like(coordinates[:, 1:2])
    return torch.cat((
        coordinates,
        cusp,
        phase.to(dtype=x.dtype, device=x.device),
    ), dim=1)


def _mix_layer_traces(plus, minus, phase):
    """Apply the exact regularity class of the eight-field mixed solution.

    The bonded/interface fields ``u,v,phi,sigma_yy,tau,D_y`` use the mean of
    the two latent traces and are therefore continuous by construction.  Only
    tangential ``sigma_xx`` and ``D_x`` select a phase trace and may jump.
    The cusp coordinate still lets continuous fields have one-sided derivative
    jumps without smearing the material interface.
    """
    average = 0.5 * (plus + minus)
    selected = average + 0.5 * phase * (plus - minus)
    discontinuous = torch.zeros_like(average)
    discontinuous[:, 3:4] = 1.0
    discontinuous[:, 6:7] = 1.0
    return average + discontinuous * (selected - average)


def _phase_enriched_outputs(net, x, phase, *, center=False):
    phase = material_phase(x) if phase is None else phase.to(
        dtype=x.dtype, device=x.device,
    )
    plus = torch.ones_like(phase)
    input_builder = _center_network_input if center else _network_input
    raw_plus = net(input_builder(x, plus, phase_enriched=True))
    raw_minus = net(input_builder(x, -plus, phase_enriched=True))
    return _mix_layer_traces(raw_plus, raw_minus, phase)


def u_constraint(x, y):
    return 0  # x * nn.functional.relu(x)  # u = 0 at x = 0


def v_constraint(x, y):
    return 0  # x * nn.functional.relu(x)  # v = 0 at x = 0


def phi_constraint(x, y, voltage=None):
    """Linear electrode lifting for an explicitly selected voltage."""
    voltage = config.VOLTAGE if voltage is None else voltage
    return voltage / HEIGHT * y  # φ = 0 at y = 0, φ = V at y = H


def apply_voltage_constraints(x, outputs, center_outputs=None,
                              reference_voltage=REFERENCE_VOLTAGE,
                              hard_natural_bcs: bool = False):
    """Apply hard BCs and linear-load scaling to all predicted fields.

    The small-strain piezoelectric BVP is linear.  A network representing the
    solution at ``reference_voltage`` therefore represents another voltage by
    multiplying *all eight fields* by ``config.VOLTAGE/reference_voltage``.
    Previously only the affine phi lifting changed, leaving displacement,
    stress and electric displacement frozen at their 100 V values.

    This is exact superposition for a fixed geometry/material/BC set; it is not
    a claim of learned parametric generalisation.
    """
    scales = indirect_scales(reference_voltage)
    coordinates = torch.cat(
        (x[:, 0:1] / WIDTH, x[:, 1:2] / HEIGHT), dim=1,
    )
    x_norm, y_norm = coordinates[:, 0:1], coordinates[:, 1:2]
    center_outputs = outputs if center_outputs is None else center_outputs
    phi = outputs[:, 2:3]
    slenderness = HEIGHT / WIDTH
    # The O(1) through-thickness variation of u is the bending rotation and
    # must remain able to cancel the O(1/lambda) slope contribution to shear.
    u_dimensionless = outputs[:, 0:1]
    v_dimensionless = (
        center_outputs[:, 1:2]
        + slenderness**2 * (outputs[:, 1:2] - center_outputs[:, 1:2])
    )
    u_reference = scales.u * x_norm * u_dimensionless
    v_reference = scales.v * x_norm * v_dimensionless
    phi_reference = reference_voltage * (
        y_norm + y_norm * (y_norm - 1.0) * phi
    )
    sigma_x = outputs[:, 3:4]
    sigma_y = outputs[:, 4:5]
    tau = outputs[:, 5:6]
    d_x = outputs[:, 6:7]
    d_y = outputs[:, 7:8]
    if hard_natural_bcs:
        horizontal_bubble = 4.0 * y_norm * (1.0 - y_norm)
        right_factor = 1.0 - x_norm
        lateral_bubble = 4.0 * x_norm * (1.0 - x_norm)
        sigma_x = right_factor * sigma_x
        sigma_y = horizontal_bubble * sigma_y
        tau = right_factor * horizontal_bubble * tau
        d_x = lateral_bubble * d_x
    stress_reference = torch.cat([
        scales.stress_x * sigma_x,
        scales.stress_y * sigma_y,
        scales.shear * tau,
    ], dim=1)
    d_reference = scales.electric_displacement * torch.cat((d_x, d_y), dim=1)
    reference_fields = torch.cat(
        [u_reference, v_reference, phi_reference,
         stress_reference, d_reference], dim=1,
    )
    return (config.VOLTAGE / reference_voltage) * reference_fields


def _material_coefficients(x, phase=None):
    """Return the eight material columns on ``x`` without NumPy."""
    n = x.shape[0]

    def constant(value):
        return torch.full(
            (n, 1), float(value), dtype=x.dtype, device=x.device,
        )

    top = (
        x[:, 1:2] >= CENTER
        if phase is None else phase.to(device=x.device) > 0
    )
    e31 = torch.where(
        top, constant(materials.e31_top), constant(materials.e31_bottom),
    )
    e33 = torch.where(
        top, constant(materials.e33_top), constant(materials.e33_bottom),
    )
    return (
        constant(materials.C11), constant(materials.C12),
        constant(materials.C22), constant(materials.G),
        constant(materials.epsilon_1), constant(materials.epsilon_2),
        e31, e33,
    )


def _constitutive_flux(x, primary, phase=None):
    """Reconstruct ``(sigma_xx,sigma_yy,tau,Dx,Dy)`` from primaries."""
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
    C11, C12, C22, G, k1, k2, e31, e33 = _material_coefficients(
        x, phase,
    )
    Ex, Ey = -phix, -phiy
    return torch.cat((
        C11 * ux + C12 * vy - e31 * Ey,
        C12 * ux + C22 * vy - e33 * Ey,
        G * (uy + vx),
        k1 * Ex,
        e31 * ux + e33 * vy + k2 * Ey,
    ), dim=1)


def _apply_constitutive_bridge(
        x, fields, phase=None, correction_limit: float = 0.1):
    """Make the mixed flux a bounded correction around its constitutive value.

    The public model still has eight outputs.  The last five raw outputs now
    parameterize a relative correction instead of an unrelated flux field.
    Hence equilibrium, Gauss and Neumann losses have a gradient path to
    ``(u,v,phi)`` while the exact solution remains the zero-correction state.
    """
    scales = indirect_scales(config.VOLTAGE)
    x_norm = x[:, 0:1] / WIDTH
    y_norm = x[:, 1:2] / HEIGHT
    primary, raw_flux = fields[:, :3], fields[:, 3:8]
    relative = torch.tanh(torch.cat((
        raw_flux[:, 0:1] / scales.stress_x,
        raw_flux[:, 1:2] / scales.stress_y,
        raw_flux[:, 2:3] / scales.shear,
        raw_flux[:, 3:4] / scales.electric_displacement,
        raw_flux[:, 4:5] / scales.electric_displacement,
    ), dim=1))
    horizontal_bubble = 4.0 * y_norm * (1.0 - y_norm)
    right_factor = 1.0 - x_norm
    side_bubble = 4.0 * x_norm * (1.0 - x_norm)
    correction_factor = torch.cat((
        right_factor * relative[:, 0:1],
        horizontal_bubble * relative[:, 1:2],
        right_factor * horizontal_bubble * relative[:, 2:3],
        side_bubble * relative[:, 3:4],
        relative[:, 4:5],
    ), dim=1)
    constitutive = _constitutive_flux(x, primary, phase)
    bridged_flux = constitutive * (
        1.0 + correction_limit * correction_factor
    )
    return torch.cat((primary, bridged_flux), dim=1)


def apply_legacy_voltage_constraints(x, outputs,
                                     reference_voltage=REFERENCE_VOLTAGE):
    """Historical physical-coordinate transform for legacy checkpoints."""
    phi_reference = (
        x[:, 1:2] * (x[:, 1:2] - HEIGHT) * outputs[:, 2:3]
        + (reference_voltage / HEIGHT) * x[:, 1:2]
    )
    fields = torch.cat([
        x[:, 0:1] * outputs[:, 0:1],
        x[:, 0:1] * outputs[:, 1:2],
        phi_reference,
        outputs[:, 3:],
    ], dim=1)
    return (config.VOLTAGE / reference_voltage) * fields


class FCNUniform(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,
                 reference_voltage=REFERENCE_VOLTAGE, legacy: bool = False,
                 phase_enriched: bool = False,
                 hard_natural_bcs: bool = False,
                 activation=nn.Tanh):
        super().__init__()
        self.reference_voltage = float(reference_voltage)
        self.legacy = bool(legacy)
        self.phase_enriched = bool(phase_enriched)
        self.hard_natural_bcs = bool(hard_natural_bcs)
        if self.legacy and self.phase_enriched:
            raise ValueError("legacy and phase_enriched modes are incompatible")
        # Known cusp and phase coordinates are internal features; the public
        # PINN input remains the original physical pair (x, y).
        network_input_size = input_size + 2 * int(self.phase_enriched)

        layers = [
            ('input', nn.Linear(network_input_size, hidden_size)),
            ('act0', activation()),
        ]
        for i in range(num_layers):
            layers.append((f'hidden_{i}', nn.Linear(hidden_size, hidden_size)))
            layers.append((f'act_{i}', activation()))
        layers.append(('output', nn.Linear(hidden_size, output_size)))

        self.net = nn.Sequential(OrderedDict(layers))

    def forward(self, x, phase=None):
        if self.legacy:
            return apply_legacy_voltage_constraints(
                x, self.net(x), self.reference_voltage,
            )
        if self.phase_enriched:
            outputs = _phase_enriched_outputs(self.net, x, phase)
            center_outputs = _phase_enriched_outputs(
                self.net, x, phase, center=True,
            )
        else:
            outputs = self.net(_network_input(x))
            center_outputs = self.net(_center_network_input(x))
        return apply_voltage_constraints(
            x, outputs, center_outputs, self.reference_voltage,
            hard_natural_bcs=self.hard_natural_bcs,
        )

    def forward_trace(self, x, phase):
        """Evaluate a specified one-sided material trace at the interface."""
        if not self.phase_enriched:
            raise RuntimeError("one-sided traces require phase_enriched=True")
        return self.forward(x, phase=phase)


class FCNPyramid(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation=nn.Tanh,
                 reference_voltage=REFERENCE_VOLTAGE, legacy: bool = False,
                 phase_enriched: bool = False,
                 hard_natural_bcs: bool = False,
                 constitutive_bridge: bool = False,
                 bridge_correction_limit: float = 0.1):
        super().__init__()
        self.reference_voltage = float(reference_voltage)
        self.legacy = bool(legacy)
        self.phase_enriched = bool(phase_enriched)
        self.hard_natural_bcs = bool(hard_natural_bcs)
        self.constitutive_bridge = bool(constitutive_bridge)
        self.bridge_correction_limit = float(bridge_correction_limit)
        if self.legacy and self.phase_enriched:
            raise ValueError("legacy and phase_enriched modes are incompatible")
        if self.legacy and self.constitutive_bridge:
            raise ValueError(
                "legacy and constitutive_bridge modes are incompatible"
            )
        if not 0.0 < self.bridge_correction_limit <= 1.0:
            raise ValueError("bridge_correction_limit must lie in (0, 1]")

        # Known cusp and phase coordinates are internal features; the public
        # PINN input remains the original physical pair (x, y).
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

        self.net = nn.Sequential(OrderedDict(layers))

    def _forward_impl(self, x, phase=None):
        if self.legacy:
            return apply_legacy_voltage_constraints(
                x, self.net(x), self.reference_voltage,
            )
        if self.phase_enriched:
            outputs = _phase_enriched_outputs(self.net, x, phase)
            center_outputs = _phase_enriched_outputs(
                self.net, x, phase, center=True,
            )
        else:
            outputs = self.net(_network_input(x))
            center_outputs = self.net(_center_network_input(x))
        fields = apply_voltage_constraints(
            x, outputs, center_outputs, self.reference_voltage,
            hard_natural_bcs=self.hard_natural_bcs,
        )
        if self.constitutive_bridge:
            fields = _apply_constitutive_bridge(
                x, fields, phase,
                correction_limit=self.bridge_correction_limit,
            )
        return fields

    def forward(self, x, phase=None):
        if not self.constitutive_bridge:
            return self._forward_impl(x, phase)
        # Boundary and evaluation tensors need not request coordinate
        # gradients, but the constitutive reference requires first
        # derivatives of the primary fields there as well.
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


def _pyramid_branch(input_size, hidden_sizes, output_size, activation):
    """Build one branch of a split mixed-field network."""
    layers = [
        ("input", nn.Linear(input_size, hidden_sizes[0])),
        ("act0", activation()),
    ]
    for i in range(1, len(hidden_sizes)):
        layers.append((f"hidden_{i - 1}", nn.Linear(
            hidden_sizes[i - 1], hidden_sizes[i],
        )))
        layers.append((f"act_{i}", activation()))
    layers.append(("output", nn.Linear(hidden_sizes[-1], output_size)))
    return nn.Sequential(OrderedDict(layers))


class SplitMixedFieldNet(nn.Module):
    """Independent approximation spaces for primal and flux variables.

    The public output ordering remains exactly
    ``(u,v,phi,sigma_xx,sigma_yy,tau_xy,Dx,Dy)``.  Only the hidden parameter
    spaces are separated so boundary gradients acting on fluxes do not have to
    rewrite the representation used by the primal variables.
    """

    def __init__(self, input_size, hidden_sizes, activation):
        super().__init__()
        self.primary = _pyramid_branch(
            input_size, hidden_sizes, 3, activation,
        )
        self.flux = _pyramid_branch(
            input_size, hidden_sizes, 5, activation,
        )

    def forward(self, inputs):
        return torch.cat((self.primary(inputs), self.flux(inputs)), dim=1)


class FCNSplitPyramid(nn.Module):
    """Eight-field mixed PINN with separate primal and flux trunks."""

    def __init__(self, input_size, hidden_sizes, activation=nn.Tanh,
                 reference_voltage=REFERENCE_VOLTAGE,
                 phase_enriched: bool = False,
                 hard_natural_bcs: bool = False):
        super().__init__()
        self.reference_voltage = float(reference_voltage)
        self.legacy = False
        self.phase_enriched = bool(phase_enriched)
        self.hard_natural_bcs = bool(hard_natural_bcs)
        network_input_size = input_size + 2 * int(self.phase_enriched)
        self.net = SplitMixedFieldNet(
            network_input_size, list(hidden_sizes), activation,
        )

    def forward(self, x, phase=None):
        if self.phase_enriched:
            outputs = _phase_enriched_outputs(self.net, x, phase)
            center_outputs = _phase_enriched_outputs(
                self.net, x, phase, center=True,
            )
        else:
            outputs = self.net(_network_input(x))
            center_outputs = self.net(_center_network_input(x))
        return apply_voltage_constraints(
            x, outputs, center_outputs, self.reference_voltage,
            hard_natural_bcs=self.hard_natural_bcs,
        )

    def forward_trace(self, x, phase):
        if not self.phase_enriched:
            raise RuntimeError("one-sided traces require phase_enriched=True")
        return self.forward(x, phase=phase)


def get_model(input_size, hidden_sizes=None, output_size=None, type='uniform',
              num_layers=3, reference_voltage=REFERENCE_VOLTAGE,
              legacy: bool = False, phase_enriched: bool = False,
              hard_natural_bcs: bool = False, activation=nn.Tanh):
    """Build either architecture, including the notebook's tuple shorthand."""
    if type == 'uniform':
        if hidden_sizes is None and output_size is None:
            input_size, hidden_size, num_layers, output_size = input_size
        else:
            hidden_size = hidden_sizes
        return FCNUniform(input_size, hidden_size, num_layers, output_size,
                          reference_voltage=reference_voltage, legacy=legacy,
                          phase_enriched=phase_enriched,
                          hard_natural_bcs=hard_natural_bcs,
                          activation=activation)
    return FCNPyramid(input_size, hidden_sizes, output_size,
                      reference_voltage=reference_voltage, legacy=legacy,
                      phase_enriched=phase_enriched,
                      hard_natural_bcs=hard_natural_bcs,
                      activation=activation)


def build_default_model(device=None,
                        model_type: str = 'pyramid',
                        input_size: int = 2,
                        output_size: int = 8,
                        hidden_sizes=(100, 250),
                        reference_voltage: float = REFERENCE_VOLTAGE,
                        legacy: bool = False,
                        phase_enriched: bool = False,
                        hard_natural_bcs: bool = False,
                        activation=nn.Tanh,
                        zero_output: bool = True,
                        split_trunks: bool = False,
                        constitutive_bridge: bool = False,
                        bridge_correction_limit: float = 0.1):
    """Reproduce the model build sequence used in PINN_pz_v3.ipynb."""
    if split_trunks and model_type != 'pyramid':
        raise ValueError("split_trunks requires model_type='pyramid'")
    if split_trunks and legacy:
        raise ValueError("split_trunks is incompatible with legacy mode")
    if constitutive_bridge and model_type != "pyramid":
        raise ValueError("constitutive_bridge requires model_type='pyramid'")
    if constitutive_bridge and split_trunks:
        raise ValueError(
            "indirect constitutive_bridge currently requires one shared trunk"
        )
    if split_trunks:
        if output_size != 8:
            raise ValueError("split_trunks requires the canonical 8 outputs")
        model = FCNSplitPyramid(
            input_size, list(hidden_sizes),
            reference_voltage=reference_voltage,
            phase_enriched=phase_enriched,
            hard_natural_bcs=hard_natural_bcs,
            activation=activation,
        )
    elif model_type == 'pyramid':
        model = FCNPyramid(input_size, list(hidden_sizes), output_size,
                           reference_voltage=reference_voltage, legacy=legacy,
                           phase_enriched=phase_enriched,
                           hard_natural_bcs=hard_natural_bcs,
                           constitutive_bridge=constitutive_bridge,
                           bridge_correction_limit=bridge_correction_limit,
                           activation=activation)
    else:
        # Uniform architecture used by the original notebook.
        model = FCNUniform(input_size, hidden_size=300, num_layers=3,
                           output_size=output_size,
                           reference_voltage=reference_voltage, legacy=legacy,
                           phase_enriched=phase_enriched,
                           hard_natural_bcs=hard_natural_bcs,
                           activation=activation)
    model.apply(init_weights)
    if zero_output:
        # The hard voltage lifting already supplies the non-trivial initial
        # field.  This legacy default is optional so a strictly cold Xavier
        # ablation can initialize every layer randomly.
        if split_trunks:
            for branch in (model.net.primary, model.net.flux):
                init.zeros_(branch.output.weight)
                init.zeros_(branch.output.bias)
        else:
            init.zeros_(model.net.output.weight)
            init.zeros_(model.net.output.bias)
    if device is not None:
        model.to(device)
    return model
