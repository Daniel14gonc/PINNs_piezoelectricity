"""Three-field strong-form PINN for the indirect piezoelectric benchmark.

The network predicts only ``(u, v, phi)``.  Stress and electric displacement
are reconstructed exactly from the linear constitutive law, so equilibrium and
Gauss require second derivatives.  This is a conventional/primal PINN baseline
for comparison with the paper's eight-output mixed first-order formulation.
No energy or variational functional is used.
"""

from __future__ import annotations

import time
from collections import OrderedDict

import torch
from torch import nn

from .. import config, materials
from ..config import CENTER, HEIGHT, REFERENCE_VOLTAGE, WIDTH
from ..scaling import indirect_scales
from .model import init_weights


class StandardFCNPyramid(nn.Module):
    """One SiLU/tanh trunk with three physical outputs and hard essential BCs."""

    def __init__(self, input_size, hidden_sizes, output_size=3,
                 activation=nn.SiLU,
                 reference_voltage=REFERENCE_VOLTAGE,
                 interface_enriched: bool = True,
                 phase_enriched: bool = False):
        super().__init__()
        self.reference_voltage = float(reference_voltage)
        self.interface_enriched = bool(interface_enriched)
        self.phase_enriched = bool(phase_enriched)
        if self.phase_enriched:
            self.interface_enriched = True
        network_input_size = (
            input_size
            + int(self.interface_enriched)
            + int(self.phase_enriched)
        )
        layers = [
            ("input", nn.Linear(network_input_size, hidden_sizes[0])),
            ("act0", activation()),
        ]
        for i in range(1, len(hidden_sizes)):
            layers.append((f"hidden_{i - 1}",
                           nn.Linear(hidden_sizes[i - 1], hidden_sizes[i])))
            layers.append((f"act_{i}", activation()))
        layers.append(("output", nn.Linear(hidden_sizes[-1], output_size)))
        self.net = nn.Sequential(OrderedDict(layers))

    def _features(self, xy, *, center=False, phase=None):
        x_norm = xy[:, 0:1] / WIDTH
        y_norm = (
            torch.full_like(xy[:, 1:2], 0.5)
            if center else xy[:, 1:2] / HEIGHT
        )
        features = [x_norm, y_norm]
        if self.interface_enriched:
            features.append(torch.abs(y_norm - 0.5))
        if self.phase_enriched:
            if phase is None:
                phase = torch.where(
                    xy[:, 1:2] >= CENTER,
                    torch.ones_like(xy[:, 1:2]),
                    -torch.ones_like(xy[:, 1:2]),
                )
            features.append(phase.to(dtype=xy.dtype, device=xy.device))
        return torch.cat(features, dim=1)

    def _raw(self, xy, *, center=False):
        if not self.phase_enriched:
            return self.net(self._features(xy, center=center))
        phase = torch.ones_like(xy[:, 1:2])
        plus = self.net(self._features(
            xy, center=center, phase=phase,
        ))
        minus = self.net(self._features(
            xy, center=center, phase=-phase,
        ))
        # The primary fields are continuous at the material interface.  This
        # is exactly the trace averaging used for u, v and phi by the mixed
        # phase-enriched model, while retaining only three network outputs.
        return 0.5 * (plus + minus)

    def forward(self, xy):
        scales = indirect_scales(self.reference_voltage)
        raw = self._raw(xy)
        center_raw = self._raw(xy, center=True)
        x_norm = xy[:, 0:1] / WIDTH
        y_norm = xy[:, 1:2] / HEIGHT
        slenderness = HEIGHT / WIDTH

        u_reference = scales.u * x_norm * raw[:, 0:1]
        v_dimensionless = (
            center_raw[:, 1:2]
            + slenderness**2 * (raw[:, 1:2] - center_raw[:, 1:2])
        )
        v_reference = scales.v * x_norm * v_dimensionless
        phi_reference = self.reference_voltage * (
            y_norm + y_norm * (y_norm - 1.0) * raw[:, 2:3]
        )
        reference_fields = torch.cat(
            (u_reference, v_reference, phi_reference), dim=1,
        )
        return (config.VOLTAGE / self.reference_voltage) * reference_fields


def build_standard_model(device=None, input_size=2, hidden_sizes=(100, 250),
                         activation=nn.SiLU,
                         reference_voltage=REFERENCE_VOLTAGE,
                         interface_enriched=True,
                         phase_enriched=False,
                         zero_output=False):
    model = StandardFCNPyramid(
        input_size, list(hidden_sizes), output_size=3,
        activation=activation,
        reference_voltage=reference_voltage,
        interface_enriched=interface_enriched,
        phase_enriched=phase_enriched,
    )
    model.apply(init_weights)
    if zero_output:
        # Optional controlled baseline: zero displacement plus electrode lift.
        torch.nn.init.zeros_(model.net.output.weight)
        torch.nn.init.zeros_(model.net.output.bias)
    if device is not None:
        model.to(device)
    return model


def coefficients_at(xy, phase=None):
    """Build the eight coefficient columns directly on ``xy.device``."""
    n = xy.shape[0]
    dtype, device = xy.dtype, xy.device

    def constant(value):
        return torch.full((n, 1), float(value), dtype=dtype, device=device)

    if phase is None:
        top = xy[:, 1:2] >= CENTER
    else:
        top = phase.to(device=device) > 0
    e31 = torch.where(
        top, constant(materials.e31_top), constant(materials.e31_bottom),
    )
    e33 = torch.where(
        top, constant(materials.e33_top), constant(materials.e33_bottom),
    )
    return torch.cat((
        constant(materials.C11), constant(materials.C12),
        constant(materials.C22), constant(materials.G),
        constant(materials.epsilon_1), constant(materials.epsilon_2),
        e31, e33,
    ), dim=1)


def _fields_and_flux(xy, model, coefficients):
    fields = model(xy)
    gradients = [
        torch.autograd.grad(
            fields[:, i].sum(), xy, create_graph=True, retain_graph=True,
        )[0]
        for i in range(3)
    ]
    ux, uy = gradients[0][:, 0:1], gradients[0][:, 1:2]
    vx, vy = gradients[1][:, 0:1], gradients[1][:, 1:2]
    phix, phiy = gradients[2][:, 0:1], gradients[2][:, 1:2]

    C11, C12 = coefficients[:, 0:1], coefficients[:, 1:2]
    C22, G = coefficients[:, 2:3], coefficients[:, 3:4]
    k1, k2 = coefficients[:, 4:5], coefficients[:, 5:6]
    e31, e33 = coefficients[:, 6:7], coefficients[:, 7:8]
    exx, eyy, gamma = ux, vy, uy + vx
    Ex, Ey = -phix, -phiy
    sigma_x = C11 * exx + C12 * eyy - e31 * Ey
    sigma_y = C12 * exx + C22 * eyy - e33 * Ey
    tau = G * gamma
    Dx = k1 * Ex
    Dy = e31 * exx + e33 * eyy + k2 * Ey
    return fields, sigma_x, sigma_y, tau, Dx, Dy


def physics_loss_standard(x, y, model, coefficients, *, return_terms=False):
    """Second-order equilibrium and Gauss residuals, normalized by units."""
    xy = torch.cat((x, y), dim=1)
    _, sigma_x, sigma_y, tau, Dx, Dy = _fields_and_flux(
        xy, model, coefficients,
    )

    def gradient(field):
        return torch.autograd.grad(
            field.sum(), xy, create_graph=True, retain_graph=True,
        )[0]

    sigma_x_grad = gradient(sigma_x)
    sigma_y_grad = gradient(sigma_y)
    tau_grad = gradient(tau)
    Dx_grad = gradient(Dx)
    Dy_grad = gradient(Dy)
    scales = indirect_scales(config.VOLTAGE)
    rx = (sigma_x_grad[:, 0:1] + tau_grad[:, 1:2]) / scales.equilibrium_x
    ry = (tau_grad[:, 0:1] + sigma_y_grad[:, 1:2]) / scales.equilibrium_y
    rD = (Dx_grad[:, 0:1] + Dy_grad[:, 1:2]) / scales.gauss
    terms = {
        "equilibrium_x": torch.mean(rx**2),
        "equilibrium_y": torch.mean(ry**2),
        "gauss": torch.mean(rD**2),
    }
    total = sum(terms.values())
    return (total, terms) if return_terms else total


def _boundary_fields(model, xy, phase=None):
    points = xy.detach().clone().requires_grad_(True)
    coeff = coefficients_at(points, phase=phase)
    return _fields_and_flux(points, model, coeff)


def bc_loss_standard(xy_top, xy_bottom, xy_right, xy_left, model,
                     *, return_terms=False):
    """All mechanical free faces plus side insulation."""
    scales = indirect_scales(config.VOLTAGE)
    top = _boundary_fields(model, xy_top)
    bottom = _boundary_fields(model, xy_bottom)
    right = _boundary_fields(model, xy_right)
    left = _boundary_fields(model, xy_left)
    terms = {
        "top_sigma_y": torch.mean((top[2] / scales.stress_y) ** 2),
        "top_tau": torch.mean((top[3] / scales.shear) ** 2),
        "bottom_sigma_y": torch.mean((bottom[2] / scales.stress_y) ** 2),
        "bottom_tau": torch.mean((bottom[3] / scales.shear) ** 2),
        "right_sigma_x": torch.mean((right[1] / scales.stress_x) ** 2),
        "right_tau": torch.mean((right[3] / scales.shear) ** 2),
        "right_Dx": torch.mean((right[4] / scales.electric_displacement) ** 2),
        "left_Dx": torch.mean((left[4] / scales.electric_displacement) ** 2),
    }
    total = sum(terms.values())
    return (total, terms) if return_terms else total


def interface_loss_standard(xy_interface, model, *, return_terms=False):
    """One-sided bonded-interface traces for fields, traction and normal D."""
    eps = 1e-7 * HEIGHT
    x = xy_interface[:, 0:1]
    top_xy = torch.cat((x, torch.full_like(x, CENTER + eps)), dim=1)
    bottom_xy = torch.cat((x, torch.full_like(x, CENTER - eps)), dim=1)
    top = _boundary_fields(model, top_xy, phase=torch.ones_like(x))
    bottom = _boundary_fields(model, bottom_xy, phase=-torch.ones_like(x))
    scales = indirect_scales(config.VOLTAGE)
    names = ("u", "v", "phi", "sigma_y", "tau", "Dy")
    indices = (0, 0, 0, 2, 3, 5)
    columns = (0, 1, 2, None, None, None)
    magnitudes = (
        scales.u, scales.v, scales.phi,
        scales.stress_y, scales.shear, scales.electric_displacement,
    )
    terms = {}
    for name, index, column, magnitude in zip(
        names, indices, columns, magnitudes,
    ):
        top_value = top[index]
        bottom_value = bottom[index]
        if column is not None:
            top_value = top_value[:, column:column + 1]
            bottom_value = bottom_value[:, column:column + 1]
        terms[f"jump_{name}"] = torch.mean(
            ((top_value - bottom_value) / magnitude) ** 2
        )
    total = sum(terms.values())
    return (total, terms) if return_terms else total


def loss_standard(tensors, model, *, return_terms=False,
                  pde_weight=1.0, bc_weight=1.0, interface_weight=1.0):
    pde, pde_terms = physics_loss_standard(
        tensors["x_collocation"], tensors["y_collocation"], model,
        tensors["coefficients"], return_terms=True,
    )
    bc, bc_terms = bc_loss_standard(
        tensors["xy_top"], tensors["xy_bottom"],
        tensors["xy_right"], tensors["xy_left"], model,
        return_terms=True,
    )
    interface, interface_terms = interface_loss_standard(
        tensors["xy_interface"], model, return_terms=True,
    )
    total = pde_weight * pde + bc_weight * bc + interface_weight * interface
    if not return_terms:
        return total
    terms = {**pde_terms, **bc_terms, **interface_terms}
    terms.update({"pde_total": pde, "bc_total": bc,
                  "interface_total": interface, "total": total})
    return total, terms


def train_standard(model, tensors, *,
                   epochs_adam=1000, epochs_lbfgs=20,
                   lr_adam=0.001, lr_lbfgs=1.0,
                   log_every=100, resample_fn=None, resample_every=200,
                   lbfgs_tensors=None):
    """Adam with controlled resampling, then deterministic strong-Wolfe."""
    history = []
    component_history = {}

    def record(terms):
        for name, value in terms.items():
            component_history.setdefault(name, []).append(
                float(value.detach().cpu())
            )

    start = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_adam)
    for epoch in range(epochs_adam):
        if (resample_fn is not None and epoch > 0
                and epoch % resample_every == 0):
            tensors.clear()
            tensors.update(resample_fn(epoch // resample_every))
        optimizer.zero_grad()
        loss, terms = loss_standard(tensors, model, return_terms=True)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
        record(terms)
        if epoch % log_every == 0:
            print(f"[ADAM] {epoch}/{epochs_adam} loss={loss.item():.6e}")

    fixed = tensors if lbfgs_tensors is None else lbfgs_tensors
    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=lr_lbfgs, line_search_fn="strong_wolfe",
    )
    for epoch in range(epochs_lbfgs):
        def closure():
            optimizer.zero_grad()
            loss = loss_standard(fixed, model)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        history.append(float(loss.detach()))
        with torch.enable_grad():
            _, terms = loss_standard(fixed, model, return_terms=True)
        record(terms)
        if epoch % log_every == 0 or epoch == epochs_lbfgs - 1:
            print(f"[LBFGS] {epoch}/{epochs_lbfgs} loss={loss.item():.6e}")

    _, final_terms = loss_standard(fixed, model, return_terms=True)
    return {
        "loss_list": history,
        "loss_components": component_history,
        "total_time": time.time() - start,
        "final_terms": {
            name: float(value.detach().cpu())
            for name, value in final_terms.items()
        },
    }
