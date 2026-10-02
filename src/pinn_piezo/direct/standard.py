"""Three-field strong-form PINN for the direct piezoelectric benchmark.

The network predicts only ``(u, v, phi)``.  Stress and electric displacement
are reconstructed from the stress-charge constitutive law.  This module uses
only strong-form PDE and boundary residuals; it contains no energy functional.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn

from ..config import CENTER, HEIGHT, REFERENCE_FORCE, WIDTH
from ..indirect.standard import coefficients_at
from ..scaling import direct_scales
from .model import init_weights


class DirectStandardFCN(nn.Module):
    """Three-output network with hard clamp, ground and interface cusp."""

    def __init__(self, input_size, hidden_sizes, output_size=3,
                 activation=nn.Tanh, reference_force=REFERENCE_FORCE,
                 interface_enriched=True, phase_enriched=False,
                 slender_warping=False, hard_floating_electrode=False):
        super().__init__()
        self.reference_force = float(reference_force)
        self.interface_enriched = bool(interface_enriched)
        self.phase_enriched = bool(phase_enriched)
        self.slender_warping = bool(slender_warping)
        self.hard_floating_electrode = bool(hard_floating_electrode)
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
        for index in range(1, len(hidden_sizes)):
            layers.append((
                f"hidden_{index - 1}",
                nn.Linear(hidden_sizes[index - 1], hidden_sizes[index]),
            ))
            layers.append((f"act_{index}", activation()))
        layers.append(("output", nn.Linear(hidden_sizes[-1], output_size)))
        self.net = nn.Sequential(OrderedDict(layers))
        if self.hard_floating_electrode:
            self.floating_voltage = nn.Parameter(torch.zeros(1))

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
        return 0.5 * (plus + minus)

    def forward(self, xy):
        raw = self._raw(xy)
        scales = direct_scales(self.reference_force)
        x_norm = xy[:, 0:1] / WIDTH
        y_norm = xy[:, 1:2] / HEIGHT
        if self.slender_warping:
            center_raw = self._raw(xy, center=True)
            slenderness = HEIGHT / WIDTH
            v_raw = (
                center_raw[:, 1:2]
                + slenderness**2
                * (raw[:, 1:2] - center_raw[:, 1:2])
            )
        else:
            v_raw = raw[:, 1:2]
        if self.hard_floating_electrode:
            phi = scales.phi * y_norm * (
                self.floating_voltage + (1.0 - y_norm) * raw[:, 2:3]
            )
        else:
            phi = scales.phi * y_norm * raw[:, 2:3]
        return torch.cat((
            scales.u * x_norm * raw[:, 0:1],
            scales.v * x_norm * v_raw,
            phi,
        ), dim=1)


def build_standard_model(device=None, input_size=2, hidden_sizes=(100, 250),
                         activation=nn.Tanh,
                         reference_force=REFERENCE_FORCE,
                         interface_enriched=True,
                         phase_enriched=False,
                         slender_warping=False,
                         hard_floating_electrode=False,
                         output_init_gain=1e-4):
    model = DirectStandardFCN(
        input_size, list(hidden_sizes), activation=activation,
        reference_force=reference_force,
        interface_enriched=interface_enriched,
        phase_enriched=phase_enriched,
        slender_warping=slender_warping,
        hard_floating_electrode=hard_floating_electrode,
    )
    model.apply(init_weights)
    if output_init_gain < 0.0:
        raise ValueError("output_init_gain must be non-negative")
    if output_init_gain == 0.0:
        nn.init.zeros_(model.net.output.weight)
    else:
        nn.init.xavier_normal_(
            model.net.output.weight, gain=output_init_gain,
        )
    nn.init.zeros_(model.net.output.bias)
    if device is not None:
        model.to(device)
    return model


def fields_and_flux(xy, model, coefficients):
    """Return primal fields and constitutively derived ``sigma`` and ``D``."""
    fields = model(xy)
    gradients = [
        torch.autograd.grad(
            fields[:, index].sum(), xy,
            create_graph=True, retain_graph=True,
        )[0]
        for index in range(3)
    ]
    ux, uy = gradients[0][:, 0:1], gradients[0][:, 1:2]
    vx, vy = gradients[1][:, 0:1], gradients[1][:, 1:2]
    phix, phiy = gradients[2][:, 0:1], gradients[2][:, 1:2]
    C11, C12 = coefficients[:, 0:1], coefficients[:, 1:2]
    C22, G = coefficients[:, 2:3], coefficients[:, 3:4]
    k1, k2 = coefficients[:, 4:5], coefficients[:, 5:6]
    e31, e33 = coefficients[:, 6:7], coefficients[:, 7:8]
    Ex, Ey = -phix, -phiy
    sigma_x = C11 * ux + C12 * vy - e31 * Ey
    sigma_y = C12 * ux + C22 * vy - e33 * Ey
    tau = G * (uy + vx)
    Dx = k1 * Ex
    Dy = e31 * ux + e33 * vy + k2 * Ey
    return fields, sigma_x, sigma_y, tau, Dx, Dy


def physics_loss_standard(x, y, model, coefficients, *, return_terms=False):
    """Strong equilibrium and Gauss residuals from the three primal fields."""
    xy = torch.cat((x, y), dim=1)
    _, sigma_x, sigma_y, tau, Dx, Dy = fields_and_flux(
        xy, model, coefficients,
    )

    def gradient(field):
        if not field.requires_grad:
            return torch.zeros_like(xy)
        return torch.autograd.grad(
            field.sum(), xy, create_graph=True, retain_graph=True,
        )[0]

    sigma_x_grad = gradient(sigma_x)
    sigma_y_grad = gradient(sigma_y)
    tau_grad = gradient(tau)
    Dx_grad = gradient(Dx)
    Dy_grad = gradient(Dy)
    scales = direct_scales(model.reference_force)
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
    coefficients = coefficients_at(points, phase=phase)
    return fields_and_flux(points, model, coefficients)


def bc_loss_standard(xy_top, xy_bottom, xy_right, xy_left, model,
                     *, return_terms=False):
    """Mechanical tractions and floating-electrode electrical conditions."""
    scales = direct_scales(model.reference_force)
    top = _boundary_fields(model, xy_top, phase=torch.ones_like(xy_top[:, 0:1]))
    bottom = _boundary_fields(
        model, xy_bottom, phase=-torch.ones_like(xy_bottom[:, 0:1]),
    )
    right = _boundary_fields(model, xy_right)
    left = _boundary_fields(model, xy_left)
    target_shear = model.reference_force / HEIGHT
    phi_top = top[0][:, 2:3] / scales.phi
    d_top = top[5] / scales.electric_displacement
    terms = {
        "top_sigma_y": torch.mean((top[2] / scales.stress_y) ** 2),
        "top_tau": torch.mean((top[3] / scales.shear) ** 2),
        "bottom_sigma_y": torch.mean((bottom[2] / scales.stress_y) ** 2),
        "bottom_tau": torch.mean((bottom[3] / scales.shear) ** 2),
        "right_sigma_x": torch.mean((right[1] / scales.stress_x) ** 2),
        "right_tau": torch.mean(
            ((right[3] - target_shear) / scales.traction) ** 2
        ),
        "right_Dx": torch.mean((right[4] / scales.electric_displacement) ** 2),
        "left_Dx": torch.mean((left[4] / scales.electric_displacement) ** 2),
        "top_zero_net_charge": d_top.mean() ** 2,
    }
    if not model.hard_floating_electrode:
        terms["top_equipotential"] = torch.mean(
            (phi_top - phi_top.mean()) ** 2
        )
    total = sum(terms.values())
    return (total, terms) if return_terms else total


def interface_loss_standard(xy_interface, model, *, return_terms=False):
    """Bonding plus traction and normal-electric transmission conditions."""
    epsilon = 1e-5 * HEIGHT
    x = xy_interface[:, 0:1]
    top_xy = torch.cat((x, torch.full_like(x, CENTER + epsilon)), dim=1)
    bottom_xy = torch.cat((x, torch.full_like(x, CENTER - epsilon)), dim=1)
    top = _boundary_fields(model, top_xy, phase=torch.ones_like(x))
    bottom = _boundary_fields(model, bottom_xy, phase=-torch.ones_like(x))
    scales = direct_scales(model.reference_force)
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
                  pde_weight=1.0, bc_weight=10.0, interface_weight=1.0):
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
    terms.update({
        "pde_total": pde,
        "bc_total": bc,
        "interface_total": interface,
        "total": total,
    })
    return total, terms


def train_standard(model, tensors, *, epochs_adam=1000, epochs_lbfgs=20,
                   lr_adam=1e-3, lr_lbfgs=1.0, log_every=100,
                   resample_fn=None, resample_every=100, lbfgs_tensors=None,
                   adam_checkpoints_dir: Path | None = None,
                   lbfgs_checkpoints_dir: Path | None = None):
    """Adam with resampling followed by deterministic strong-Wolfe L-BFGS."""
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
            if adam_checkpoints_dir is not None:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item(),
                }, Path(adam_checkpoints_dir) / f"model_epoch_{epoch}.pt")

    adam_state_dict = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
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
        if lbfgs_checkpoints_dir is not None:
            torch.save(
                model.state_dict(),
                Path(lbfgs_checkpoints_dir) / f"model_epoch_{epoch}.pt",
            )

    _, final_terms = loss_standard(fixed, model, return_terms=True)
    return {
        "loss_list": history,
        "loss_components": component_history,
        "total_time": time.time() - start,
        "adam_state_dict": adam_state_dict,
        "final_terms": {
            name: float(value.detach().cpu())
            for name, value in final_terms.items()
        },
    }
