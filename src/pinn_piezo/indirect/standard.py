"""Conventional ("Case A") PINN baseline for the architecture ablation.

The paper's contribution (Cluster 1/3) is the *explicit-output* network:
stress ``(sigma_x, sigma_z, tau_xz)`` and electric displacement
``(D_x, D_y)`` are extra network outputs, so the governing equations are
enforced with **first-order** derivatives only (see
:mod:`pinn_piezo.indirect.losses`). Reviewers asked us to show this is
actually better than the conventional formulation.

This module implements that conventional baseline:

* outputs are only ``(u, v, phi)`` (``output_size = 3``);
* stresses / electric displacement are reconstructed from the
  constitutive law and the network derivatives;
* equilibrium ``div(sigma) = 0`` and Gauss ``div(D) = 0`` are enforced
  directly, which requires **second-order** derivatives of ``(u, v, phi)``.

Everything else (hard constraints on the clamp and the electrodes, the
material coefficients, the boundary terms) mirrors the explicit model so
the comparison is apples-to-apples. The two formulations are then trained
identically (same data, optimiser schedule, weights) and compared against
the FEM/analytical reference in the ablation notebook.
"""

from __future__ import annotations

import time
from collections import OrderedDict

import torch
from torch import nn

from .losses import loss_fn  # noqa: F401  (kept for parity / convenience)
from .model import init_weights, phi_constraint
from ..config import HEIGHT


class StandardFCNPyramid(nn.Module):
    """Same trunk + hard constraints as ``FCNPyramid`` but only (u, v, phi)."""

    def __init__(self, input_size, hidden_sizes, output_size=3,
                 activation=nn.Tanh):
        super().__init__()
        layers = [
            ('input', nn.Linear(input_size, hidden_sizes[0])),
            ('act0', activation()),
        ]
        for i in range(1, len(hidden_sizes)):
            layers.append((f'hidden_{i - 1}',
                           nn.Linear(hidden_sizes[i - 1], hidden_sizes[i])))
            layers.append((f'act_{i}', activation()))
        layers.append(('output', nn.Linear(hidden_sizes[-1], output_size)))
        self.net = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        outputs = self.net(x)
        u, v, phi = outputs[:, 0:1], outputs[:, 1:2], outputs[:, 2:3]

        u_modified = x[:, 0:1] * u
        v_modified = x[:, 0:1] * v
        phi_modified = (x[:, 1:2] * (x[:, 1:2] - HEIGHT) * phi
                        + phi_constraint(x[:, 0:1], x[:, 1:2]))
        return torch.cat([u_modified, v_modified, phi_modified], dim=1)


def build_standard_model(device=None, input_size=2, hidden_sizes=(100, 250)):
    model = StandardFCNPyramid(input_size, list(hidden_sizes), output_size=3)
    model.apply(init_weights)
    if device is not None:
        model.to(device)
    return model


def _grad(out, wrt):
    return torch.autograd.grad(out, wrt, grad_outputs=torch.ones_like(out),
                               create_graph=True, retain_graph=True)[0]


def _constitutive(x_data, y_data, model, coefficients):
    """Return stresses / electric displacement from (u, v, phi) derivatives."""
    data = torch.hstack((x_data, y_data))
    y_hat = model(data)
    u_pred, v_pred, phi_pred = y_hat[:, 0:1], y_hat[:, 1:2], y_hat[:, 2:3]

    ux, uy = _grad(u_pred, x_data), _grad(u_pred, y_data)
    vx, vy = _grad(v_pred, x_data), _grad(v_pred, y_data)
    phix, phiy = _grad(phi_pred, x_data), _grad(phi_pred, y_data)

    epsilon_xx = ux
    epsilon_yy = vy
    epsilon_xy = 0.5 * (uy + vx)
    Ex, Ey = -phix, -phiy

    C11 = coefficients[:, 0:1]
    C12 = coefficients[:, 1:2]
    C22 = coefficients[:, 2:3]
    G = coefficients[:, 3:4]
    epsilon1 = coefficients[:, 4:5]
    epsilon2 = coefficients[:, 5:6]
    e31 = coefficients[:, 6:7]
    e33 = coefficients[:, 7:8]

    sigmax = C11 * epsilon_xx + C12 * epsilon_yy - e31 * Ey
    sigmaz = C12 * epsilon_xx + C22 * epsilon_yy - e33 * Ey
    tauxz = G * epsilon_xy
    Dx = epsilon1 * Ex
    Dy = e31 * epsilon_xx + e33 * epsilon_yy + epsilon2 * Ey
    return sigmax, sigmaz, tauxz, Dx, Dy


def physics_loss_standard(x, y, model, coefficients):
    """Equilibrium + Gauss residuals using SECOND-order derivatives."""
    x_data, y_data = x, y
    sigmax, sigmaz, tauxz, Dx, Dy = _constitutive(x_data, y_data, model,
                                                  coefficients)

    sigmax_x = _grad(sigmax, x_data)
    tauxz_y = _grad(tauxz, y_data)
    tauxz_x = _grad(tauxz, x_data)
    sigmaz_y = _grad(sigmaz, y_data)
    Dx_x = _grad(Dx, x_data)
    Dy_y = _grad(Dy, y_data)

    div_sigma1 = sigmax_x + tauxz_y
    div_sigma2 = tauxz_x + sigmaz_y
    div_D = Dx_x + Dy_y

    return (torch.mean(div_sigma1 ** 2)
            + torch.mean(div_sigma2 ** 2)
            + torch.mean(div_D ** 2))


def _boundary_stress_D(model, xy, coeff):
    """Stresses / D at boundary points, reconstructed from derivatives.

    ``coeff`` is an ``(N, 8)`` tensor of per-point material constants
    (built like the collocation coefficients, so the bimorph sign flip of
    ``e31``/``e33`` across the mid-plane is respected).
    """
    xb = xy[:, 0:1].clone().detach().requires_grad_(True)
    yb = xy[:, 1:2].clone().detach().requires_grad_(True)
    return _constitutive(xb, yb, model, coeff)


def bc_loss_standard(xy_right, coeff_right, xy_left, coeff_left, model):
    """Traction-free right edge + charge-free right/left edges (Neumann)."""
    sxx_r, szz_r, txz_r, Dx_r, Dy_r = _boundary_stress_D(model, xy_right,
                                                         coeff_right)
    _, _, _, Dx_l, Dy_l = _boundary_stress_D(model, xy_left, coeff_left)

    # right edge outward normal = (1, 0)
    traction_x_right = sxx_r
    traction_y_right = txz_r
    stress_term = (torch.mean(traction_x_right ** 2)
                   + torch.mean(traction_y_right ** 2))

    D_dot_n_right = Dx_r
    D_dot_n_left = -Dx_l
    elec_term = torch.mean(D_dot_n_right ** 2) + torch.mean(D_dot_n_left ** 2)
    return stress_term + elec_term


def train_standard(model, tensors, *,
                   epochs_adam=1000, epochs_lbfgs=200,
                   lr_adam=0.001, lr_lbfgs=0.01,
                   log_every=100, record_components=False):
    """Adam + L-BFGS training of the conventional baseline.

    Mirrors :func:`pinn_piezo.indirect.train.train` (fixed loss weights of
    1) so the comparison with the explicit model is fair. ``tensors`` must
    additionally provide ``coeff_right`` / ``coeff_left`` (per-point
    boundary coefficients). Returns a dict with the loss history and
    wall-clock time; if ``record_components`` is set it also returns
    per-epoch PDE / BC histories (for loss curves).
    """
    xy_right = tensors["xy_right"]
    xy_left = tensors["xy_left"]
    coeff_right = tensors["coeff_right"]
    coeff_left = tensors["coeff_left"]
    xcol = tensors["x_collocation"]
    ycol = tensors["y_collocation"]
    coeff = tensors["coefficients"]

    def total_loss():
        pde = physics_loss_standard(xcol, ycol, model, coeff)
        bc = bc_loss_standard(xy_right, coeff_right, xy_left, coeff_left, model)
        return pde + bc, pde, bc

    history, pde_hist, bc_hist = [], [], []
    start = time.time()

    opt = torch.optim.Adam(model.parameters(), lr=lr_adam)
    for epoch in range(epochs_adam):
        opt.zero_grad()
        loss, pde, bc = total_loss()
        loss.backward()
        opt.step()
        history.append(loss.item())
        if record_components:
            pde_hist.append(pde.item())
            bc_hist.append(bc.item())
        if epoch % log_every == 0:
            print(f"[ADAM] {epoch}/{epochs_adam} loss={loss.item():.4e}")

    opt = torch.optim.LBFGS(model.parameters(), lr=lr_lbfgs)
    for epoch in range(epochs_lbfgs):
        def closure():
            opt.zero_grad()
            loss, _, _ = total_loss()
            loss.backward()
            return loss
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        opt.step(closure)
        loss, pde, bc = total_loss()
        history.append(loss.item())
        if record_components:
            pde_hist.append(pde.item())
            bc_hist.append(bc.item())
        if epoch % log_every == 0:
            print(f"[LBFGS] {epoch}/{epochs_lbfgs} loss={loss.item():.4e}")

    out = {"loss_list": history, "total_time": time.time() - start}
    if record_components:
        out["pde_list"] = pde_hist
        out["bc_list"] = bc_hist
    return out
