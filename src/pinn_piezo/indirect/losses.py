"""Physics and boundary losses for the indirect PINN."""

from __future__ import annotations

import torch
from torch import nn


loss_fn = nn.MSELoss()


def physics_loss(x, y, model, coefficients):
    """PDE residual: stress / displacement field / charge density."""
    scale_factor = 1  # noqa: F841 (kept to match the notebook structure)

    x_data = x
    y_data = y
    data = torch.hstack((x_data, y_data))
    y_hat = model(data)

    u_pred = y_hat[:, 0:1]
    v_pred = y_hat[:, 1:2]
    phi_pred = y_hat[:, 2:3]
    sigmax_pred = y_hat[:, 3:4]
    sigmaz_pred = y_hat[:, 4:5]
    tauxz_pred = y_hat[:, 5:6]
    Dx_pred = y_hat[:, 6:7]
    Dy_pred = y_hat[:, 7:8]

    def _g(out, wrt):
        return torch.autograd.grad(outputs=out, inputs=wrt,
                                   grad_outputs=torch.ones_like(out),
                                   create_graph=True, retain_graph=True)[0]

    ux = _g(u_pred, x_data)
    uy = _g(u_pred, y_data)

    vx = _g(v_pred, x_data)
    vy = _g(v_pred, y_data)

    phix = _g(phi_pred, x_data)
    phiy = _g(phi_pred, y_data)

    sigmax_pred_x = _g(sigmax_pred, x_data)
    tauxz_pred_y = _g(tauxz_pred, y_data)
    tauxz_pred_x = _g(tauxz_pred, x_data)
    sigmaz_pred_y = _g(sigmaz_pred, y_data)

    Dx_pred_x = _g(Dx_pred, x_data)
    Dy_pred_y = _g(Dy_pred, y_data)

    epsilon_xx = ux
    epsilon_yy = vy
    epsilon_xy = 0.5 * (uy + vx)

    Ex = -phix
    Ey = -phiy

    C11 = coefficients[:, 0:1]
    C12 = coefficients[:, 1:2]
    C22 = coefficients[:, 2:3]
    G = coefficients[:, 3:4]
    epsilon1 = coefficients[:, 4:5]
    epsilon2 = coefficients[:, 5:6]
    e31 = coefficients[:, 6:7]
    e33 = coefficients[:, 7:8]

    sigmax = (C11 * epsilon_xx + C12 * epsilon_yy - e31 * Ey)
    sigmaz = (C12 * epsilon_xx + C22 * epsilon_yy - e33 * Ey)
    tauxz = G * epsilon_xy

    Dx = epsilon1 * Ex
    Dy = (e31 * epsilon_xx + e33 * epsilon_yy + epsilon2 * Ey)

    divergence_sigma1 = sigmax_pred_x + tauxz_pred_y
    divergence_sigma2 = tauxz_pred_x + sigmaz_pred_y
    divergence_D = Dx_pred_x + Dy_pred_y

    residual_sigmax = sigmax_pred - sigmax
    residual_sigmaz = sigmaz_pred - sigmaz
    residual_tauxz = tauxz_pred - tauxz

    residual_Dx = Dx_pred - Dx
    residual_Dy = Dy_pred - Dy

    loss_mech = (torch.mean(residual_sigmax ** 2)
                 + torch.mean(residual_sigmaz ** 2)
                 + torch.mean(residual_tauxz ** 2))

    loss_elec = torch.mean(residual_Dx ** 2) + torch.mean(residual_Dy ** 2)

    loss_divergence = (torch.mean(divergence_sigma1 ** 2)
                       + torch.mean(divergence_sigma2 ** 2)
                       + torch.mean(divergence_D ** 2))

    return loss_mech + loss_elec + loss_divergence


def stress_BC_loss(xy_top, xy_bottom, xy_right, xy_left, model):
    y_hat_top = model(xy_top)         # noqa: F841 -- kept to mirror the notebook
    y_hat_bottom = model(xy_bottom)   # noqa: F841
    y_hat_right = model(xy_right)

    sigmax_pred_right = y_hat_right[:, 3:4]
    sigmaz_pred_right = y_hat_right[:, 4:5]
    tauxz_pred_right = y_hat_right[:, 5:6]

    n_right = torch.ones_like(xy_right)
    n_right[:, 1] = 0

    traction_x_right = (sigmax_pred_right * n_right[:, 0:1]
                        + tauxz_pred_right * n_right[:, 1:2])
    traction_y_right = (tauxz_pred_right * n_right[:, 0:1]
                        + sigmaz_pred_right * n_right[:, 1:2])

    return torch.mean(traction_x_right ** 2) + torch.mean(traction_y_right ** 2)


def electric_BC_loss(xy_right, xy_left, model):
    y_hat_right = model(xy_right)
    y_hat_left = model(xy_left)

    Dx_pred_right = y_hat_right[:, 6:7]
    Dy_pred_right = y_hat_right[:, 7:8]

    Dx_pred_left = y_hat_left[:, 6:7]
    Dy_pred_left = y_hat_left[:, 7:8]

    n_right = torch.ones_like(xy_right)
    n_right[:, 1] = 0

    n_left = torch.ones_like(xy_left)
    n_left[:, 1] = 0
    n_left[:, 0] = -1

    D_dot_n_right = (Dx_pred_right * n_right[:, 0:1]
                     + Dy_pred_right * n_right[:, 1:2])
    D_dot_n_left = (Dx_pred_left * n_left[:, 0:1]
                    + Dy_pred_left * n_left[:, 1:2])

    return torch.mean(D_dot_n_right ** 2) + torch.mean(D_dot_n_left ** 2)


def get_BC_loss(xy_top, xy_bottom, xy_right, xy_left, model):
    stress_loss_term = stress_BC_loss(xy_top, xy_bottom, xy_right, xy_left, model)
    electric_loss_term = electric_BC_loss(xy_right, xy_left, model)
    return stress_loss_term + electric_loss_term


def update_weights(pde_loss, bc_loss, weights, model):
    lambda_1 = weights['pde']
    lambda_2 = weights['bc']

    grad_pde = torch.autograd.grad(pde_loss, model.parameters(),
                                   retain_graph=True, allow_unused=True)[0]
    grad_bc = torch.autograd.grad(bc_loss, model.parameters(),
                                  retain_graph=True, allow_unused=True)[0]

    gradients = grad_pde.norm() + grad_bc.norm()

    lambda_1_hat = gradients / grad_pde.norm()
    lambda_2_hat = gradients / grad_bc.norm()

    lambda_1 = 0.9 * lambda_1 + (1 - 0.9) * lambda_1_hat
    lambda_2 = 0.9 * lambda_2 + (1 - 0.9) * lambda_2_hat

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {'pde': lambda_1, 'bc': lambda_2}


def loss_func(xy_top, xy_bottom, xy_right, xy_left,
              x_collocation, y_collocation,
              model, coefficients, loss_weights, n, f, adjust=False):
    BC_term = get_BC_loss(xy_top, xy_bottom, xy_right, xy_left, model)
    physics_loss_term = physics_loss(x_collocation, y_collocation, model,
                                     coefficients)

    if n % f == 0 and adjust:
        loss_weights = update_weights(physics_loss_term, BC_term, loss_weights,
                                      model)

    lambda1 = loss_weights['bc']
    lambda3 = loss_weights['pde']

    loss = lambda1 * BC_term + lambda3 * physics_loss_term
    return loss, loss_weights
