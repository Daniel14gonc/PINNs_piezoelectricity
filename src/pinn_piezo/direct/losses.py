"""Physics and boundary losses for the direct (force-driven) PINN."""

from __future__ import annotations

import torch
from torch import nn

from ..config import HEIGHT


loss_fn = nn.MSELoss()


def physics_loss(x, y, model, coefficients):
    """PDE residual built from a column-wise Jacobian of the network output."""
    x_data = x
    y_data = y
    data = torch.hstack((x_data, y_data))
    y_hat = model(data)

    u_pred = y_hat[:, 0:1]            # noqa: F841
    v_pred = y_hat[:, 1:2]            # noqa: F841
    phi_pred = y_hat[:, 2:3]          # noqa: F841
    sigmax_pred = y_hat[:, 3:4]
    sigmaz_pred = y_hat[:, 4:5]
    tauxz_pred = y_hat[:, 5:6]
    Dx_pred = y_hat[:, 6:7]
    Dy_pred = y_hat[:, 7:8]

    all_grads = [
        torch.autograd.grad(y_hat[:, i].sum(), data, create_graph=True)[0]
        for i in range(y_hat.shape[1])
    ]

    ux = all_grads[0][:, 0:1]
    uy = all_grads[0][:, 1:2]

    vx = all_grads[1][:, 0:1]
    vy = all_grads[1][:, 1:2]

    phix = all_grads[2][:, 0:1]
    phiy = all_grads[2][:, 1:2]

    sigmax_pred_x = all_grads[3][:, 0:1]
    sigmaz_pred_y = all_grads[4][:, 1:2]

    tauxz_pred_x = all_grads[5][:, 0:1]
    tauxz_pred_y = all_grads[5][:, 1:2]

    Dx_pred_x = all_grads[6][:, 0:1]
    Dy_pred_y = all_grads[7][:, 1:2]

    epsilon_xx = ux
    epsilon_yy = vy
    epsilon_xy = (uy + vx)

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

    w_mech = 1
    w_elec = 1
    w_div = 1

    total = (w_mech * loss_mech + w_elec * loss_elec + w_div * loss_divergence)

    return (total, residual_sigmax, residual_sigmaz, residual_tauxz,
            residual_Dx, residual_Dy,
            divergence_sigma1, divergence_sigma2, divergence_D)


def traction_BC_loss(xy_right, model, applied_force_y):
    """Apply a uniformly distributed traction on the right end of the beam."""
    y_hat_right = model(xy_right)

    sigmax_pred_right = y_hat_right[:, 3:4]
    sigmaz_pred_right = y_hat_right[:, 4:5]
    tauxz_pred_right = y_hat_right[:, 5:6]

    n_x = 1.0
    n_y = 0.0

    traction_x = sigmax_pred_right * n_x + tauxz_pred_right * n_y
    traction_y = tauxz_pred_right * n_x + sigmaz_pred_right * n_y

    target_traction_y = applied_force_y / HEIGHT

    loss_traction_x = torch.mean(traction_x ** 2)
    loss_traction_y = torch.mean((traction_y - target_traction_y) ** 2)

    return loss_traction_x + loss_traction_y


def stress_BC_loss(xy_top, xy_bottom, xy_right, xy_left, model):
    y_hat_top = model(xy_top)
    y_hat_bottom = model(xy_bottom)

    sigmax_pred_top = y_hat_top[:, 3:4]
    tauxz_pred_top = y_hat_top[:, 5:6]

    sigmax_pred_bottom = y_hat_bottom[:, 3:4]
    tauxz_pred_bottom = y_hat_bottom[:, 5:6]

    loss_top = (torch.mean(sigmax_pred_top ** 2)
                + torch.mean(tauxz_pred_top ** 2))
    loss_bottom = (torch.mean(sigmax_pred_bottom ** 2)
                   + torch.mean(tauxz_pred_bottom ** 2))

    loss_right = traction_BC_loss(xy_right, model, applied_force_y=0.1)

    return loss_top + loss_right + loss_bottom


def electric_BC_loss(xy_right, xy_left, xy_top, model):
    y_hat_right = model(xy_right)
    y_hat_left = model(xy_left)
    y_hat_top = model(xy_top)

    Dx_pred_right = y_hat_right[:, 6:7]
    Dy_pred_right = y_hat_right[:, 7:8]

    Dx_pred_left = y_hat_left[:, 6:7]
    Dy_pred_left = y_hat_left[:, 7:8]

    Dx_pred_top = y_hat_top[:, 6:7]
    Dy_pred_top = y_hat_top[:, 7:8]

    n_right = torch.ones_like(xy_right)
    n_right[:, 1] = 0

    n_left = torch.ones_like(xy_left)
    n_left[:, 0] = -1
    n_left[:, 1] = 0

    n_top = torch.ones_like(xy_top)
    n_top[:, 0] = 0
    n_top[:, 1] = 1

    D_dot_n_right = (Dx_pred_right * n_right[:, 0:1]
                     + Dy_pred_right * n_right[:, 1:2])
    D_dot_n_left = (Dx_pred_left * n_left[:, 0:1]
                    + Dy_pred_left * n_left[:, 1:2])
    D_dot_n_top = (Dx_pred_top * n_top[:, 0:1]
                   + Dy_pred_top * n_top[:, 1:2])

    return (torch.mean(D_dot_n_right ** 2)
            + torch.mean(D_dot_n_left ** 2)
            + torch.mean(D_dot_n_top ** 2))


def electric_potential_BC_loss(xy_bottom, model):
    y_hat_bottom = model(xy_bottom)
    potential_pred = y_hat_bottom[:, 2:3]
    return torch.mean(potential_pred ** 2)


def displacement_BC_loss(xy_left, model):
    y_hat_left = model(xy_left)
    u = y_hat_left[:, 0:1]
    v = y_hat_left[:, 1:2]
    return torch.mean(u ** 2) + torch.mean(v ** 2)


def update_weights(pde_loss, bc_loss, weights, model):
    lambda_1 = weights['pde']
    lambda_2 = weights['bc']

    all_grad_pde = torch.autograd.grad(pde_loss, model.parameters(),
                                       retain_graph=True, allow_unused=True)
    grad_pde_vec = torch.cat([g.view(-1) for g in all_grad_pde if g is not None])
    norm_pde = torch.linalg.norm(grad_pde_vec)

    all_grad_bc = torch.autograd.grad(bc_loss, model.parameters(),
                                      retain_graph=True, allow_unused=True)
    grad_bc_vec = torch.cat([g.view(-1) for g in all_grad_bc if g is not None])
    norm_bc = torch.linalg.norm(grad_bc_vec)

    gradients = norm_pde + norm_bc

    lambda_1_hat = gradients / (norm_pde + 1e-12)
    lambda_2_hat = gradients / (norm_bc + 1e-12)

    alpha = 0.9
    lambda_1 = alpha * lambda_1 + (1 - alpha) * lambda_1_hat
    lambda_2 = alpha * lambda_2 + (1 - alpha) * lambda_2_hat

    return {'pde': lambda_1, 'bc': lambda_2}


def get_BC_loss(xy_top, xy_bottom, xy_right, xy_left, model):
    stress_loss_term = stress_BC_loss(xy_top, xy_bottom, xy_right, xy_left, model)
    electric_loss_term = electric_BC_loss(xy_right, xy_left, xy_top, model)
    return (stress_loss_term + electric_loss_term,
            stress_loss_term, electric_loss_term)


def loss_func(xy_top, xy_bottom, xy_right, xy_left,
              x_collocation, y_collocation,
              model, coefficients, loss_weights, n, f,
              only_BCs: bool = False, adjust: bool = False):
    BC_term, stress_loss_term, electric_loss_term = get_BC_loss(
        xy_top, xy_bottom, xy_right, xy_left, model,
    )
    (physics_loss_term, residual_sigmax, residual_sigmaz, residual_tauxz,
     residual_Dx, residual_Dy,
     divergence_sigma1, divergence_sigma2, divergence_D) = physics_loss(
        x_collocation, y_collocation, model, coefficients,
    )

    if only_BCs:
        physics_loss_term = 0

    if n % f == 0 and adjust:
        loss_weights = update_weights(physics_loss_term, BC_term, loss_weights,
                                      model)

    lambda1 = loss_weights['bc']
    lambda3 = loss_weights['pde']

    loss = lambda1 * BC_term + lambda3 * physics_loss_term
    return loss, loss_weights
