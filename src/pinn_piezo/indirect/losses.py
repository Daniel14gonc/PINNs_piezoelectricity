"""Physics and boundary losses for the indirect PINN."""

from __future__ import annotations

import torch
from torch import nn

from .. import config, materials
from ..scaling import (dominant_term_scales, indirect_scales,
                       represented_term_scales)


loss_fn = nn.MSELoss()


def physics_loss(x, y, model, coefficients, scales=None,
                 return_terms: bool = False,
                 normal_constitutive_weight: float = 1.0,
                 constitutive_normalization: str = "represented",
                 constitutive_flux_stopgrad: bool = False,
                 mechanical_constitutive_stopgrad: bool = False):
    """PDE residual: stress / displacement field / charge density."""
    scales = indirect_scales(config.VOLTAGE) if scales is None else scales
    if constitutive_normalization not in ("represented", "dominant"):
        raise ValueError(
            "constitutive_normalization must be 'represented' or 'dominant'"
        )
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
    gamma_xy = uy + vx

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
    tauxz = G * gamma_xy

    Dx = epsilon1 * Ex
    Dy = (e31 * epsilon_xx + e33 * epsilon_yy + epsilon2 * Ey)

    divergence_sigma1 = sigmax_pred_x + tauxz_pred_y
    divergence_sigma2 = tauxz_pred_x + sigmaz_pred_y
    divergence_D = Dx_pred_x + Dy_pred_y

    # Optional one-way gradient routing for the mixed constitutive equations.
    # The residual values and therefore the reported loss are unchanged.
    # Only their backward path through the independent flux outputs is
    # removed, forcing these equations to update (u,v,phi).  Equilibrium and
    # boundary losses below still train every stress/D output normally.
    route_mechanical = (
        constitutive_flux_stopgrad or mechanical_constitutive_stopgrad
    )
    constitutive_sigmax_pred = (
        sigmax_pred.detach() if route_mechanical else sigmax_pred
    )
    constitutive_sigmaz_pred = (
        sigmaz_pred.detach() if route_mechanical else sigmaz_pred
    )
    constitutive_tauxz_pred = (
        tauxz_pred.detach() if route_mechanical else tauxz_pred
    )
    constitutive_Dx_pred = (
        Dx_pred.detach() if constitutive_flux_stopgrad else Dx_pred
    )
    constitutive_Dy_pred = (
        Dy_pred.detach() if constitutive_flux_stopgrad else Dy_pred
    )

    if constitutive_normalization == "dominant":
        constitutive = dominant_term_scales(scales)
    else:
        constitutive = represented_term_scales(
            scales, strict_transverse=True,
        )

    residual_sigmax = (
        constitutive_sigmax_pred - sigmax
    ) / constitutive.sigma_xx
    residual_sigmaz = (
        constitutive_sigmaz_pred - sigmaz
    ) / constitutive.sigma_yy
    residual_tauxz = (
        constitutive_tauxz_pred - tauxz
    ) / constitutive.tau_xy

    residual_Dx = (
        constitutive_Dx_pred - Dx
    ) / constitutive.d_x
    residual_Dy = (
        constitutive_Dy_pred - Dy
    ) / constitutive.d_y

    divergence_sigma1 = divergence_sigma1 / scales.equilibrium_x
    divergence_sigma2 = divergence_sigma2 / scales.equilibrium_y
    divergence_D = divergence_D / scales.gauss

    terms = {
        "constitutive_sigma_xx": torch.mean(residual_sigmax ** 2),
        "constitutive_sigma_yy": torch.mean(residual_sigmaz ** 2),
        "constitutive_tau_xy": torch.mean(residual_tauxz ** 2),
        "constitutive_D_x": torch.mean(residual_Dx ** 2),
        "constitutive_D_y": torch.mean(residual_Dy ** 2),
        "equilibrium_x": torch.mean(divergence_sigma1 ** 2),
        "equilibrium_y": torch.mean(divergence_sigma2 ** 2),
        "gauss": torch.mean(divergence_D ** 2),
    }
    if normal_constitutive_weight <= 0.0:
        raise ValueError("normal_constitutive_weight must be positive")
    total = sum(terms.values()) + (normal_constitutive_weight - 1.0) * (
        terms["constitutive_sigma_xx"] + terms["constitutive_sigma_yy"]
    )
    return (total, terms) if return_terms else total


def diagnostic_loss_terms(
        tensors, model, *, normal_constitutive_weight=1.0,
        constitutive_normalization: str = "represented",
        constitutive_traction_bcs: bool = False,
        constitutive_traction_weight: float = 1.0,
        constitutive_flux_stopgrad: bool = False,
        mechanical_constitutive_stopgrad: bool = False):
    """Return detached, named loss families on an explicitly fixed cloud."""
    total_pde, terms = physics_loss(
        tensors["x_collocation"], tensors["y_collocation"], model,
        tensors["coefficients"], return_terms=True,
        normal_constitutive_weight=normal_constitutive_weight,
        constitutive_normalization=constitutive_normalization,
        constitutive_flux_stopgrad=constitutive_flux_stopgrad,
        mechanical_constitutive_stopgrad=mechanical_constitutive_stopgrad,
    )
    terms = dict(terms)
    terms["boundary"] = get_BC_loss(
        tensors["xy_top"], tensors["xy_bottom"],
        tensors["xy_right"], tensors["xy_left"], model,
    )
    if constitutive_traction_bcs:
        terms["boundary_constitutive_traction"] = (
            constitutive_traction_weight * constitutive_traction_BC_loss(
                tensors["xy_top"], tensors["xy_bottom"],
                tensors["xy_right"], model,
            )
        )
    if tensors.get("xy_interface") is not None:
        terms["interface"] = interface_loss(tensors["xy_interface"], model)
    terms["pde_total"] = total_pde
    terms["total_unweighted"] = sum(
        value for name, value in terms.items() if name != "pde_total"
    )
    return {name: float(value.detach().cpu()) for name, value in terms.items()}


def stress_BC_loss(xy_top, xy_bottom, xy_right, xy_left, model, scales=None):
    scales = indirect_scales(config.VOLTAGE) if scales is None else scales
    y_hat_top = model(xy_top)
    y_hat_bottom = model(xy_bottom)
    y_hat_right = model(xy_right)

    sigmaz_pred_top = y_hat_top[:, 4:5]
    tauxz_pred_top = y_hat_top[:, 5:6]
    sigmaz_pred_bottom = y_hat_bottom[:, 4:5]
    tauxz_pred_bottom = y_hat_bottom[:, 5:6]
    sigmax_pred_right = y_hat_right[:, 3:4]
    tauxz_pred_right = y_hat_right[:, 5:6]

    # Horizontal free faces have n=(0,+/-1), hence t=(tau_xy,sigma_yy).
    # The free right face has n=(1,0), hence t=(sigma_xx,tau_xy).
    return (torch.mean((sigmaz_pred_top / scales.stress_y) ** 2)
            + torch.mean((tauxz_pred_top / scales.shear) ** 2)
            + torch.mean((sigmaz_pred_bottom / scales.stress_y) ** 2)
            + torch.mean((tauxz_pred_bottom / scales.shear) ** 2)
            + torch.mean((sigmax_pred_right / scales.stress_x) ** 2)
            + torch.mean((tauxz_pred_right / scales.shear) ** 2))


def constitutive_traction_BC_loss(
        xy_top, xy_bottom, xy_right, model, scales=None):
    """Apply the free-face tractions directly through ``(u, v, phi)``.

    In the exact mixed solution this is redundant with the boundary
    conditions on the independent stress outputs and the interior
    constitutive equations.  Numerically it prevents those independent
    outputs from satisfying every homogeneous mechanical boundary condition
    while the primal fields drift toward the null or mirrored bending branch.
    No beam kinematics, resultant, energy, or external data enters this term.
    """
    scales = indirect_scales(config.VOLTAGE) if scales is None else scales

    def constitutive_stress(xy):
        differentiable_xy = xy
        if not differentiable_xy.requires_grad:
            differentiable_xy = xy.detach().requires_grad_(True)
        primary = model(differentiable_xy)[:, :3]
        u, v, phi = (
            primary[:, 0:1], primary[:, 1:2], primary[:, 2:3],
        )

        def gradient(field):
            return torch.autograd.grad(
                outputs=field,
                inputs=differentiable_xy,
                grad_outputs=torch.ones_like(field),
                create_graph=True,
                retain_graph=True,
            )[0]

        grad_u = gradient(u)
        grad_v = gradient(v)
        grad_phi = gradient(phi)
        epsilon_xx = grad_u[:, 0:1]
        epsilon_yy = grad_v[:, 1:2]
        gamma_xy = grad_u[:, 1:2] + grad_v[:, 0:1]
        phi_y = grad_phi[:, 1:2]

        def coefficient(value):
            return torch.as_tensor(
                value, dtype=xy.dtype, device=xy.device,
            )

        top_layer = differentiable_xy[:, 1:2] >= config.CENTER
        e31 = torch.where(
            top_layer,
            coefficient(materials.e31_top),
            coefficient(materials.e31_bottom),
        )
        e33 = torch.where(
            top_layer,
            coefficient(materials.e33_top),
            coefficient(materials.e33_bottom),
        )
        sigma_xx = (
            coefficient(materials.C11) * epsilon_xx
            + coefficient(materials.C12) * epsilon_yy
            + e31 * phi_y
        )
        sigma_yy = (
            coefficient(materials.C12) * epsilon_xx
            + coefficient(materials.C22) * epsilon_yy
            + e33 * phi_y
        )
        tau_xy = coefficient(materials.G) * gamma_xy
        return torch.cat((sigma_xx, sigma_yy, tau_xy), dim=1)

    top_stress = constitutive_stress(xy_top)
    bottom_stress = constitutive_stress(xy_bottom)
    right_stress = constitutive_stress(xy_right)
    return (
        torch.mean((top_stress[:, 1:2] / scales.stress_y) ** 2)
        + torch.mean((top_stress[:, 2:3] / scales.shear) ** 2)
        + torch.mean((bottom_stress[:, 1:2] / scales.stress_y) ** 2)
        + torch.mean((bottom_stress[:, 2:3] / scales.shear) ** 2)
        + torch.mean((right_stress[:, 0:1] / scales.stress_x) ** 2)
        + torch.mean((right_stress[:, 2:3] / scales.shear) ** 2)
    )


def electric_BC_loss(xy_right, xy_left, model, scales=None):
    scales = indirect_scales(config.VOLTAGE) if scales is None else scales
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

    return (torch.mean((D_dot_n_right / scales.electric_displacement) ** 2)
            + torch.mean((D_dot_n_left / scales.electric_displacement) ** 2))


def get_BC_loss(xy_top, xy_bottom, xy_right, xy_left, model):
    scales = indirect_scales(config.VOLTAGE)
    stress_loss_term = stress_BC_loss(
        xy_top, xy_bottom, xy_right, xy_left, model, scales=scales,
    )
    electric_loss_term = electric_BC_loss(
        xy_right, xy_left, model, scales=scales,
    )
    return stress_loss_term + electric_loss_term


def interface_loss(xy_interface, model, scales=None):
    """Enforce only the six transmission conditions of a bonded interface.

    The phase-enriched *single* network has two traces at ``y=H/2``.  Bonding
    and Maxwell transmission require continuity of ``u``, ``v``, ``phi``,
    ``sigma_yy``, ``tau_xy`` and ``D_y``.  ``sigma_xx`` and ``D_x`` are
    deliberately absent: neither is a normal flux across this horizontal
    interface and both may jump when the material polarization changes.
    """
    if not hasattr(model, "forward_trace"):
        raise TypeError("interface_loss requires a phase-enriched model")
    scales = indirect_scales(config.VOLTAGE) if scales is None else scales
    top_phase = torch.ones_like(xy_interface[:, 0:1])
    bottom_phase = -top_phase
    top = model.forward_trace(xy_interface, top_phase)
    bottom = model.forward_trace(xy_interface, bottom_phase)

    field_scales = (
        scales.u,
        scales.v,
        scales.phi,
        scales.stress_y,
        scales.shear,
        scales.electric_displacement,
    )
    continuous_fields = (0, 1, 2, 4, 5, 7)
    return sum(
        torch.mean(((top[:, i:i + 1] - bottom[:, i:i + 1]) / scale) ** 2)
        for i, scale in zip(continuous_fields, field_scales)
    )


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
              model, coefficients, loss_weights, n, f, adjust=False,
              xy_interface=None, interface_weight: float = 1.0,
              normal_constitutive_weight: float = 1.0,
              constitutive_normalization: str = "represented",
              constitutive_traction_bcs: bool = False,
              constitutive_traction_weight: float = 1.0,
              constitutive_flux_stopgrad: bool = False,
              mechanical_constitutive_stopgrad: bool = False,
              return_components: bool = False):
    if constitutive_traction_weight < 0.0:
        raise ValueError("constitutive_traction_weight must be non-negative")
    BC_term = get_BC_loss(xy_top, xy_bottom, xy_right, xy_left, model)
    if constitutive_traction_bcs:
        BC_term = BC_term + (
            constitutive_traction_weight * constitutive_traction_BC_loss(
                xy_top, xy_bottom, xy_right, model,
                scales=indirect_scales(config.VOLTAGE),
            )
        )
    physics_result = physics_loss(
        x_collocation, y_collocation, model, coefficients,
        scales=indirect_scales(config.VOLTAGE),
        return_terms=return_components,
        normal_constitutive_weight=normal_constitutive_weight,
        constitutive_normalization=constitutive_normalization,
        constitutive_flux_stopgrad=constitutive_flux_stopgrad,
        mechanical_constitutive_stopgrad=mechanical_constitutive_stopgrad,
    )
    if return_components:
        physics_loss_term, equation_terms = physics_result
    else:
        physics_loss_term = physics_result

    if n % f == 0 and adjust:
        loss_weights = update_weights(physics_loss_term, BC_term, loss_weights,
                                      model)

    lambda1 = loss_weights['bc']
    lambda3 = loss_weights['pde']

    weighted_pde = lambda3 * physics_loss_term
    weighted_boundary = lambda1 * BC_term
    loss = weighted_boundary + weighted_pde
    interface_loss_term = torch.zeros_like(physics_loss_term)
    if xy_interface is not None:
        interface_loss_term = interface_weight * interface_loss(
            xy_interface, model, scales=indirect_scales(config.VOLTAGE),
        )
        loss = loss + interface_loss_term
    if return_components:
        constitutive = sum(
            term for name, term in equation_terms.items()
            if name.startswith("constitutive_")
        )
        balance = sum(
            equation_terms[name]
            for name in ("equilibrium_x", "equilibrium_y", "gauss")
        )
        return loss, loss_weights, {
            "constitutive": lambda3 * constitutive,
            "balance": lambda3 * balance,
            "pde": weighted_pde,
            "boundary": weighted_boundary,
            "interface": interface_loss_term,
            "total": loss,
        }
    return loss, loss_weights
