"""Physics and boundary losses for the direct (force-driven) PINN."""

from __future__ import annotations

import math

import torch
from torch import nn

from .. import materials
from ..config import CENTER, HEIGHT, REFERENCE_FORCE
from ..scaling import (ConstitutiveScales, direct_scales,
                       dominant_term_scales, represented_term_scales)


loss_fn = nn.MSELoss()

# Tip traction resultant (N) applied on the right edge. Module-level so the
# generalization study (Cluster 8) can sweep the load by setting
# ``pinn_piezo.direct.losses.APPLIED_FORCE_Y`` before training.
APPLIED_FORCE_Y = REFERENCE_FORCE

# Canonical paper device: grounded lower electrode and conducting upper
# electrode in open circuit.  ``insulated`` remains available as a controlled
# bare-dielectric comparison.
DIRECT_ELECTRICAL_BC = "floating_electrode"
DIRECT_TRACTION_PROFILE = "uniform"
DIRECT_POINT_LOAD_WIDTH_RATIO = 1.0 / 16.0


def point_load_traction(y, applied_force_y, *, width_ratio=None):
    """Return a normalized half-Gaussian approximation to a tip Dirac load.

    The load is centred at the upper-right corner ``y=HEIGHT``.  Its integral
    over the complete right face is exactly ``applied_force_y``.  Reducing
    ``width_ratio`` approaches the boundary Dirac distribution without adding
    an artificial moment condition or an interior stress lift.
    """
    if width_ratio is None:
        width_ratio = DIRECT_POINT_LOAD_WIDTH_RATIO
    if not 0.0 < width_ratio <= 1.0:
        raise ValueError("point-load width ratio must lie in (0, 1]")
    width = float(width_ratio) * HEIGHT
    normalization = (
        width * math.sqrt(math.pi / 2.0)
        * math.erf(HEIGHT / (math.sqrt(2.0) * width))
    )
    return (
        applied_force_y
        * torch.exp(-0.5 * ((HEIGHT - y) / width) ** 2)
        / normalization
    )


def physics_loss(x, y, model, coefficients, scales=None, *,
                 include_shear_constitutive: bool = True,
                 shear_constitutive_mode: str = "full",
                 constitutive_weight: float = 1.0,
                 normal_constitutive_weight: float = 1.0,
                 strict_transverse_constitutive: bool = False,
                 constitutive_normalization: str = "represented",
                 shear_rotation_stopgrad: bool = False,
                 normalize_residuals: bool = True):
    """PDE residual built from a column-wise Jacobian of the network output.

    ``shear_rotation_stopgrad`` detaches ``u_y`` inside the shear constitutive
    residual only.  The converged residual is unchanged -- it is still
    ``tau - G (u_y + v_x)`` -- but the optimizer no longer back-propagates the
    shear penalty into ``u``.  This removes the locking brake (the ~(2.1e4)^2
    curvature that forbids ``u`` from developing a through-thickness rotation
    until ``v`` follows in lockstep) while keeping the same equation as the
    one-way, well-conditioned update ``v_x <- tau/G - u_y``: the gradient
    analogue of solving the kinematic chain sequentially.
    """
    scales = direct_scales(APPLIED_FORCE_Y) if scales is None else scales
    if shear_constitutive_mode not in ("full", "midplane"):
        raise ValueError("shear_constitutive_mode must be 'full' or 'midplane'")
    if constitutive_normalization not in ("represented", "dominant"):
        raise ValueError(
            "constitutive_normalization must be 'represented' or 'dominant'"
        )
    if constitutive_weight <= 0.0:
        raise ValueError("constitutive_weight must be positive")
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
    if shear_rotation_stopgrad:
        tauxz = G * (uy.detach() + vx)
    else:
        tauxz = G * epsilon_xy

    Dx = epsilon1 * Ex
    Dy = (e31 * epsilon_xx + e33 * epsilon_yy + epsilon2 * Ey)

    divergence_sigma1 = sigmax_pred_x + tauxz_pred_y
    divergence_sigma2 = tauxz_pred_x + sigmaz_pred_y
    divergence_D = Dx_pred_x + Dy_pred_y

    equilibrium_x_scale = scales.equilibrium_x if normalize_residuals else 1.0
    equilibrium_y_scale = scales.equilibrium_y if normalize_residuals else 1.0
    gauss_scale = scales.gauss if normalize_residuals else 1.0

    # Choice of constitutive normalization.  ``represented`` divides each
    # residual by the scale of the quantity it represents, which is the
    # historical behavior; for a slender beam it over-weights the shear law by
    # ~2.1e4 and the strict transverse law by ~174, because in both cases the
    # individual terms are far larger than their difference.  ``dominant``
    # divides by the largest term of each equation instead, giving every family
    # unit sensitivity to relative representation error.  See
    # ``pinn_piezo.scaling.dominant_term_scales``.
    if not normalize_residuals:
        constitutive = ConstitutiveScales(1.0, 1.0, 1.0, 1.0, 1.0)
    elif constitutive_normalization == "dominant":
        constitutive = dominant_term_scales(scales)
    else:
        constitutive = represented_term_scales(
            scales, strict_transverse=strict_transverse_constitutive,
        )

    residual_sigmax = (sigmax_pred - sigmax) / constitutive.sigma_xx
    residual_sigmaz = (sigmaz_pred - sigmaz) / constitutive.sigma_yy
    residual_tauxz = (tauxz_pred - tauxz) / constitutive.tau_xy

    residual_Dx = (Dx_pred - Dx) / constitutive.d_x
    residual_Dy = (Dy_pred - Dy) / constitutive.d_y

    divergence_sigma1 = divergence_sigma1 / equilibrium_x_scale
    divergence_sigma2 = divergence_sigma2 / equilibrium_y_scale
    divergence_D = divergence_D / gauss_scale

    loss_mech = normal_constitutive_weight * (
        torch.mean(residual_sigmax ** 2)
        + torch.mean(residual_sigmaz ** 2)
    )
    if include_shear_constitutive:
        if shear_constitutive_mode == "full":
            shear_loss = torch.mean(residual_tauxz ** 2)
        else:
            # Selective (one-line) integration of the locking-prone shear
            # relation.  It still ties v_x to u_y and tau, but avoids forcing a
            # four-order cancellation independently at every thickness point.
            mid_data = torch.cat(
                (x_data, torch.full_like(y_data, CENTER)), dim=1,
            )
            mid_hat = model(mid_data)
            mid_u_grad = torch.autograd.grad(
                mid_hat[:, 0].sum(), mid_data, create_graph=True,
            )[0]
            mid_v_grad = torch.autograd.grad(
                mid_hat[:, 1].sum(), mid_data, create_graph=True,
            )[0]
            gamma_mid = mid_u_grad[:, 1:2] + mid_v_grad[:, 0:1]
            tau_mid = mid_hat[:, 5:6]
            shear_loss = torch.mean(
                ((tau_mid - coefficients[:, 3:4] * gamma_mid)
                 / constitutive.tau_xy) ** 2
            )
        loss_mech = loss_mech + shear_loss

    loss_elec = torch.mean(residual_Dx ** 2) + torch.mean(residual_Dy ** 2)

    loss_divergence = (torch.mean(divergence_sigma1 ** 2)
                       + torch.mean(divergence_sigma2 ** 2)
                       + torch.mean(divergence_D ** 2))

    w_mech = 1
    w_elec = 1
    w_div = 1

    total = (
        constitutive_weight * (w_mech * loss_mech + w_elec * loss_elec)
        + w_div * loss_divergence
    )

    return (total, residual_sigmax, residual_sigmaz, residual_tauxz,
            residual_Dx, residual_Dy,
            divergence_sigma1, divergence_sigma2, divergence_D)


def section_equilibrium_loss(model, applied_force_y, *, n_stations: int = 128,
                             n_thickness: int = 24, scales=None,
                             generator=None):
    """Thickness-integrated form of the two equilibrium equations.

    Integrating ``sigma_xx,x + tau_xy,y = 0`` across the thickness weighted by
    ``(y - H/2)``, and ``tau_xy,x + sigma_yy,y = 0`` unweighted, and using the
    traction-free horizontal faces, gives

        dM/dx - Q = 0        with  M(x) = \\int sigma_xx (y-H/2) dy
        dQ/dx     = 0        with  Q(x) = \\int tau_xy dy

    These are the same two equations already in ``physics_loss``, only in their
    section-resultant form, so they are redundant for the exact solution and
    cannot change it.  Both targets are **zero**: no known moment or shear
    distribution is injected.

    They matter numerically because the pointwise equilibrium residual is
    homogeneous -- any small field satisfies it -- so a field that decays away
    from the loaded end can satisfy it almost exactly while absorbing the tip
    traction in a thin boundary layer.  That shortcut leaves the beam unstressed
    and the load never reaches the clamp.  ``dQ/dx = 0`` is a statement about
    how the resultant varies along ``x`` and is precisely what a thin layer
    cannot fake; combined with the tip traction giving ``Q(L) = F`` it forces
    ``Q`` constant along the beam and ``M`` linear.
    """
    scales = direct_scales(applied_force_y) if scales is None else scales
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    stations = torch.rand(
        (n_stations, 1), device=device, dtype=dtype, generator=generator,
    ) * scales.length
    stations = stations.detach().requires_grad_(True)

    # Composite trapezoid across the full thickness, including both faces.
    nodes = torch.linspace(0.0, HEIGHT, n_thickness, device=device, dtype=dtype)
    weights = torch.full_like(nodes, HEIGHT / (n_thickness - 1))
    weights[0] = weights[0] * 0.5
    weights[-1] = weights[-1] * 0.5

    x_flat = stations.repeat_interleave(n_thickness, dim=0)
    y_flat = nodes.repeat(n_stations).unsqueeze(1)
    prediction = model(torch.cat((x_flat, y_flat), dim=1))

    sigma_x = prediction[:, 3].reshape(n_stations, n_thickness)
    tau = prediction[:, 5].reshape(n_stations, n_thickness)
    lever = (nodes - CENTER).unsqueeze(0)

    moment = (sigma_x * lever * weights).sum(dim=1)
    shear = (tau * weights).sum(dim=1)

    d_moment = torch.autograd.grad(
        moment.sum(), stations, create_graph=True, retain_graph=True,
    )[0][:, 0]
    d_shear = torch.autograd.grad(
        shear.sum(), stations, create_graph=True, retain_graph=True,
    )[0][:, 0]

    # Q ~ the applied resultant; dQ/dx ~ that resultant over the beam length.
    shear_scale = abs(float(applied_force_y))
    shear_scale = shear_scale if shear_scale > 0.0 else 1.0
    residual_moment = (d_moment - shear) / shear_scale
    residual_shear = d_shear / (shear_scale / scales.length)
    return (torch.mean(residual_moment ** 2)
            + torch.mean(residual_shear ** 2))


def section_shear_constitutive_loss(model, applied_force_y, *,
                                    n_stations: int = 128,
                                    n_thickness: int = 24, scales=None,
                                    generator=None):
    """Thickness-integrated form of the shear constitutive law.

    The pointwise law ``tau_xy = G (u_y + v_x)`` is the only equation that ties
    ``v_x`` -- hence the bending deflection amplitude -- to the rest of the
    system.  Enforced pointwise it is catastrophically ill conditioned in a
    slender beam: ``G u_y`` and ``G v_x`` are each of order ``G u_c/H ~ 2e6 Pa``
    while their sum is the applied traction ``~1e2 Pa``, so demanding it
    pointwise asks the network for a five-digit cancellation and it prefers to
    set both terms to zero.  Relaxing the pointwise term instead leaves ``v``
    undetermined, because ``v_x`` appears nowhere else: that is the exact null
    space ``v -> v + f(x)``.

    REFUTED 2026-07-24.  The original motivation was that integrating removes
    the cancellation, because ``\\int u_y dy = u(x,H) - u(x,0)`` is a difference
    of *values* rather than of derivatives.  That is true for the ``u`` term but
    false for the ``v`` term: ``\\int v_x dy = H w'(x)`` is still an x-derivative
    of ``v``, and it is precisely the one that cancels against the rotation
    coming from ``u``.  Measured on the exact Saint-Venant solution at
    ``x = 3L/4``, the two contributions are ``+2179.59`` and ``-2179.69`` with
    sum ``-0.1`` -- a cancellation factor of ``2.2e4``, the same as the pointwise
    form.  Training confirms it: with this term active, ``v_tip`` stays frozen at
    ``1.5e-4`` of the FEM value.

    Kept only as a documented negative result.  Do not enable expecting it to
    determine the bending amplitude.
    """
    scales = direct_scales(applied_force_y) if scales is None else scales
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    stations = torch.rand(
        (n_stations, 1), device=device, dtype=dtype, generator=generator,
    ) * scales.length
    stations = stations.detach().requires_grad_(True)

    nodes = torch.linspace(0.0, HEIGHT, n_thickness, device=device, dtype=dtype)
    weights = torch.full_like(nodes, HEIGHT / (n_thickness - 1))
    weights[0] = weights[0] * 0.5
    weights[-1] = weights[-1] * 0.5

    x_flat = stations.repeat_interleave(n_thickness, dim=0)
    y_flat = nodes.repeat(n_stations).unsqueeze(1).requires_grad_(True)
    coordinates = torch.cat((x_flat, y_flat), dim=1)
    prediction = model(coordinates)

    u_gradient = torch.autograd.grad(
        prediction[:, 0].sum(), coordinates, create_graph=True,
        retain_graph=True,
    )[0]
    v_gradient = torch.autograd.grad(
        prediction[:, 1].sum(), coordinates, create_graph=True,
        retain_graph=True,
    )[0]
    gamma = (u_gradient[:, 1:2] + v_gradient[:, 0:1]).reshape(
        n_stations, n_thickness,
    )
    tau = prediction[:, 5].reshape(n_stations, n_thickness)

    shear_resultant = (tau * weights).sum(dim=1)
    constitutive_resultant = (
        float(materials.G) * gamma * weights
    ).sum(dim=1)

    reference = abs(float(applied_force_y))
    reference = reference if reference > 0.0 else 1.0
    residual = (shear_resultant - constitutive_resultant) / reference
    return torch.mean(residual ** 2)


def traction_BC_loss(xy_right, model, applied_force_y, scales=None, *,
                     normalize_residuals: bool = True):
    """Apply the selected vertical traction on the right end of the beam."""
    scales = direct_scales(applied_force_y) if scales is None else scales
    y_hat_right = model(xy_right)

    sigmax_pred_right = y_hat_right[:, 3:4]
    sigmaz_pred_right = y_hat_right[:, 4:5]
    tauxz_pred_right = y_hat_right[:, 5:6]

    n_x = 1.0
    n_y = 0.0

    traction_x = sigmax_pred_right * n_x + tauxz_pred_right * n_y
    traction_y = tauxz_pred_right * n_x + sigmaz_pred_right * n_y

    if DIRECT_TRACTION_PROFILE == "uniform":
        target_traction_y = torch.full_like(
            traction_y, applied_force_y / HEIGHT,
        )
    elif DIRECT_TRACTION_PROFILE == "parabolic":
        y = xy_right[:, 1:2]
        target_traction_y = (
            6.0 * applied_force_y * y * (HEIGHT - y) / HEIGHT**3
        )
    elif DIRECT_TRACTION_PROFILE == "point":
        target_traction_y = point_load_traction(
            xy_right[:, 1:2], applied_force_y,
        )
    else:
        raise ValueError(
            "DIRECT_TRACTION_PROFILE must be 'uniform', 'parabolic', or 'point'"
        )

    stress_scale = scales.stress_x if normalize_residuals else 1.0
    traction_scale = scales.traction if normalize_residuals else 1.0
    loss_traction_x = torch.mean((traction_x / stress_scale) ** 2)
    loss_traction_y = torch.mean(
        ((traction_y - target_traction_y) / traction_scale) ** 2
    )

    return loss_traction_x + loss_traction_y


def stress_BC_loss(xy_top, xy_bottom, xy_right, xy_left, model, scales=None, *,
                   normalize_residuals: bool = True):
    scales = direct_scales(APPLIED_FORCE_Y) if scales is None else scales
    y_hat_top = model(xy_top)
    y_hat_bottom = model(xy_bottom)

    sigmaz_pred_top = y_hat_top[:, 4:5]
    tauxz_pred_top = y_hat_top[:, 5:6]

    sigmaz_pred_bottom = y_hat_bottom[:, 4:5]
    tauxz_pred_bottom = y_hat_bottom[:, 5:6]

    # On y=0,H the outward normal is vertical, so a traction-free face
    # requires sigma_yy=tau_xy=0.  Penalising sigma_xx here suppresses the
    # bending stress precisely where it should be largest.
    stress_y_scale = scales.stress_y if normalize_residuals else 1.0
    shear_scale = scales.shear if normalize_residuals else 1.0
    loss_top = (torch.mean((sigmaz_pred_top / stress_y_scale) ** 2)
                + torch.mean((tauxz_pred_top / shear_scale) ** 2))
    loss_bottom = (torch.mean((sigmaz_pred_bottom / stress_y_scale) ** 2)
                   + torch.mean((tauxz_pred_bottom / shear_scale) ** 2))

    loss_right = traction_BC_loss(
        xy_right, model, applied_force_y=APPLIED_FORCE_Y, scales=scales,
        normalize_residuals=normalize_residuals,
    )

    return loss_top + loss_right + loss_bottom


def constitutive_traction_BC_loss(
        xy_top, xy_bottom, xy_right, model, applied_force_y, scales=None, *,
        normalize_residuals: bool = True):
    """Apply the same physical tractions through the primal constitutive trace.

    In an exact mixed solution this term is redundant with the independent
    stress-output BC and the constitutive PDE residual.  Numerically it gives
    the Neumann load a direct gradient path into ``(u,v,phi)`` instead of
    allowing the flux branch to absorb the complete load.
    """
    if not hasattr(model, "_constitutive_flux"):
        raise TypeError(
            "constitutive traction BCs require model._constitutive_flux"
        )
    scales = direct_scales(applied_force_y) if scales is None else scales

    def constitutive_stress(xy):
        differentiable_xy = xy
        if not differentiable_xy.requires_grad:
            differentiable_xy = xy.detach().requires_grad_(True)
        primary = model(differentiable_xy)[:, :3]
        return model._constitutive_flux(
            differentiable_xy, primary,
        )[:, :3]

    top_stress = constitutive_stress(xy_top)
    bottom_stress = constitutive_stress(xy_bottom)
    right_stress = constitutive_stress(xy_right)

    if DIRECT_TRACTION_PROFILE == "uniform":
        target_traction_y = torch.full_like(
            right_stress[:, 2:3], applied_force_y / HEIGHT,
        )
    elif DIRECT_TRACTION_PROFILE == "parabolic":
        y = xy_right[:, 1:2]
        target_traction_y = (
            6.0 * applied_force_y * y * (HEIGHT - y) / HEIGHT**3
        )
    elif DIRECT_TRACTION_PROFILE == "point":
        target_traction_y = point_load_traction(
            xy_right[:, 1:2], applied_force_y,
        )
    else:
        raise ValueError(
            "DIRECT_TRACTION_PROFILE must be 'uniform', 'parabolic', or 'point'"
        )

    stress_x_scale = scales.stress_x if normalize_residuals else 1.0
    stress_y_scale = scales.stress_y if normalize_residuals else 1.0
    shear_scale = scales.shear if normalize_residuals else 1.0
    traction_scale = scales.traction if normalize_residuals else 1.0
    return (
        torch.mean((top_stress[:, 1:2] / stress_y_scale) ** 2)
        + torch.mean((top_stress[:, 2:3] / shear_scale) ** 2)
        + torch.mean((bottom_stress[:, 1:2] / stress_y_scale) ** 2)
        + torch.mean((bottom_stress[:, 2:3] / shear_scale) ** 2)
        + torch.mean((right_stress[:, 0:1] / stress_x_scale) ** 2)
        + torch.mean(
            ((right_stress[:, 2:3] - target_traction_y)
             / traction_scale) ** 2
        )
    )


def electric_BC_loss(xy_right, xy_left, xy_top, model, *,
                     mode: str = DIRECT_ELECTRICAL_BC, scales=None,
                     normalize_residuals: bool = True):
    """Electrical side insulation plus the selected upper-face model."""
    if mode not in ("floating_electrode", "insulated"):
        raise ValueError("mode must be 'floating_electrode' or 'insulated'")
    scales = direct_scales(APPLIED_FORCE_Y) if scales is None else scales
    y_hat_right = model(xy_right)
    y_hat_left = model(xy_left)
    y_hat_top = model(xy_top)

    Dx_pred_right = y_hat_right[:, 6:7]
    Dy_pred_right = y_hat_right[:, 7:8]

    Dx_pred_left = y_hat_left[:, 6:7]
    Dy_pred_left = y_hat_left[:, 7:8]

    Dy_pred_top = y_hat_top[:, 7:8]
    phi_pred_top = y_hat_top[:, 2:3]

    n_right = torch.ones_like(xy_right)
    n_right[:, 1] = 0

    n_left = torch.ones_like(xy_left)
    n_left[:, 0] = -1
    n_left[:, 1] = 0

    D_dot_n_right = (Dx_pred_right * n_right[:, 0:1]
                     + Dy_pred_right * n_right[:, 1:2])
    D_dot_n_left = (Dx_pred_left * n_left[:, 0:1]
                    + Dy_pred_left * n_left[:, 1:2])
    electric_scale = (
        scales.electric_displacement if normalize_residuals else 1.0
    )
    phi_scale = scales.phi if normalize_residuals else 1.0
    side_loss = (
        torch.mean((D_dot_n_right / electric_scale) ** 2)
        + torch.mean((D_dot_n_left / electric_scale) ** 2)
    )
    d_top = Dy_pred_top / electric_scale
    if mode == "insulated":
        return side_loss + torch.mean(d_top ** 2)

    # A floating conductor is equipotential but may carry a nonuniform local
    # surface charge.  Open circuit constrains only its integral (uniform point
    # samples make the normalized mean an equivalent quadrature constraint).
    phi_top = phi_pred_top / phi_scale
    equipotential = torch.mean((phi_top - torch.mean(phi_top)) ** 2)
    zero_net_charge = torch.mean(d_top) ** 2
    return side_loss + equipotential + zero_net_charge


def electric_potential_BC_loss(xy_bottom, model):
    y_hat_bottom = model(xy_bottom)
    potential_pred = y_hat_bottom[:, 2:3]
    return torch.mean(potential_pred ** 2)


def displacement_BC_loss(xy_left, model):
    y_hat_left = model(xy_left)
    u = y_hat_left[:, 0:1]
    v = y_hat_left[:, 1:2]
    return torch.mean(u ** 2) + torch.mean(v ** 2)


def interface_loss(xy_interface, model, scales=None):
    """Enforce the six bonded-interface transmission conditions."""
    if not hasattr(model, "forward_trace"):
        raise TypeError("interface_loss requires a phase-enriched model")
    scales = direct_scales(APPLIED_FORCE_Y) if scales is None else scales
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


def update_family_weights(components, weights, model, *, rate=0.15,
                          stress_bounds=(1.0, 1e10),
                          electric_bounds=(1.0, 1e16)):
    """Smoothly equalize BC gradient norms to the raw PDE gradient norm."""
    norms = {}
    for name in ("pde", "bc_stress", "bc_electric"):
        gradients = torch.autograd.grad(
            components[name], model.parameters(), retain_graph=True,
            allow_unused=True,
        )
        finite = [gradient.reshape(-1) for gradient in gradients
                  if gradient is not None]
        if finite:
            norms[name] = float(torch.linalg.vector_norm(
                torch.cat(finite),
            ).detach().cpu())
        else:
            norms[name] = 0.0

    reference = max(norms["pde"], 1e-30)
    updated = dict(weights)
    for name, bounds in (
        ("bc_stress", stress_bounds),
        ("bc_electric", electric_bounds),
    ):
        target = reference / max(norms[name], 1e-30)
        target = min(max(target, bounds[0]), bounds[1])
        current = float(updated[name])
        log_weight = (
            (1.0 - rate) * math.log(max(current, 1e-30))
            + rate * math.log(target)
        )
        updated[name] = min(max(math.exp(log_weight), bounds[0]), bounds[1])
    updated["pde"] = float(updated.get("pde", 1.0))
    return updated, norms


def get_BC_loss(xy_top, xy_bottom, xy_right, xy_left, model, *,
                electrical_mode: str = DIRECT_ELECTRICAL_BC, scales=None,
                normalize_residuals: bool = True):
    scales = direct_scales(APPLIED_FORCE_Y) if scales is None else scales
    stress_loss_term = stress_BC_loss(
        xy_top, xy_bottom, xy_right, xy_left, model, scales=scales,
        normalize_residuals=normalize_residuals,
    )
    electric_loss_term = electric_BC_loss(
        xy_right, xy_left, xy_top, model,
        mode=electrical_mode, scales=scales,
        normalize_residuals=normalize_residuals,
    )
    return (stress_loss_term + electric_loss_term,
            stress_loss_term, electric_loss_term)


def loss_func(xy_top, xy_bottom, xy_right, xy_left,
              x_collocation, y_collocation,
              model, coefficients, loss_weights, n, f,
              only_BCs: bool = False, adjust: bool = False,
              electrical_mode: str = DIRECT_ELECTRICAL_BC,
              include_shear_constitutive: bool = True,
              shear_constitutive_mode: str = "full",
              constitutive_weight: float = 1.0,
              normal_constitutive_weight: float = 1.0,
              strict_transverse_constitutive: bool = False,
              constitutive_normalization: str = "represented",
              shear_rotation_stopgrad: bool = False,
              section_equilibrium_weight: float = 0.0,
              section_shear_weight: float = 0.0,
              section_stations: int = 128,
              section_thickness_points: int = 24,
              xy_interface=None,
              interface_weight: float = 1.0,
              normalize_residuals: bool = True,
              constitutive_traction_bcs: bool = False,
              equation_weights=None,
              return_components: bool = False):
    # Normally the dimensional load and the residual/output reference scales
    # coincide. A controlled load-continuation experiment may keep a smaller
    # scale while increasing the physical force; that changes conditioning,
    # not the dimensional PDE or boundary data.
    scales = getattr(model, "scales", direct_scales(APPLIED_FORCE_Y))
    BC_term, stress_loss_term, electric_loss_term = get_BC_loss(
        xy_top, xy_bottom, xy_right, xy_left, model,
        electrical_mode=electrical_mode, scales=scales,
        normalize_residuals=normalize_residuals,
    )
    constitutive_traction_loss_term = torch.zeros_like(stress_loss_term)
    if constitutive_traction_bcs:
        constitutive_traction_loss_term = constitutive_traction_BC_loss(
            xy_top, xy_bottom, xy_right, model, APPLIED_FORCE_Y,
            scales=scales, normalize_residuals=normalize_residuals,
        )
        stress_loss_term = (
            stress_loss_term + constitutive_traction_loss_term
        )
        BC_term = BC_term + constitutive_traction_loss_term
    (physics_loss_term, residual_sigmax, residual_sigmaz, residual_tauxz,
     residual_Dx, residual_Dy,
     divergence_sigma1, divergence_sigma2, divergence_D) = physics_loss(
        x_collocation, y_collocation, model, coefficients, scales=scales,
        include_shear_constitutive=include_shear_constitutive,
        shear_constitutive_mode=shear_constitutive_mode,
        constitutive_weight=constitutive_weight,
        normal_constitutive_weight=normal_constitutive_weight,
        strict_transverse_constitutive=strict_transverse_constitutive,
        constitutive_normalization=constitutive_normalization,
        shear_rotation_stopgrad=shear_rotation_stopgrad,
        normalize_residuals=normalize_residuals,
    )

    equation_terms = {
        "constitutive_sigmax": (
            constitutive_weight * normal_constitutive_weight
            * torch.mean(residual_sigmax ** 2)
        ),
        "constitutive_sigmay": (
            constitutive_weight * normal_constitutive_weight
            * torch.mean(residual_sigmaz ** 2)
        ),
        "constitutive_Dx": (
            constitutive_weight * torch.mean(residual_Dx ** 2)
        ),
        "constitutive_Dy": (
            constitutive_weight * torch.mean(residual_Dy ** 2)
        ),
        "equilibrium_x": torch.mean(divergence_sigma1 ** 2),
        "equilibrium_y": torch.mean(divergence_sigma2 ** 2),
        "gauss": torch.mean(divergence_D ** 2),
    }
    if include_shear_constitutive:
        if shear_constitutive_mode != "full" and equation_weights is not None:
            raise ValueError(
                "equation_weights currently require full shear constitutive"
            )
        if shear_constitutive_mode == "full":
            equation_terms["constitutive_tauxy"] = (
                constitutive_weight * torch.mean(residual_tauxz ** 2)
            )
    if equation_weights is not None:
        missing = set(equation_terms) - set(equation_weights)
        extra = set(equation_weights) - set(equation_terms)
        if missing or extra:
            raise ValueError(
                f"equation_weights mismatch: missing={missing}, extra={extra}"
            )
        physics_loss_term = sum(
            float(equation_weights[name]) * term
            for name, term in equation_terms.items()
        )

    if only_BCs:
        physics_loss_term = 0

    if n % f == 0 and adjust:
        loss_weights = update_weights(physics_loss_term, BC_term, loss_weights,
                                      model)

    lambda3 = loss_weights['pde']
    if "bc_stress" in loss_weights and "bc_electric" in loss_weights:
        loss = (
            lambda3 * physics_loss_term
            + loss_weights["bc_stress"] * stress_loss_term
            + loss_weights["bc_electric"] * electric_loss_term
        )
    else:
        loss = loss_weights['bc'] * BC_term + lambda3 * physics_loss_term
    interface_loss_term = torch.zeros(
        (), dtype=stress_loss_term.dtype, device=stress_loss_term.device,
    )
    if xy_interface is not None and getattr(model, "phase_enriched", False):
        interface_loss_term = interface_weight * interface_loss(
            xy_interface, model, scales=scales,
        )
        loss = loss + interface_loss_term
    section_loss_term = torch.zeros(
        (), dtype=stress_loss_term.dtype, device=stress_loss_term.device,
    )
    if section_equilibrium_weight > 0.0:
        section_loss_term = section_equilibrium_weight * section_equilibrium_loss(
            model, APPLIED_FORCE_Y, n_stations=section_stations,
            n_thickness=section_thickness_points, scales=scales,
        )
        loss = loss + section_loss_term
    if section_shear_weight > 0.0:
        section_shear_term = section_shear_weight * section_shear_constitutive_loss(
            model, APPLIED_FORCE_Y, n_stations=section_stations,
            n_thickness=section_thickness_points, scales=scales,
        )
        section_loss_term = section_loss_term + section_shear_term
        loss = loss + section_shear_term
    if return_components:
        return loss, loss_weights, {
            "pde": physics_loss_term,
            "pde_equations": equation_terms,
            "bc_stress": stress_loss_term,
            "bc_constitutive_traction": constitutive_traction_loss_term,
            "bc_electric": electric_loss_term,
            "interface": interface_loss_term,
            "section_equilibrium": section_loss_term,
        }
    return loss, loss_weights
