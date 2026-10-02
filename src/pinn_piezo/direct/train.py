"""Training driver for the direct PINN."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .losses import loss_func, update_family_weights


def _new_component_history():
    return {
        name: [] for name in (
            "constitutive", "balance", "pde", "boundary", "interface",
            "total",
        )
    }


def _direct_components(components, loss_weights, total):
    equations = components["pde_equations"]
    constitutive = sum(
        term for name, term in equations.items()
        if name.startswith("constitutive_")
    )
    balance = sum(
        equations[name]
        for name in ("equilibrium_x", "equilibrium_y", "gauss")
    )
    pde_weight = float(loss_weights["pde"])
    if "bc_stress" in loss_weights and "bc_electric" in loss_weights:
        boundary = (
            float(loss_weights["bc_stress"]) * components["bc_stress"]
            + float(loss_weights["bc_electric"])
            * components["bc_electric"]
        )
    else:
        boundary = float(loss_weights["bc"]) * (
            components["bc_stress"] + components["bc_electric"]
        )
    return {
        "constitutive": pde_weight * constitutive,
        "balance": pde_weight * balance,
        "pde": pde_weight * components["pde"],
        "boundary": boundary,
        "interface": components["interface"],
        "total": total,
    }


def _record_components(history, components):
    for name in history:
        history[name].append(float(components[name].detach().cpu()))


def tensorize(x, device, dtype=torch.float64):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=True)


def load_dataset(data_dir: Path, suffix: str = "_m1_d", fraction: float = 0.75):
    data_dir = Path(data_dir)
    xy_top = np.load(data_dir / f"xy_top_non_normalized{suffix}.npy")
    xy_bottom = np.load(data_dir / f"xy_bottom_non_normalized{suffix}.npy")
    xy_right = np.load(data_dir / f"xy_right_non_normalized{suffix}.npy")
    xy_left = np.load(data_dir / f"xy_left_non_normalized{suffix}.npy")
    x_collocation_orig = np.load(data_dir / f"x_collocation_non_normalized{suffix}.npy")

    num_samples = int(fraction * len(x_collocation_orig))
    indices = np.random.choice(len(x_collocation_orig), num_samples, replace=False)
    x_collocation = x_collocation_orig[indices]

    x_collocation, coefficients = np.split(x_collocation, [2], axis=1)
    x_collocation, y_collocation = np.split(x_collocation, [1], axis=1)
    # Accept legacy .npy files while using the physical stress-charge
    # convention: kappa is positive and e31/e33 reverse together with poling.
    coefficients[:, 4:6] = np.abs(coefficients[:, 4:6])

    return {
        "xy_top": xy_top,
        "xy_bottom": xy_bottom,
        "xy_right": xy_right,
        "xy_left": xy_left,
        "x_collocation": x_collocation,
        "y_collocation": y_collocation,
        "coefficients": coefficients,
    }


def to_device(arrays, device, dtype=torch.float64):
    return {k: tensorize(v, device, dtype=dtype).to(device)
            for k, v in arrays.items()}


def _freeze_stress_output_step(model):
    """Hold the three learned stress rows during a physics-only warm-up."""
    output = getattr(getattr(model, "net", None), "output", None)
    stress_slice = slice(3, 6)
    if output is None:
        flux = getattr(getattr(model, "net", None), "flux", None)
        output = getattr(flux, "output", None)
        stress_slice = slice(0, 3)
    if output is None:
        raise TypeError("stress warm-up requires a model.net.output layer")
    if output.weight.grad is not None:
        output.weight.grad[stress_slice].zero_()
    if output.bias.grad is not None:
        output.bias.grad[stress_slice].zero_()


def _pcgrad_families(components, loss_weights, *, only_bcs=False):
    """Rebuild the original scalar objective as three optimization families."""
    equations = components["pde_equations"]
    zero = torch.zeros_like(components["bc_stress"])
    constitutive = sum(
        (
            term for name, term in equations.items()
            if name.startswith("constitutive_")
        ),
        start=zero,
    )
    balance = sum(
        (
            equations[name]
            for name in ("equilibrium_x", "equilibrium_y", "gauss")
        ),
        start=zero,
    )
    if "bc_stress" in loss_weights and "bc_electric" in loss_weights:
        boundary = (
            loss_weights["bc_stress"] * components["bc_stress"]
            + loss_weights["bc_electric"] * components["bc_electric"]
        )
    else:
        boundary = loss_weights["bc"] * (
            components["bc_stress"] + components["bc_electric"]
        )
    boundary = (
        boundary
        + components["interface"]
        + components["section_equilibrium"]
    )
    if only_bcs:
        return {"boundary": boundary}
    pde_weight = loss_weights["pde"]
    return {
        "constitutive": pde_weight * constitutive,
        "balance": pde_weight * balance,
        "boundary": boundary,
    }


def _project_conflicting_gradients(gradients):
    """Return the sum of PCGrad-projected flattened gradient vectors."""
    if not gradients:
        raise ValueError("PCGrad requires at least one gradient family")
    original = [gradient.detach().clone() for gradient in gradients]
    projected = [gradient.clone() for gradient in original]
    for index, gradient in enumerate(projected):
        order = torch.randperm(
            len(original), device=gradient.device,
        ).tolist()
        for other_index in order:
            if other_index == index:
                continue
            other = original[other_index]
            inner = torch.dot(gradient, other)
            denominator = torch.dot(other, other)
            if inner < 0.0 and denominator > 0.0:
                gradient = gradient - inner / denominator * other
        projected[index] = gradient
    return torch.stack(projected, dim=0).sum(dim=0)


def _apply_pcgrad(model, families):
    """Populate ``parameter.grad`` with conflict-projected family gradients."""
    parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    flat_gradients = []
    names = list(families)
    for family in families.values():
        gradients = torch.autograd.grad(
            family, parameters, retain_graph=True, allow_unused=True,
        )
        flat_gradients.append(torch.cat([
            (
                torch.zeros_like(parameter)
                if gradient is None else gradient
            ).reshape(-1)
            for parameter, gradient in zip(parameters, gradients)
        ]))

    diagnostics = {}
    for index, name in enumerate(names):
        norm = torch.linalg.vector_norm(flat_gradients[index])
        diagnostics[f"norm_{name}"] = float(norm.detach().cpu())
        for other_index in range(index + 1, len(names)):
            other_name = names[other_index]
            other_norm = torch.linalg.vector_norm(
                flat_gradients[other_index],
            )
            denominator = norm * other_norm
            cosine = (
                torch.dot(
                    flat_gradients[index], flat_gradients[other_index],
                ) / denominator
                if denominator > 0.0 else torch.zeros_like(denominator)
            )
            diagnostics[f"cos_{name}_{other_name}"] = float(
                cosine.detach().cpu()
            )

    merged = _project_conflicting_gradients(flat_gradients)
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        parameter.grad = merged[offset:offset + count].view_as(
            parameter,
        ).clone()
        offset += count
    return diagnostics


def run_adam(model, tensors, *,
             epochs: int = 3000,
             lr: float = 1e-5,
             epochs_bc_warmup: int = 0,
             lr_bc_warmup: float = 1e-3,
             loss_weights=None,
             f: int = 200,
             checkpoints_dir: Path | None = None,
             electrical_mode: str = "floating_electrode",
             include_shear_constitutive: bool = True,
             shear_constitutive_mode: str = "full",
             constitutive_weight: float = 1.0,
             normal_constitutive_weight: float = 1.0,
             strict_transverse_constitutive: bool = False,
             constitutive_normalization: str = "represented",
             section_equilibrium_weight: float = 0.0,
             section_shear_weight: float = 0.0,
             stress_warmup_epochs: int = 0,
             freeze_flux_epochs: int = 0,
             interface_weight: float = 1.0,
             normalize_residuals: bool = True,
             constitutive_traction_bcs: bool = False,
             adjust_weights: bool = False,
             balance_weights: bool = False,
             balance_every: int = 100,
             balance_rate: float = 0.15,
             equation_minimax: bool = False,
             equation_weight_lr: float = 1e-2,
             equation_weight_entropy: float = 0.05,
             pcgrad: bool = False,
             resample_fn=None,
             resample_every: int = 200,
             component_history=None,
             snapshots_dir: Path | None = None,
             snapshot_epochs=()):
    if loss_weights is None:
        loss_weights = {'pde': 1.0, 'bc': 10.0}

    if freeze_flux_epochs < 0:
        raise ValueError("freeze_flux_epochs must be non-negative")
    if pcgrad and (adjust_weights or balance_weights or equation_minimax):
        raise ValueError(
            "pcgrad cannot be combined with another gradient/weight scheme"
        )
    if pcgrad and shear_constitutive_mode != "full":
        raise ValueError("pcgrad requires full shear constitutive")
    flux_branch = getattr(getattr(model, "net", None), "flux", None)
    if freeze_flux_epochs > 0 and flux_branch is None:
        raise TypeError("freeze_flux_epochs requires a split-trunk model")
    if flux_branch is not None and freeze_flux_epochs > 0:
        for parameter in flux_branch.parameters():
            parameter.requires_grad_(False)
        print(f"Flux branch frozen for {freeze_flux_epochs} Adam epochs.")

    initial_lr = lr_bc_warmup if epochs_bc_warmup > 0 else lr
    optimizer = torch.optim.Adam(params=model.parameters(), lr=initial_lr)
    equation_logits = None
    equation_optimizer = None
    equation_names = None
    equation_weights = None
    if equation_minimax:
        if balance_weights or adjust_weights:
            raise ValueError(
                "equation_minimax cannot be combined with other weight schemes"
            )
        if shear_constitutive_mode != "full":
            raise ValueError("equation_minimax requires full shear constitutive")
        equation_names = [
            "constitutive_sigmax",
            "constitutive_sigmay",
            "constitutive_Dx",
            "constitutive_Dy",
            "equilibrium_x",
            "equilibrium_y",
            "gauss",
        ]
        if include_shear_constitutive:
            equation_names.insert(2, "constitutive_tauxy")
        equation_logits = torch.zeros(
            len(equation_names), dtype=next(model.parameters()).dtype,
            device=next(model.parameters()).device, requires_grad=True,
        )
        equation_optimizer = torch.optim.Adam(
            [equation_logits], lr=equation_weight_lr,
        )
    # Matches the notebook scheduler definition (kept for parity, not stepped):
    _scheduler = torch.optim.lr_scheduler.StepLR(  # noqa: F841
        optimizer, step_size=5000, gamma=0.95,
    )

    best_loss = float('inf')
    loss_list = []
    if component_history is None:
        component_history = _new_component_history()

    for epoch in range(epochs):
        if epoch == freeze_flux_epochs and flux_branch is not None:
            for parameter in flux_branch.parameters():
                parameter.requires_grad_(True)
            if freeze_flux_epochs > 0:
                print("Flux branch released; continuing joint training.")
        if (resample_fn is not None and epoch > 0
                and epoch % resample_every == 0):
            tensors.clear()
            tensors.update(resample_fn(epoch // resample_every))
        if epoch == epochs_bc_warmup:
            for group in optimizer.param_groups:
                group['lr'] = lr
        optimizer.zero_grad()
        if equation_optimizer is not None:
            equation_optimizer.zero_grad()
        loss_result = loss_func(
            tensors["xy_top"], tensors["xy_bottom"],
            tensors["xy_right"], tensors["xy_left"],
            tensors["x_collocation"], tensors["y_collocation"],
            model, tensors["coefficients"], loss_weights, epoch, f,
            only_BCs=epoch < epochs_bc_warmup,
            electrical_mode=electrical_mode,
            include_shear_constitutive=include_shear_constitutive,
            shear_constitutive_mode=shear_constitutive_mode,
            constitutive_weight=constitutive_weight,
            normal_constitutive_weight=normal_constitutive_weight,
            strict_transverse_constitutive=strict_transverse_constitutive,
            constitutive_normalization=constitutive_normalization,
            section_equilibrium_weight=section_equilibrium_weight,
            section_shear_weight=section_shear_weight,
            xy_interface=tensors.get("xy_interface"),
            interface_weight=interface_weight,
            normalize_residuals=normalize_residuals,
            constitutive_traction_bcs=constitutive_traction_bcs,
            adjust=adjust_weights,
            return_components=True,
        )
        dual_loss = None
        pcgrad_diagnostics = None
        if equation_minimax:
            _, loss_weights, components = loss_result
            terms = torch.stack([
                components["pde_equations"][name]
                for name in equation_names
            ])
            probabilities = torch.softmax(equation_logits, dim=0)
            live_weights = len(equation_names) * probabilities
            weighted_pde = torch.sum(live_weights.detach() * terms)
            loss = (
                loss_weights["pde"] * weighted_pde
                + loss_weights.get("bc_stress", loss_weights.get("bc", 1.0))
                * components["bc_stress"]
                + loss_weights.get("bc_electric", loss_weights.get("bc", 1.0))
                * components["bc_electric"]
                + components["interface"]
            )
            # Gradient ascent on a fixed-sum simplex emphasizes whichever
            # strong equation the network is currently sacrificing.  A small
            # KL penalty prevents immediate collapse to a single equation.
            kl_uniform = torch.sum(
                probabilities
                * torch.log(probabilities * len(equation_names) + 1e-30)
            )
            dual_objective = (
                torch.sum(live_weights * terms.detach())
                - equation_weight_entropy * kl_uniform
            )
            dual_loss = -dual_objective
            equation_weights = {
                name: float(weight.detach().cpu())
                for name, weight in zip(equation_names, live_weights)
            }
        elif balance_weights:
            loss, loss_weights, components = loss_result
            if epoch % balance_every == 0:
                loss_weights, gradient_norms = update_family_weights(
                    components, loss_weights, model, rate=balance_rate,
                )
                loss = (
                    loss_weights["pde"] * components["pde"]
                    + loss_weights["bc_stress"] * components["bc_stress"]
                    + loss_weights["bc_electric"]
                    * components["bc_electric"]
                )
                print(
                    "Adaptive weights:", loss_weights,
                    "gradient norms:", gradient_norms,
                )
        elif pcgrad:
            loss, loss_weights, components = loss_result
            families = _pcgrad_families(
                components, loss_weights,
                only_bcs=epoch < epochs_bc_warmup,
            )
            pcgrad_diagnostics = _apply_pcgrad(model, families)
        else:
            loss, loss_weights, components = loss_result
        if not pcgrad:
            loss.backward()
        if dual_loss is not None:
            dual_loss.backward()
        if epoch < stress_warmup_epochs:
            _freeze_stress_output_step(model)
        optimizer.step()
        if equation_optimizer is not None:
            equation_optimizer.step()
        loss_list.append(loss.item())
        _record_components(
            component_history,
            _direct_components(components, loss_weights, loss),
        )

        completed_epoch = epoch + 1
        if snapshots_dir is not None and completed_epoch in snapshot_epochs:
            snapshot_path = (
                Path(snapshots_dir) / f"model_epoch_{completed_epoch}.pt"
            )
            torch.save(
                {
                    "epoch": completed_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item(),
                    "loss_weights": loss_weights,
                    "equation_weights": equation_weights,
                },
                snapshot_path,
            )
            print(f"Milestone snapshot saved to {snapshot_path}")

        if epoch % 100 == 0:
            print(f"Epoch: {epoch}/{epochs}. Loss: {loss.item()}.")
            print(optimizer.state_dict()['param_groups'][0]['lr'])
            if pcgrad_diagnostics is not None:
                print("PCGrad diagnostics:", pcgrad_diagnostics)
            if checkpoints_dir is not None:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss.item(),
                        "loss_weights": loss_weights,
                        "equation_weights": equation_weights,
                    },
                    Path(checkpoints_dir)
                    / f"model_epoch_{epoch}_loss_{loss.item():.4f}.pt",
                )

        if epoch % f == 0:
            print(f"Lambda_1: {loss_weights}.")
            if equation_weights is not None:
                print(f"Strong-equation weights: {equation_weights}.")

    return loss_list, loss_weights, equation_weights, best_loss


def run_lbfgs(model, tensors, loss_weights, *,
              epochs: int = 0,
              lr: float = 0.0001,
              f: int = 200,
              epochs_adam_offset: int = 0,
              electrical_mode: str = "floating_electrode",
              include_shear_constitutive: bool = True,
              shear_constitutive_mode: str = "full",
              constitutive_weight: float = 1.0,
              normal_constitutive_weight: float = 1.0,
              strict_transverse_constitutive: bool = False,
              constitutive_normalization: str = "represented",
              section_equilibrium_weight: float = 0.0,
              section_shear_weight: float = 0.0,
              line_search_fn: str | None = "strong_wolfe",
              freeze_stress: bool = False,
              grad_clip: float | None = 0.5,
              interface_weight: float = 1.0,
              normalize_residuals: bool = True,
              constitutive_traction_bcs: bool = False,
              equation_weights=None,
              checkpoints_dir: Path | None = None,
              component_history=None):
    optimizer = torch.optim.LBFGS(
        params=model.parameters(), lr=lr,
        line_search_fn=line_search_fn,
    )
    loss_list = []
    if component_history is None:
        component_history = _new_component_history()
    total_epochs = epochs_adam_offset + epochs

    for epoch in range(epochs):

        closure_components = None
        closure_loss = None

        def closure():
            nonlocal loss_weights, closure_components, closure_loss
            optimizer.zero_grad()
            loss, loss_weights, closure_components = loss_func(
                tensors["xy_top"], tensors["xy_bottom"],
                tensors["xy_right"], tensors["xy_left"],
                tensors["x_collocation"], tensors["y_collocation"],
                model, tensors["coefficients"], loss_weights, epoch, f,
                electrical_mode=electrical_mode,
                include_shear_constitutive=include_shear_constitutive,
                shear_constitutive_mode=shear_constitutive_mode,
                constitutive_weight=constitutive_weight,
                normal_constitutive_weight=normal_constitutive_weight,
                strict_transverse_constitutive=strict_transverse_constitutive,
                constitutive_normalization=constitutive_normalization,
                section_equilibrium_weight=section_equilibrium_weight,
                section_shear_weight=section_shear_weight,
                xy_interface=tensors.get("xy_interface"),
                interface_weight=interface_weight,
                normalize_residuals=normalize_residuals,
                constitutive_traction_bcs=constitutive_traction_bcs,
                equation_weights=equation_weights,
                return_components=True,
            )
            closure_loss = loss
            loss.backward()
            if freeze_stress:
                _freeze_stress_output_step(model)
            if grad_clip is not None and grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip,
                )
            return loss

        loss = optimizer.step(closure)
        loss_list.append(loss.item())
        if closure_components is not None and closure_loss is not None:
            _record_components(
                component_history,
                _direct_components(
                    closure_components, loss_weights, closure_loss,
                ),
            )

        if not torch.isfinite(loss):
            print('nan')
            break

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch: {epochs_adam_offset + epoch}/{total_epochs}. "
                  f"Loss: {loss.item()}.")
        if checkpoints_dir is not None:
            torch.save(
                model.state_dict(),
                Path(checkpoints_dir)
                / f"model_epoch_{epoch}_loss_{loss.item():.4f}.pt",
            )

    return loss_list, loss_weights


def train(model, tensors, *,
          epochs_adam: int = 3000,
          epochs_lbfgs: int = 0,
          lr_adam: float = 1e-5,
          epochs_bc_warmup: int = 0,
          lr_bc_warmup: float = 1e-3,
          lr_lbfgs: float = 0.0001,
          loss_weights=None,
          f: int = 200,
          electrical_mode: str = "floating_electrode",
          include_shear_constitutive: bool = True,
          shear_constitutive_mode: str = "full",
          constitutive_weight: float = 1.0,
          normal_constitutive_weight: float = 1.0,
          strict_transverse_constitutive: bool = False,
          constitutive_normalization: str = "represented",
          section_equilibrium_weight: float = 0.0,
          section_shear_weight: float = 0.0,
          stress_warmup_epochs: int = 0,
          freeze_flux_epochs: int = 0,
          stress_warmup_lbfgs: bool = False,
          interface_weight: float = 1.0,
          normalize_residuals: bool = True,
          constitutive_traction_bcs: bool = False,
          adjust_weights: bool = False,
          balance_weights: bool = False,
          balance_every: int = 100,
          balance_rate: float = 0.15,
          equation_minimax: bool = False,
          equation_weight_lr: float = 1e-2,
          equation_weight_entropy: float = 0.05,
          pcgrad: bool = False,
          lbfgs_grad_clip: float | None = 0.5,
          resample_fn=None,
          resample_every: int = 200,
          lbfgs_tensors=None,
          checkpoints_adam_dir: Path | None = None,
          checkpoints_lbfgs_dir: Path | None = None,
          snapshots_dir: Path | None = None,
          snapshot_epochs=()):
    if loss_weights is None:
        loss_weights = {'pde': 1.0, 'bc': 10.0}

    start_time = time.time()
    components_adam = _new_component_history()
    components_lbfgs = _new_component_history()

    loss_list_adam, loss_weights, equation_weights, _ = run_adam(
        model, tensors,
        epochs=epochs_adam, lr=lr_adam,
        epochs_bc_warmup=epochs_bc_warmup,
        lr_bc_warmup=lr_bc_warmup,
        loss_weights=loss_weights, f=f,
        electrical_mode=electrical_mode,
        include_shear_constitutive=include_shear_constitutive,
        shear_constitutive_mode=shear_constitutive_mode,
        constitutive_weight=constitutive_weight,
        normal_constitutive_weight=normal_constitutive_weight,
        strict_transverse_constitutive=strict_transverse_constitutive,
        constitutive_normalization=constitutive_normalization,
        section_equilibrium_weight=section_equilibrium_weight,
        section_shear_weight=section_shear_weight,
        stress_warmup_epochs=stress_warmup_epochs,
        freeze_flux_epochs=freeze_flux_epochs,
        interface_weight=interface_weight,
        normalize_residuals=normalize_residuals,
        constitutive_traction_bcs=constitutive_traction_bcs,
        adjust_weights=adjust_weights,
        balance_weights=balance_weights,
        balance_every=balance_every,
        balance_rate=balance_rate,
        equation_minimax=equation_minimax,
        equation_weight_lr=equation_weight_lr,
        equation_weight_entropy=equation_weight_entropy,
        pcgrad=pcgrad,
        resample_fn=resample_fn,
        resample_every=resample_every,
        checkpoints_dir=checkpoints_adam_dir,
        component_history=components_adam,
        snapshots_dir=snapshots_dir,
        snapshot_epochs=snapshot_epochs,
    )
    # Preserve the stable Adam endpoint even if a later quasi-Newton step is
    # rejected or diverges.  Direct runs are expensive enough that silently
    # overwriting this state makes the experiment irreproducible.
    adam_state_dict = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }

    loss_list_lbfgs, loss_weights = run_lbfgs(
        model, tensors if lbfgs_tensors is None else lbfgs_tensors, loss_weights,
        epochs=epochs_lbfgs, lr=lr_lbfgs, f=f,
        epochs_adam_offset=epochs_adam,
        electrical_mode=electrical_mode,
        include_shear_constitutive=include_shear_constitutive,
        shear_constitutive_mode=shear_constitutive_mode,
        constitutive_weight=constitutive_weight,
        normal_constitutive_weight=normal_constitutive_weight,
        strict_transverse_constitutive=strict_transverse_constitutive,
        constitutive_normalization=constitutive_normalization,
        section_equilibrium_weight=section_equilibrium_weight,
        section_shear_weight=section_shear_weight,
        freeze_stress=stress_warmup_lbfgs,
        interface_weight=interface_weight,
        normalize_residuals=normalize_residuals,
        constitutive_traction_bcs=constitutive_traction_bcs,
        equation_weights=equation_weights,
        grad_clip=lbfgs_grad_clip,
        checkpoints_dir=checkpoints_lbfgs_dir,
        component_history=components_lbfgs,
    )

    total_time = time.time() - start_time
    print(total_time)
    print(total_time / 60)

    return {
        "loss_list": loss_list_adam + loss_list_lbfgs,
        "loss_weights": loss_weights,
        "equation_weights": equation_weights,
        "total_time": total_time,
        "adam_state_dict": adam_state_dict,
        "loss_components": {
            name: components_adam[name] + components_lbfgs[name]
            for name in components_adam
        },
    }
