"""Training driver for the indirect PINN."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .losses import loss_func


def _new_component_history():
    return {
        name: [] for name in (
            "constitutive", "balance", "pde", "boundary", "interface",
            "total",
        )
    }


def _record_components(history, components):
    for name in history:
        history[name].append(float(components[name].detach().cpu()))


def tensorize(x, device, dtype=torch.float64):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=True)


def load_dataset(data_dir: Path, suffix: str = "_m1", fraction: float = 1.0):
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
    # Older generated files stored the permittivity with the legacy negative
    # sign.  The stress-charge law used by the losses requires positive kappa.
    # The raw e31/e33 columns already reverse *together* between layers; do not
    # flip e33 a second time here.
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


def _freeze_stress_output_step(model):
    """Keep only mixed stress rows fixed during the mechanical warm-up."""
    output = getattr(getattr(model, "net", None), "output", None)
    if output is None:
        raise TypeError("stress warm-up requires a model.net.output layer")
    if output.weight.grad is not None:
        output.weight.grad[3:6].zero_()
    if output.bias.grad is not None:
        output.bias.grad[3:6].zero_()


def to_device(arrays, device, dtype=torch.float64):
    return {k: tensorize(v, device, dtype=dtype).to(device)
            for k, v in arrays.items()}


def run_adam(model, tensors, *,
             epochs: int = 1000,
             lr: float = 0.001,
             loss_weights=None,
             f: int = 500,
             checkpoints_dir: Path | None = None,
             mlflow=None,
             resample_fn=None,
             resample_every: int = 200,
             interface_weight: float = 1.0,
             stress_warmup_epochs: int = 0,
             normal_constitutive_weight: float = 1.0,
             constitutive_normalization: str = "represented",
             constitutive_traction_bcs: bool = False,
             constitutive_traction_weight: float = 1.0,
             constitutive_flux_stopgrad: bool = False,
             mechanical_constitutive_stopgrad: bool = False,
             component_history=None,
             snapshots_dir: Path | None = None,
             snapshot_epochs=()):
    if loss_weights is None:
        loss_weights = {'pde': 1.0, 'bc': 1.0}

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    best_loss = float('inf')
    loss_list = []
    if component_history is None:
        component_history = _new_component_history()

    for epoch in range(epochs):
        if (resample_fn is not None and epoch > 0
                and epoch % resample_every == 0):
            # Mutate in place so the caller retains the latest Adam cloud.
            tensors.clear()
            tensors.update(resample_fn(epoch // resample_every))
        optimizer.zero_grad()
        loss, loss_weights, components = loss_func(
            tensors["xy_top"], tensors["xy_bottom"],
            tensors["xy_right"], tensors["xy_left"],
            tensors["x_collocation"], tensors["y_collocation"],
            model, tensors["coefficients"], loss_weights, epoch, f,
            xy_interface=tensors.get("xy_interface"),
            interface_weight=interface_weight,
            normal_constitutive_weight=normal_constitutive_weight,
            constitutive_normalization=constitutive_normalization,
            constitutive_traction_bcs=constitutive_traction_bcs,
            constitutive_traction_weight=constitutive_traction_weight,
            constitutive_flux_stopgrad=constitutive_flux_stopgrad,
            mechanical_constitutive_stopgrad=(
                mechanical_constitutive_stopgrad
            ),
            return_components=True,
        )
        loss.backward()
        if epoch < stress_warmup_epochs:
            _freeze_stress_output_step(model)
        optimizer.step()
        current_loss = loss.item()
        loss_list.append(current_loss)
        if current_loss < best_loss:
            best_loss = current_loss
        _record_components(component_history, components)

        completed_epoch = epoch + 1
        if snapshots_dir is not None and completed_epoch in snapshot_epochs:
            snapshot_path = (
                Path(snapshots_dir) / f"model_epoch_{completed_epoch}.pt"
            )
            torch.save({
                "epoch": completed_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item(),
                "loss_weights": loss_weights,
            }, snapshot_path)
            print(f"Milestone snapshot saved to {snapshot_path}")

        if epoch % 100 == 0:
            print(f"Epoch: {epoch}/{epochs}. Loss: {loss.item()}.")
            print(optimizer.state_dict()['param_groups'][0]['lr'])
            if mlflow is not None:
                mlflow.log_metric("loss_ADAM", loss.item(), step=epoch)

            if current_loss <= best_loss and checkpoints_dir is not None:
                ckpt_path = Path(checkpoints_dir) / (
                    f"model_epoch_{epoch}_loss_{best_loss:.4f}.pt"
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'loss_weights': loss_weights,
                }, ckpt_path)
                print(f"Checkpoint saved at epoch {epoch} with loss "
                      f"{best_loss:.4f}")

        if epoch % f == 0:
            print(f"Lambda_1: {loss_weights}.")

    return loss_list, loss_weights, best_loss


def run_lbfgs(model, tensors, loss_weights, *,
              epochs: int = 200,
              lr: float = 0.01,
              f: int = 500,
              checkpoints_dir: Path | None = None,
              epochs_adam_offset: int = 0,
              mlflow=None,
              interface_weight: float = 1.0,
              line_search_fn: str | None = "strong_wolfe",
              freeze_stress: bool = False,
              normal_constitutive_weight: float = 1.0,
              constitutive_normalization: str = "represented",
              constitutive_traction_bcs: bool = False,
              constitutive_traction_weight: float = 1.0,
              grad_clip: float | None = 0.5,
              component_history=None):
    # Every tensor in this objective is fixed.  L-BFGS/strong-Wolfe is invalid
    # if boundary or interface points are resampled inside the closure.
    optimizer = torch.optim.LBFGS(
        params=model.parameters(), lr=lr,
        line_search_fn=line_search_fn,
    )
    best_loss = float('inf')
    loss_list = []
    if component_history is None:
        component_history = _new_component_history()
    total_epochs = epochs_adam_offset + epochs

    for epoch in range(epochs):

        closure_components = None

        def closure():
            nonlocal loss_weights, closure_components
            optimizer.zero_grad()
            loss, loss_weights, closure_components = loss_func(
                tensors["xy_top"], tensors["xy_bottom"],
                tensors["xy_right"], tensors["xy_left"],
                tensors["x_collocation"], tensors["y_collocation"],
                model, tensors["coefficients"], loss_weights, epoch, f,
                xy_interface=tensors.get("xy_interface"),
                interface_weight=interface_weight,
                normal_constitutive_weight=normal_constitutive_weight,
                constitutive_normalization=constitutive_normalization,
                constitutive_traction_bcs=constitutive_traction_bcs,
                constitutive_traction_weight=constitutive_traction_weight,
                return_components=True,
            )
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
        if closure_components is not None:
            _record_components(component_history, closure_components)

        if not torch.isfinite(loss):
            print('nan')
            break

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch: {epochs_adam_offset + epoch}/{total_epochs}. "
                  f"Loss: {loss.item()}.")
            if mlflow is not None:
                mlflow.log_metric("loss_LBFGS", loss.item(),
                                  step=epochs_adam_offset + epoch)

        if loss.item() < best_loss and checkpoints_dir is not None:
            best_loss = loss.item()
            ckpt_path = Path(checkpoints_dir) / (
                f"model_epoch_{epoch}_loss_{best_loss:.4f}.pt"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'loss_weights': loss_weights,
            }, ckpt_path)

    return loss_list, loss_weights, best_loss


def train(model, tensors, *,
          epochs_adam: int = 1000,
          epochs_lbfgs: int = 200,
          lr_adam: float = 0.001,
          lr_lbfgs: float = 0.01,
          loss_weights=None,
          f: int = 500,
          checkpoints_adam_dir: Path | None = None,
          checkpoints_lbfgs_dir: Path | None = None,
          mlflow=None,
          resample_fn=None,
          resample_every: int = 200,
          lbfgs_tensors=None,
          interface_weight: float = 1.0,
          stress_warmup_epochs: int = 0,
          stress_warmup_lbfgs: bool = False,
          normal_constitutive_weight: float = 1.0,
          constitutive_normalization: str = "represented",
          constitutive_traction_bcs: bool = False,
          constitutive_traction_weight: float = 1.0,
          constitutive_flux_stopgrad: bool = False,
          mechanical_constitutive_stopgrad: bool = False,
          lbfgs_grad_clip: float | None = 0.5,
          snapshots_dir: Path | None = None,
          snapshot_epochs=()):
    if loss_weights is None:
        loss_weights = {'pde': 1.0, 'bc': 1.0}
    if (
        constitutive_flux_stopgrad or mechanical_constitutive_stopgrad
    ) and epochs_lbfgs > 0:
        raise ValueError(
            "constitutive gradient routing is incompatible with L-BFGS: "
            "the routed gradient is not the complete derivative used by "
            "quasi-Newton line search"
        )

    start_time = time.time()
    components_adam = _new_component_history()
    components_lbfgs = _new_component_history()

    loss_list_adam, loss_weights, best_loss_adam = run_adam(
        model, tensors,
        epochs=epochs_adam, lr=lr_adam,
        loss_weights=loss_weights, f=f,
        checkpoints_dir=checkpoints_adam_dir,
        mlflow=mlflow,
        resample_fn=resample_fn,
        resample_every=resample_every,
        interface_weight=interface_weight,
        stress_warmup_epochs=stress_warmup_epochs,
        normal_constitutive_weight=normal_constitutive_weight,
        constitutive_normalization=constitutive_normalization,
        constitutive_traction_bcs=constitutive_traction_bcs,
        constitutive_traction_weight=constitutive_traction_weight,
        constitutive_flux_stopgrad=constitutive_flux_stopgrad,
        mechanical_constitutive_stopgrad=mechanical_constitutive_stopgrad,
        component_history=components_adam,
        snapshots_dir=snapshots_dir,
        snapshot_epochs=snapshot_epochs,
    )

    loss_list_lbfgs, loss_weights, best_loss_lbfgs = run_lbfgs(
        model, tensors if lbfgs_tensors is None else lbfgs_tensors, loss_weights,
        epochs=epochs_lbfgs, lr=lr_lbfgs, f=f,
        checkpoints_dir=checkpoints_lbfgs_dir,
        epochs_adam_offset=epochs_adam,
        mlflow=mlflow,
        interface_weight=interface_weight,
        freeze_stress=stress_warmup_lbfgs,
        normal_constitutive_weight=normal_constitutive_weight,
        constitutive_normalization=constitutive_normalization,
        constitutive_traction_bcs=constitutive_traction_bcs,
        constitutive_traction_weight=constitutive_traction_weight,
        grad_clip=lbfgs_grad_clip,
        component_history=components_lbfgs,
    )

    total_time = time.time() - start_time
    print(total_time)
    print(total_time / 60)

    return {
        "loss_list": loss_list_adam + loss_list_lbfgs,
        "best_loss_adam": best_loss_adam,
        "best_loss_lbfgs": best_loss_lbfgs,
        "loss_weights": loss_weights,
        "total_time": total_time,
        "loss_components": {
            name: components_adam[name] + components_lbfgs[name]
            for name in components_adam
        },
    }
