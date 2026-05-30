"""Training driver for the indirect PINN."""

from __future__ import annotations

import os
import time
from math import nan
from pathlib import Path

import numpy as np
import torch

from .losses import loss_func


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
    coefficients[:, 7] = -coefficients[:, 7]

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


def run_adam(model, tensors, *,
             epochs: int = 1000,
             lr: float = 0.001,
             loss_weights=None,
             f: int = 500,
             checkpoints_dir: Path | None = None,
             mlflow=None):
    if loss_weights is None:
        loss_weights = {'pde': 1.0, 'bc': 1.0}

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    best_loss = float('inf')
    loss_list = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss, loss_weights = loss_func(
            tensors["xy_top"], tensors["xy_bottom"],
            tensors["xy_right"], tensors["xy_left"],
            tensors["x_collocation"], tensors["y_collocation"],
            model, tensors["coefficients"], loss_weights, epoch, f,
        )
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

        if epoch % 100 == 0:
            print(f"Epoch: {epoch}/{epochs}. Loss: {loss.item()}.")
            print(optimizer.state_dict()['param_groups'][0]['lr'])
            if mlflow is not None:
                mlflow.log_metric("loss_ADAM", loss.item(), step=epoch)

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
              mlflow=None):
    optimizer = torch.optim.LBFGS(params=model.parameters(), lr=lr)
    best_loss = float('inf')
    loss_list = []
    total_epochs = epochs_adam_offset + epochs

    for epoch in range(epochs):

        def closure():
            nonlocal loss_weights
            optimizer.zero_grad()
            loss, loss_weights = loss_func(
                tensors["xy_top"], tensors["xy_bottom"],
                tensors["xy_right"], tensors["xy_left"],
                tensors["x_collocation"], tensors["y_collocation"],
                model, tensors["coefficients"], loss_weights, epoch, f,
            )
            loss.backward()
            return loss

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step(closure)
        loss = closure()
        loss_list.append(loss.item())

        if loss.item() == nan:
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
          mlflow=None):
    if loss_weights is None:
        loss_weights = {'pde': 1.0, 'bc': 1.0}

    start_time = time.time()

    loss_list_adam, loss_weights, best_loss_adam = run_adam(
        model, tensors,
        epochs=epochs_adam, lr=lr_adam,
        loss_weights=loss_weights, f=f,
        checkpoints_dir=checkpoints_adam_dir,
        mlflow=mlflow,
    )

    loss_list_lbfgs, loss_weights, best_loss_lbfgs = run_lbfgs(
        model, tensors, loss_weights,
        epochs=epochs_lbfgs, lr=lr_lbfgs, f=f,
        checkpoints_dir=checkpoints_lbfgs_dir,
        epochs_adam_offset=epochs_adam,
        mlflow=mlflow,
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
    }
