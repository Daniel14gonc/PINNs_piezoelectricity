"""Training driver for the direct PINN."""

from __future__ import annotations

import time
from math import nan
from pathlib import Path

import numpy as np
import torch

from .losses import loss_func


def tensorize(x, device, dtype=torch.float32):
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


def to_device(arrays, device, dtype=torch.float32):
    return {k: tensorize(v, device, dtype=dtype).to(device)
            for k, v in arrays.items()}


def run_adam(model, tensors, *,
             epochs: int = 3000,
             lr: float = 0.005,
             loss_weights=None,
             f: int = 200,
             checkpoints_dir: Path | None = None):
    if loss_weights is None:
        loss_weights = {'pde': 1, 'bc': 1}

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    # Matches the notebook scheduler definition (kept for parity, not stepped):
    _scheduler = torch.optim.lr_scheduler.StepLR(  # noqa: F841
        optimizer, step_size=5000, gamma=0.95,
    )

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

        if epoch % f == 0:
            print(f"Lambda_1: {loss_weights}.")

    return loss_list, loss_weights, best_loss


def run_lbfgs(model, tensors, loss_weights, *,
              epochs: int = 0,
              lr: float = 0.0001,
              f: int = 200,
              epochs_adam_offset: int = 0):
    optimizer = torch.optim.LBFGS(params=model.parameters(), lr=lr)
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

    return loss_list, loss_weights


def train(model, tensors, *,
          epochs_adam: int = 3000,
          epochs_lbfgs: int = 0,
          lr_adam: float = 0.005,
          lr_lbfgs: float = 0.0001,
          loss_weights=None,
          f: int = 200):
    if loss_weights is None:
        loss_weights = {'pde': 1, 'bc': 1}

    start_time = time.time()

    loss_list_adam, loss_weights, _ = run_adam(
        model, tensors,
        epochs=epochs_adam, lr=lr_adam,
        loss_weights=loss_weights, f=f,
    )

    loss_list_lbfgs, loss_weights = run_lbfgs(
        model, tensors, loss_weights,
        epochs=epochs_lbfgs, lr=lr_lbfgs, f=f,
        epochs_adam_offset=epochs_adam,
    )

    total_time = time.time() - start_time
    print(total_time)
    print(total_time / 60)

    return {
        "loss_list": loss_list_adam + loss_list_lbfgs,
        "loss_weights": loss_weights,
        "total_time": total_time,
    }
