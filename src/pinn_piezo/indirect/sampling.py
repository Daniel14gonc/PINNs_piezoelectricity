"""Reproducible collocation sampling for the one-network interface PINN."""

from __future__ import annotations

import numpy as np
import torch

from ..config import CENTER, HEIGHT, WIDTH
from ..geometry import build_coefficients


def sample_training_tensors(
    *,
    n_interior_per_layer: int,
    n_boundary: int,
    n_interface: int,
    seed: int,
    device,
    dtype=torch.float64,
    interior_x_margin: float = 0.0,
    interior_y_margin: float = 0.0,
    singularity_radius: float = 0.0,
    right_boundary_focus_fraction: float = 0.0,
    right_boundary_focus_width: float = 0.0,
):
    """Sample both layers independently and include an explicit interface set.

    This replaces the accidental 30x30 tensor-product cloud (900 points) with
    a genuine two-dimensional sample.  A fixed seed makes validation and the
    L-BFGS objective deterministic; Adam may call this function with changing
    seeds for controlled resampling.
    """
    if min(n_interior_per_layer, n_boundary, n_interface) <= 0:
        raise ValueError("all sample counts must be positive")
    if not 0.0 <= interior_x_margin < 0.5 * WIDTH:
        raise ValueError("interior_x_margin must lie in [0, WIDTH/2)")
    if not 0.0 <= interior_y_margin < 0.25 * HEIGHT:
        raise ValueError("interior_y_margin must lie in [0, HEIGHT/4)")
    if not 0.0 <= singularity_radius < 0.1 * WIDTH:
        raise ValueError("singularity_radius must lie in [0, WIDTH/10)")
    if not 0.0 <= right_boundary_focus_fraction < 1.0:
        raise ValueError("right_boundary_focus_fraction must lie in [0, 1)")
    if right_boundary_focus_fraction > 0.0:
        if not 0.0 < right_boundary_focus_width <= HEIGHT:
            raise ValueError(
                "right_boundary_focus_width must lie in (0, HEIGHT]"
            )

    rng = np.random.default_rng(seed)
    margin = 1e-8 * HEIGHT

    def interior(y0, y1):
        accepted = []
        remaining = n_interior_per_layer
        singular_points = np.array([
            [0.0, 0.0], [0.0, CENTER], [0.0, HEIGHT],
            [WIDTH, 0.0], [WIDTH, CENTER], [WIDTH, HEIGHT],
        ])
        while remaining:
            count = max(remaining * 2, 64)
            candidates = np.column_stack((
                rng.uniform(
                    interior_x_margin, WIDTH - interior_x_margin, count,
                ),
                rng.uniform(
                    y0 + interior_y_margin, y1 - interior_y_margin, count,
                ),
            ))
            if singularity_radius > 0.0:
                distance_squared = np.sum(
                    (candidates[:, None, :] - singular_points[None, :, :]) ** 2,
                    axis=2,
                )
                candidates = candidates[
                    np.all(distance_squared >= singularity_radius**2, axis=1)
                ]
            take = candidates[:remaining]
            accepted.append(take)
            remaining -= len(take)
        return np.vstack(accepted)

    bottom = interior(0.0 + margin, CENTER - margin)
    top = interior(CENTER + margin, HEIGHT - margin)
    xy = np.vstack((bottom, top))
    coefficients = build_coefficients(xy)

    xb = rng.uniform(0.0, WIDTH, n_boundary)
    n_focused = int(round(right_boundary_focus_fraction * n_boundary))
    n_uniform = n_boundary - n_focused
    yb = rng.uniform(0.0, HEIGHT, n_uniform)
    if n_focused:
        focused_lower = max(0.0, HEIGHT - right_boundary_focus_width)
        yb = np.concatenate((
            yb,
            rng.uniform(focused_lower, HEIGHT, n_focused),
        ))
        rng.shuffle(yb)
    xy_top = np.column_stack((xb, np.full(n_boundary, HEIGHT)))
    xy_bottom = np.column_stack((xb, np.zeros(n_boundary)))
    xy_right = np.column_stack((np.full(n_boundary, WIDTH), yb))
    xy_left = np.column_stack((np.zeros(n_boundary), yb))
    xy_interface = np.column_stack((
        rng.uniform(0.0, WIDTH, n_interface),
        np.full(n_interface, CENTER),
    ))

    def tensor(values, *, grad=False):
        return torch.tensor(values, dtype=dtype, device=device,
                            requires_grad=grad)

    return {
        "xy_top": tensor(xy_top),
        "xy_bottom": tensor(xy_bottom),
        "xy_right": tensor(xy_right),
        "xy_left": tensor(xy_left),
        "xy_interface": tensor(xy_interface),
        "x_collocation": tensor(xy[:, 0:1], grad=True),
        "y_collocation": tensor(xy[:, 1:2], grad=True),
        "coefficients": tensor(coefficients),
    }
