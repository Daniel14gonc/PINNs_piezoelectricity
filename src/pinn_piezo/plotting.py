"""Plotting helpers shared across the two PINN formulations."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


PAPER_RC = {
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
}


def apply_paper_style() -> None:
    plt.rcParams.update(PAPER_RC)


def plot_results(x, y, value, title, filename=None,
                 xlabel='x', ylabel='y',
                 colorbar_label='Value', save=False, save_dir=None,
                 show=True):
    """Scatter colormap used to visualise the PINN/FEM fields."""
    fig = plt.figure(figsize=(7, 2))
    ax = fig.add_subplot(111)

    ax.set_title(title, fontsize=11, fontfamily='serif')
    ax.set_xlabel(xlabel, fontsize=10, fontfamily='serif')
    ax.set_ylabel(ylabel, fontsize=10, fontfamily='serif')

    scatter = ax.scatter(x, y, c=value, cmap='jet', s=30, edgecolors='none')

    ax.tick_params(axis='both', which='major', labelsize=8,
                   direction='in', length=4, width=0.8)
    for side in ('top', 'right', 'bottom', 'left'):
        ax.spines[side].set_linewidth(0.8)

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(colorbar_label, fontsize=10, fontfamily='serif')
    cbar.ax.tick_params(labelsize=8)

    if save and filename and save_dir is not None:
        from pathlib import Path
        out = Path(save_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, format='png', dpi=300, bbox_inches='tight')
        print(f"Figure saved at {out}")

    if show:
        plt.show()
    plt.close(fig)


def plot_loss_curve(loss_list, save=False, save_dir=None,
                    filename="loss_curve.png", show=True):
    apply_paper_style()
    loss_array = np.array(loss_list)
    log_loss = np.log(loss_array)

    fig = plt.figure(figsize=(6, 4))
    plt.plot(log_loss, color='navy', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss (log scale)')
    plt.title('Training Loss Curve')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    if save and save_dir is not None:
        from pathlib import Path
        out = Path(save_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, format='png', dpi=300, bbox_inches='tight')
        print(f"Figure saved at {out}")

    if show:
        plt.show()
    plt.close(fig)


def plot_beam_deformation(x_coords, y_coords, u_pred, v_pred,
                          scale_factor=50,
                          ylim=(-0.03, 0.03),
                          save=False, save_dir=None,
                          filename="beam_deformation.png", show=True):
    apply_paper_style()

    x_deformed = x_coords + scale_factor * u_pred
    y_deformed = y_coords + scale_factor * v_pred

    fig = plt.figure(figsize=(6, 4))
    original_color = "#1f77b4"
    deformed_color = "#ff7f0e"

    plt.plot(x_coords, y_coords, color=original_color, linestyle='-',
             linewidth=1.2, marker='o', markersize=1.5, label='Original beam')
    plt.plot(x_deformed, y_deformed, color=deformed_color, linestyle='-',
             linewidth=1.2, marker='o', markersize=1.5,
             label=f'Deformed beam (x{scale_factor} scale)')

    if ylim is not None:
        plt.ylim(*ylim)
    plt.title('2D Beam Deformation')
    plt.xlabel('x coordinate (m)')
    plt.ylabel('y coordinate (m)')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    plt.tight_layout()

    if save and save_dir is not None:
        from pathlib import Path
        out = Path(save_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, format='png', dpi=300, bbox_inches='tight')
        print(f"Figure saved at {out}")

    if show:
        plt.show()
    plt.close(fig)
