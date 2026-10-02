"""Separate publication figures: interior cloud, boundary/interface sets,
evaluation grid. Thickness magnified (beam is 100:1); units in mm."""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

L_MM, H_MM = 100.0, 1.0
HC = H_MM / 2
rng = np.random.default_rng(20260728)

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
})


def new_ax():
    fig, ax = plt.subplots(figsize=(6.5, 1.9))
    ax.set_xlim(-2.5, L_MM + 2.5)
    ax.set_ylim(-0.12, H_MM + 0.12)
    ax.set_yticks([0, HC, H_MM])
    ax.set_yticklabels(["0", "0.5", "1"])
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    return fig, ax


def save(fig, name):
    fig.savefig(f"{name}.png", dpi=400, bbox_inches="tight")
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}.png / .pdf")


# ------------------------------------------------ 1. interior collocation
fig, ax = new_ax()
n_int = 2048
top = np.c_[rng.uniform(0, L_MM, n_int), rng.uniform(HC, H_MM, n_int)]
bot = np.c_[rng.uniform(0, L_MM, n_int), rng.uniform(0, HC, n_int)]
ax.scatter(top[:, 0], top[:, 1], s=1.2, c="#1f77b4", lw=0,
           label=r"upper layer $\Omega^{+}$ (2048)")
ax.scatter(bot[:, 0], bot[:, 1], s=1.2, c="#d62728", lw=0,
           label=r"lower layer $\Omega^{-}$ (2048)")
ax.axhline(HC, color="k", lw=0.7, ls="--")
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2,
          frameon=False, fontsize=8.5, markerscale=5,
          handletextpad=0.2, columnspacing=1.2, borderaxespad=0.0)
save(fig, "fig_interior_collocation")

# ------------------------------------------------ 2. boundary + interface
fig, ax = new_ax()
n_b = 256
xb = lambda: rng.uniform(0, L_MM, n_b)
yb = lambda: rng.uniform(0, H_MM, n_b)
ax.scatter(xb(), np.full(n_b, H_MM), s=4, c="#d62728", lw=0,
           label="top electrode")
ax.scatter(xb(), np.zeros(n_b), s=4, c="#1f77b4", lw=0,
           label="bottom electrode")
ax.scatter(np.zeros(n_b), yb(), s=4, c="#9467bd", lw=0,
           label="clamped end")
ax.scatter(np.full(n_b, L_MM), yb(), s=4, c="#ff7f0e", lw=0,
           label="free end")
ax.scatter(xb(), np.full(n_b, HC), s=4, c="#2ca02c", lw=0,
           label="interface")
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=5,
          frameon=False, fontsize=8.5, markerscale=2.5,
          handletextpad=0.2, columnspacing=0.7, borderaxespad=0.0)
save(fig, "fig_boundary_interface")

# ------------------------------------------------ 3. evaluation grid
fig, ax = new_ax()
gx, gy = np.meshgrid(np.linspace(0, L_MM, 201), np.linspace(0, H_MM, 21))
ax.scatter(gx.ravel(), gy.ravel(), s=0.8, c="k", lw=0)
ax.axhline(HC, color="k", lw=0.7, ls="--")
save(fig, "fig_evaluation_grid")
