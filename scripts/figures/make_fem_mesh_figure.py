"""FEM mesh schematic in the same style as the sampling figures.

The actual mesh uses 200 x 8 divisions (single-diagonal structured
triangles, scikit-fem MeshTri.init_tensor); drawing 200 axial divisions
is illegible, so the schematic shows fewer axial divisions and the
caption states the real resolution.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

L_MM, H_MM = 100.0, 1.0
HC = H_MM / 2
NX_SCHEM, NY = 40, 8

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

fig, ax = plt.subplots(figsize=(6.5, 1.9))
xs = np.linspace(0, L_MM, NX_SCHEM + 1)
ys = np.linspace(0, H_MM, NY + 1)

kw = dict(color="#5b7fa6", lw=0.5)
for x in xs:
    ax.plot([x, x], [0, H_MM], **kw)
for y in ys:
    ax.plot([0, L_MM], [y, y], **kw)
for i in range(NX_SCHEM):
    for j in range(NY):
        ax.plot([xs[i], xs[i + 1]], [ys[j], ys[j + 1]], **kw)

ax.axhline(HC, color="k", lw=1.0)

ax.set_xlim(-2.5, L_MM + 2.5)
ax.set_ylim(-0.12, H_MM + 0.12)
ax.set_yticks([0, HC, H_MM])
ax.set_yticklabels(["0", "0.5", "1"])
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")

fig.savefig("fig_fem_mesh.png", dpi=400, bbox_inches="tight")
fig.savefig("fig_fem_mesh.pdf", bbox_inches="tight")
print("saved fig_fem_mesh.png / .pdf")
