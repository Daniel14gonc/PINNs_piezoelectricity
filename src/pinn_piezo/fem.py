"""Finite-element reference solver for the 2-D piezoelectric beam.

Added for the reviewer revision. Reviewers asked for (a) an independent
validation of the *direct* piezoelectric effect (Cluster 5) and (b) a
runtime comparison against FEM (Cluster 6). This module provides a small,
self-contained, pure-Python coupled piezoelectric solver built on
``scikit-fem`` so both can be produced inside a Colab notebook.

Physics
-------
Plane-stress, linear, stress-charge (e-form) piezoelectricity:

    sigma = C^E : eps  -  e^T . E
    D     = e   : eps  +  kappa^S . E ,      E = -grad(phi)

with the Voigt ordering ``[eps_xx, eps_yy, gamma_xy]`` (engineering shear
``gamma_xy = u_y + v_x``). The constitutive matrices ``C``, ``e`` and
``kappa^S`` are assembled from :mod:`pinn_piezo.materials`, i.e. the same
coefficients the PINN is trained on. The beam is a *bimorph*: the two
layers are oppositely poled, so the piezoelectric coupling ``e`` flips
sign across the mid-plane ``y = HEIGHT/2``.

Two boundary-value problems mirror the two PINN formulations:

* ``"indirect"`` (voltage-driven / converse effect): phi = V on the top
  electrode, phi = 0 on the bottom electrode, clamped left edge; the beam
  deforms.
* ``"direct"`` (force-driven / direct effect): a tip traction on the
  right edge, clamped left edge, bottom electrode grounded (phi = 0), top
  electrode open-circuit (charge-free, natural BC); a voltage is
  generated.

The solver returns the displacement/potential at the mesh nodes plus a
``probe`` callable that evaluates the FE solution at arbitrary points, so
results can be compared on the very same grid as the PINN or an external
FEM export.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import materials
from .config import CENTER, HEIGHT, WIDTH


# --- Constitutive matrices (built once from the trained-on coefficients) -----
def constitutive_matrices():
    """Return ``(C, e, kappa)`` in plane-stress Voigt form.

    * ``C``     : 3x3 elastic stiffness ``[xx, yy, xy]``.
    * ``e``     : 2x3 piezo stress matrix (rows = electric x/y).
    * ``kappa`` : 2x2 clamped permittivity ``kappa^S``.
    """
    C = materials.c2d.copy()                       # [[C11,C12,0],[C12,C22,0],[0,0,G]]

    # e-form coupling for a single poling direction. Only the y electric
    # field couples to the normal strains (poling through the thickness):
    #   sigma_xx -= e31 * E_y ,  sigma_yy -= e33 * E_y .
    e31 = materials.pze_E[0, 1]                     # C11*d31 + C12*d33
    e33 = materials.pze_E[1, 1]                     # C12*d31 + C11*d33
    e = np.array([[0.0, 0.0, 0.0],
                  [e31, e33, 0.0]])

    kappa = materials.D_const_strain.copy()         # clamped permittivity (2x2)
    return C, e, kappa


@dataclass
class FEMResult:
    case: str
    points: np.ndarray            # (N, 2) node coordinates
    u: np.ndarray                 # (N,) x-displacement at nodes
    v: np.ndarray                 # (N,) y-displacement at nodes
    phi: np.ndarray               # (N,) electric potential at nodes
    runtime_assemble: float
    runtime_solve: float
    n_dofs: int
    voltage: float | None = None
    force: float | None = None
    probe: Callable[[np.ndarray], dict] | None = field(default=None, repr=False)

    @property
    def runtime_total(self) -> float:
        return self.runtime_assemble + self.runtime_solve


def _structured_tri_mesh(nx: int, ny: int):
    """Structured triangular mesh on ``[0, WIDTH] x [0, HEIGHT]``."""
    from skfem import MeshTri

    xs = np.linspace(0.0, WIDTH, nx + 1)
    ys = np.linspace(0.0, HEIGHT, ny + 1)
    try:
        return MeshTri.init_tensor(xs, ys)
    except AttributeError:  # pragma: no cover - older scikit-fem
        from skfem import MeshQuad
        return MeshQuad.init_tensor(xs, ys).to_meshtri()


def solve_piezo(case: str = "indirect",
                *,
                nx: int = 200,
                ny: int = 8,
                voltage: float = 100.0,
                force: float = 1.0,
                poling_sign: float = 1.0,
                element_order: int = 2,
                eval_points: np.ndarray | None = None):
    """Solve the coupled piezoelectric BVP with the finite-element method.

    Parameters
    ----------
    case : ``"indirect"`` (voltage-driven) or ``"direct"`` (force-driven).
    nx, ny : mesh divisions along the length / thickness. The beam is very
        slender (100:1), so keep ``nx`` large and ``ny`` modest.
    voltage : applied electrode voltage (V), ``"indirect"`` only.
    force : tip force in the -y... actually +y traction resultant (N) on
        the right edge, ``"direct"`` only. Traction = force / HEIGHT.
    poling_sign : flips which layer is poled +/-. Use the cross-check
        against a trusted FEM export to fix its value (default +1).
    element_order : 1 or 2 (P1/P2 triangles). P2 strongly recommended for
        the slender bending beam.
    eval_points : optional ``(M, 2)`` array; the returned ``probe`` is
        also evaluated here for convenience (see ``FEMResult.probe``).
    """
    from skfem import (Basis, ElementTriP1, ElementTriP2, ElementVector,
                       BilinearForm, LinearForm, condense, solve)
    import scipy.sparse as sp

    if case not in ("indirect", "direct"):
        raise ValueError("case must be 'indirect' or 'direct'")

    C, e, kappa = constitutive_matrices()
    C11, C12, C22, G = C[0, 0], C[0, 1], C[1, 1], C[2, 2]
    e31, e33 = e[1, 0], e[1, 1]
    kxx, kyy = kappa[0, 0], kappa[1, 1]

    mesh = _structured_tri_mesh(nx, ny)

    Elem = ElementTriP2 if element_order == 2 else ElementTriP1
    ub = Basis(mesh, ElementVector(Elem()))     # displacement (2 comps)
    pb = Basis(mesh, Elem())                     # electric potential

    def poling(w):
        # +poling_sign in the top layer, -poling_sign in the bottom layer.
        return poling_sign * np.where(w.x[1] > CENTER, 1.0, -1.0)

    # --- Bilinear forms ------------------------------------------------------
    @BilinearForm
    def a_uu(u, v, w):
        exx, eyy = u.grad[0][0], u.grad[1][1]
        gxy = u.grad[0][1] + u.grad[1][0]
        Exx, Eyy = v.grad[0][0], v.grad[1][1]
        Gxy = v.grad[0][1] + v.grad[1][0]
        sxx = C11 * exx + C12 * eyy
        syy = C12 * exx + C22 * eyy
        sxy = G * gxy
        return sxx * Exx + syy * Eyy + sxy * Gxy

    @BilinearForm
    def a_uphi(phi, v, w):
        # trial = phi (scalar), test = v (vector); returns coupling to sigma.
        Exx, Eyy = v.grad[0][0], v.grad[1][1]
        s = poling(w)
        return (Exx * (s * e31) + Eyy * (s * e33)) * phi.grad[1]

    @BilinearForm
    def a_phiphi(phi, psi, w):
        return kxx * phi.grad[0] * psi.grad[0] + kyy * phi.grad[1] * psi.grad[1]

    t0 = time.perf_counter()
    Kuu = a_uu.assemble(ub)
    Kup = a_uphi.assemble(pb, ub)        # shape (ub.N, pb.N)
    Kpp = a_phiphi.assemble(pb)

    Nu, Np = ub.N, pb.N
    # Symmetric indefinite block system:
    #   [ Kuu    Kup ] [U]   [F]
    #   [ Kup^T -Kpp ] [P] = [Q]
    K = sp.bmat([[Kuu, Kup], [Kup.T, -Kpp]], format="csr")
    b = np.zeros(Nu + Np)

    # --- Right-hand side: tip traction (direct case) ------------------------
    if case == "direct":
        traction_y = force / HEIGHT
        right = mesh.facets_satisfying(lambda x: np.abs(x[0] - WIDTH) < 1e-12)
        fb = ub.boundary(facets=right)

        @LinearForm
        def tip_load(v, w):
            return traction_y * v[1]

        Fu = tip_load.assemble(fb)
        b[:Nu] = Fu

    # --- Dirichlet boundary conditions --------------------------------------
    tol = 1e-9
    left = mesh.facets_satisfying(lambda x: np.abs(x[0]) < tol)
    top = mesh.facets_satisfying(lambda x: np.abs(x[1] - HEIGHT) < tol)
    bottom = mesh.facets_satisfying(lambda x: np.abs(x[1]) < tol)

    u_clamp = ub.get_dofs(facets=left)              # u = v = 0 on the left edge
    D_dofs = list(u_clamp.all())
    x_full = np.zeros(Nu + Np)

    if case == "indirect":
        phi_top = pb.get_dofs(facets=top)
        phi_bot = pb.get_dofs(facets=bottom)
        for d in phi_top.all():
            D_dofs.append(Nu + int(d))
            x_full[Nu + int(d)] = voltage
        for d in phi_bot.all():
            D_dofs.append(Nu + int(d))
            x_full[Nu + int(d)] = 0.0
    else:
        # direct: ground the bottom electrode; top is open-circuit (natural).
        phi_bot = pb.get_dofs(facets=bottom)
        for d in phi_bot.all():
            D_dofs.append(Nu + int(d))
            x_full[Nu + int(d)] = 0.0

    D_dofs = np.unique(np.array(D_dofs, dtype=int))
    t1 = time.perf_counter()

    sol = solve(*condense(K, b, x=x_full, D=D_dofs))
    t2 = time.perf_counter()

    U = sol[:Nu]
    P = sol[Nu:]

    # --- Sample displacement & potential at the mesh vertices ---------------
    # ``nodal_dofs[c]`` are the dof indices of vector component ``c`` at the
    # mesh vertices, so this reads the nodal displacement directly.
    nodes = mesh.p.T                                # (Nnodes, 2)
    u_nodes = U[ub.nodal_dofs[0]]
    v_nodes = U[ub.nodal_dofs[1]]
    phi_nodes = P[pb.nodal_dofs[0]]

    u_interp = ub.interpolator(U)                   # callable -> (2, M)
    phi_interp = pb.interpolator(P)                 # callable -> (M,)

    def probe(points: np.ndarray) -> dict:
        pts = np.asarray(points, dtype=float).T     # (2, M)
        uv = np.asarray(u_interp(pts))
        return {"u": uv[0], "v": uv[1], "phi": np.asarray(phi_interp(pts))}

    res = FEMResult(
        case=case, points=nodes, u=u_nodes, v=v_nodes, phi=phi_nodes,
        runtime_assemble=t1 - t0, runtime_solve=t2 - t1, n_dofs=Nu + Np,
        voltage=voltage if case == "indirect" else None,
        force=force if case == "direct" else None,
        probe=probe,
    )
    if eval_points is not None:
        res.eval = probe(eval_points)               # type: ignore[attr-defined]
    return res
