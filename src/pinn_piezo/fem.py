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
  right edge, clamped left edge and bottom electrode grounded (phi = 0).
  The paper device uses a conducting top electrode in open circuit, modelled
  as equipotential with zero net free charge.  ``direct_electrical_bc`` can
  select the legacy bare-insulator condition for controlled comparisons.

The solver returns displacement/potential at mesh nodes and ``probe`` and
``probe_flux`` callables for values and element-local stresses/electric
displacements at arbitrary points.  Fluxes use analytic shape derivatives,
without nodal smoothing or finite differences.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import materials
from .config import (CANONICAL_POLING_SIGN, CENTER, HEIGHT, REFERENCE_FORCE,
                     WIDTH)


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


def piezo_coefficients_at(
    points: np.ndarray,
    poling_sign: float = CANONICAL_POLING_SIGN,
    center: float = CENTER,
) -> np.ndarray:
    """Return FEM ``(e31, e33)`` values at physical ``(x, y)`` points.

    This small public helper makes the layer convention directly testable
    against the coefficient columns loaded by the PINN.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    _, e, _ = constitutive_matrices()
    layer_sign = poling_sign * np.where(
        points[:, 1] >= center, 1.0, -1.0,
    )
    return layer_sign[:, None] * e[1, :2][None, :]


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
    direct_electrical_bc: str | None = None
    direct_load: str | None = None
    point_load_y: float | None = None
    poling_sign: float = CANONICAL_POLING_SIGN
    width: float = WIDTH
    height: float = HEIGHT
    probe: Callable[[np.ndarray], dict] | None = field(default=None, repr=False)
    probe_flux: Callable[[np.ndarray], dict] | None = field(default=None, repr=False)
    element_order: int = 2

    @property
    def runtime_total(self) -> float:
        return self.runtime_assemble + self.runtime_solve


def _structured_tri_mesh(nx: int, ny: int, *, width=WIDTH, height=HEIGHT):
    """Structured triangular mesh on a rectangular domain."""
    from skfem import MeshTri

    xs = np.linspace(0.0, width, nx + 1)
    ys = np.linspace(0.0, height, ny + 1)
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
                force: float = REFERENCE_FORCE,
                direct_electrical_bc: str = "floating_electrode",
                direct_load: str = "uniform",
                point_load_y: float | None = None,
                poling_sign: float = CANONICAL_POLING_SIGN,
                element_order: int = 2,
                eval_points: np.ndarray | None = None,
                width: float = WIDTH,
                height: float = HEIGHT):
    """Solve the coupled piezoelectric BVP with the finite-element method.

    Parameters
    ----------
    case : ``"indirect"`` (voltage-driven) or ``"direct"`` (force-driven).
    nx, ny : mesh divisions along the length / thickness. The beam is very
        slender (100:1), so keep ``nx`` large and ``ny`` modest.
    voltage : applied electrode voltage (V), ``"indirect"`` only.
    force : signed vertical tip force (N), ``"direct"`` only. Positive is
        +y and negative is downward.
    direct_electrical_bc : direct-case electrical model.  Use
        ``"floating_electrode"`` for the paper device (constant top potential
        and zero net charge), or ``"insulated"`` for a bare dielectric top
        surface with pointwise ``D.n = 0``.
    direct_load : mechanical loading for the direct case. ``"uniform"``
        distributes ``force`` over the right face. ``"point"`` adds the
        complete force to one displacement DOF on that face, which is the
        finite-element representation of a boundary Dirac load.
    point_load_y : physical y-coordinate of the point force. Defaults to the
        mid-plane. The selected mesh vertex must lie at this coordinate.
    poling_sign : flips which layer is poled +/-.  The default is the project
        convention used by the PINN datasets: ``-e_base`` on the top layer and
        ``+e_base`` on the bottom layer.
    element_order : 1, 2 or 3 (P1/P2/P3 triangles). P2 is the compatibility
        default; P3 improves stress recovery in the slender bending beam.
        Resolve the end regions and check mesh convergence for local stresses.
    eval_points : optional ``(M, 2)`` array; the returned ``probe`` is
        also evaluated here for convenience (see ``FEMResult.probe``).
    """
    from skfem import (Basis, ElementTriP1, ElementTriP2, ElementTriP3,
                       ElementVector, BilinearForm, LinearForm, condense)
    import scipy.sparse as sp
    from scipy.sparse.linalg import splu

    def equilibrated_solve(A, rhs):
        """Solve a badly scaled coupled matrix after symmetric equilibration.

        Piezoelectric blocks mix elastic coefficients near 1e9 with
        permittivities near 1e-10.  Solving that saddle system without algebraic
        scaling can visibly violate load linearity even with a direct solver.
        """
        diagonal = np.abs(A.diagonal())
        if not np.all(np.isfinite(diagonal) & (diagonal > 0.0)):
            raise ValueError("FEM equilibration requires finite nonzero diagonals")
        # Do not floor electric entries using the much larger elastic block:
        # that leaves the dielectric equations poorly scaled.
        scale = 1.0 / np.sqrt(diagonal)
        S = sp.diags(scale)
        scaled_matrix = (S @ A @ S).tocsc()
        scaled_rhs = scale * rhs
        lu = splu(scaled_matrix, options={"Equil": False})
        z = lu.solve(scaled_rhs)
        # Accumulate the refinement residual with extra precision where the
        # platform provides it; cancellation is severe in slender bending.
        residual_matrix = scaled_matrix.astype(np.longdouble)
        residual_rhs = scaled_rhs.astype(np.longdouble)
        for _ in range(3):
            residual = residual_rhs - residual_matrix @ z.astype(np.longdouble)
            z += lu.solve(np.asarray(residual, dtype=float))
        return scale * z

    if case not in ("indirect", "direct"):
        raise ValueError("case must be 'indirect' or 'direct'")
    if width <= 0.0 or height <= 0.0:
        raise ValueError("width and height must be positive")
    if element_order not in (1, 2, 3):
        raise ValueError("element_order must be 1, 2 or 3")
    if direct_electrical_bc not in ("floating_electrode", "insulated"):
        raise ValueError(
            "direct_electrical_bc must be 'floating_electrode' or 'insulated'"
        )
    if direct_load not in ("uniform", "point"):
        raise ValueError("direct_load must be 'uniform' or 'point'")
    if point_load_y is None:
        point_load_y = 0.5 * height
    if not 0.0 <= point_load_y <= height:
        raise ValueError("point_load_y must lie in [0, height]")

    C, e, kappa = constitutive_matrices()
    C11, C12, C22, G = C[0, 0], C[0, 1], C[1, 1], C[2, 2]
    e31, e33 = e[1, 0], e[1, 1]
    kxx, kyy = kappa[0, 0], kappa[1, 1]

    center = 0.5 * height
    mesh = _structured_tri_mesh(nx, ny, width=width, height=height)

    Elem = {1: ElementTriP1, 2: ElementTriP2, 3: ElementTriP3}[element_order]
    ub = Basis(mesh, ElementVector(Elem()))     # displacement (2 comps)
    pb = Basis(mesh, Elem())                     # electric potential

    def poling(w):
        # +poling_sign in the top layer, -poling_sign in the bottom layer.
        return poling_sign * np.where(w.x[1] > center, 1.0, -1.0)

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
        if direct_load == "uniform":
            traction_y = force / height
            right = mesh.facets_satisfying(
                lambda x: np.abs(x[0] - width) < 1e-12
            )
            fb = ub.boundary(facets=right)

            @LinearForm
            def tip_load(v, w):
                return traction_y * v[1]

            Fu = tip_load.assemble(fb)
            b[:Nu] = Fu
        else:
            target = np.array([width, point_load_y])
            distances = np.linalg.norm(mesh.p.T - target, axis=1)
            vertex = int(np.argmin(distances))
            tolerance = 1e-12 * max(1.0, width, height)
            if distances[vertex] > tolerance:
                raise ValueError(
                    "point_load_y must coincide with a mesh vertex; "
                    "choose a compatible ny"
                )
            vertical_dof = int(ub.nodal_dofs[1, vertex])
            b[vertical_dof] += force

    # --- Dirichlet boundary conditions --------------------------------------
    tol = 1e-9
    left = mesh.facets_satisfying(lambda x: np.abs(x[0]) < tol)
    top = mesh.facets_satisfying(lambda x: np.abs(x[1] - height) < tol)
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
        phi_bot = pb.get_dofs(facets=bottom)
        for d in phi_bot.all():
            D_dofs.append(Nu + int(d))
            x_full[Nu + int(d)] = 0.0

    D_dofs = np.unique(np.array(D_dofs, dtype=int))
    t1 = time.perf_counter()

    if case == "direct" and direct_electrical_bc == "floating_electrode":
        # Tie every top-electrode potential DOF to one unknown.  The equation
        # associated with that shared unknown is the sum of the top electrical
        # equations, i.e. the zero-net-free-charge condition.  All prescribed
        # values in the direct problem are zero, so a sparse Boolean transform
        # is sufficient: full_solution = T @ reduced_solution.
        top_global = Nu + pb.get_dofs(facets=top).all().astype(int)
        fixed = set(D_dofs.tolist())
        top_set = set(int(d) for d in top_global if int(d) not in fixed)
        regular = [
            i for i in range(Nu + Np) if i not in fixed and i not in top_set
        ]
        rows = regular + sorted(top_set)
        cols = list(range(len(regular))) + [len(regular)] * len(top_set)
        T = sp.coo_matrix(
            (np.ones(len(rows)), (rows, cols)),
            shape=(Nu + Np, len(regular) + 1),
        ).tocsr()
        K_reduced = T.T @ K @ T
        b_reduced = T.T @ b
        sol = np.asarray(T @ equilibrated_solve(K_reduced, b_reduced)).reshape(-1)
    else:
        # In the insulated direct case, leaving the top electrical DOFs free
        # gives D.n=0 as the natural weak-form boundary condition.
        A, rhs, sol, free = condense(K, b, x=x_full, D=D_dofs)
        sol[free] = equilibrated_solve(A, rhs)
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

    def probe_flux(points: np.ndarray) -> dict:
        """Return sigma_xx, sigma_yy, tau_xy [Pa] and D_x, D_y [C/m²].

        Uses the total stress, including piezoelectric coupling. At an element
        edge the value is a one-sided trace from the element selected by the
        mesh finder. At the material interface its material is used as well;
        sample just above/below the interface to request particular traces.
        Forces integrated through y assume out-of-plane thickness 1 m.
        """
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must have shape (N, 2)")
        values = np.empty((len(points), 5))
        find = mesh.element_finder(mapping=pb.mapping)
        for start in range(0, len(points), 1024):
            xy = points[start:start + 1024].T
            cells = find(*xy)
            ref = pb.mapping.invF(xy[:, :, None], tind=cells)
            ug = np.zeros((2, 2, len(cells)))
            pg = np.zeros((2, len(cells)))
            for basis, coefficients, gradient in ((ub, U, ug), (pb, P, pg)):
                for k in range(basis.Nbfun):
                    shape = basis.elem.gbasis(basis.mapping, ref, k, tind=cells)[0]
                    coef = coefficients[basis.element_dofs[k, cells]]
                    gradient += shape.grad[..., 0] * coef
            cell_y = mesh.p[1, mesh.t[:, cells]].mean(axis=0)
            material_y = np.where(xy[1] == center, cell_y, xy[1])
            sign = poling_sign * np.where(material_y >= center, 1.0, -1.0)
            local_e31, local_e33 = sign * e31, sign * e33
            ux, uy, vx, vy = ug[0, 0], ug[0, 1], ug[1, 0], ug[1, 1]
            px, py = pg
            values[start:start + len(cells)] = np.column_stack((
                C11 * ux + C12 * vy + local_e31 * py,
                C12 * ux + C22 * vy + local_e33 * py,
                G * (uy + vx),
                -kxx * px,
                local_e31 * ux + local_e33 * vy - kyy * py,
            ))
        return dict(zip(("sigma_xx", "sigma_yy", "tau_xy", "D_x", "D_y"), values.T))

    res = FEMResult(
        case=case, points=nodes, u=u_nodes, v=v_nodes, phi=phi_nodes,
        runtime_assemble=t1 - t0, runtime_solve=t2 - t1, n_dofs=int(Nu + Np),
        voltage=voltage if case == "indirect" else None,
        force=force if case == "direct" else None,
        direct_electrical_bc=(direct_electrical_bc if case == "direct" else None),
        direct_load=(direct_load if case == "direct" else None),
        point_load_y=(
            float(point_load_y)
            if case == "direct" and direct_load == "point"
            else None
        ),
        poling_sign=float(poling_sign),
        width=float(width), height=float(height),
        probe=probe, probe_flux=probe_flux, element_order=element_order,
    )
    if eval_points is not None:
        res.eval = probe(eval_points)               # type: ignore[attr-defined]
    return res
