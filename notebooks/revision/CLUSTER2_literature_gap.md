# Cluster 2 — Research gap / literature review (writing task)

**Reviewer comments:** R1.2, R2.3 (and the third reviewer's point 3, which
explicitly requests **≥5 recent (last 3 years) studies, including
DOI: 10.3390/buildings15213960**, and a statement of how this work differs).

This cluster needs **no notebook** — it is an Introduction rewrite. Below is a
ready-to-adapt scaffold. Fill the citation keys with real references from your
bibliography and verify each claim.

---

## New subsection: *Limitations of existing PINN approaches and research gap*

> Physics-informed neural networks have been applied across fluid mechanics
> [refs], structural and solid mechanics [refs], and, more recently,
> multiphysics and engineering-design problems [refs incl.
> DOI:10.3390/buildings15213960]. Despite this progress, existing PINN
> formulations face several limitations when applied to coupled
> electromechanical problems such as piezoelectric energy harvesting:
>
> 1. **High-order derivatives.** Enforcing equilibrium and Gauss's law directly
>    in a displacement/potential formulation requires second-order derivatives
>    of the network outputs, which are expensive to evaluate and degrade
>    training stability [refs].
> 2. **Optimization instability in coupled systems.** Simultaneously satisfying
>    mechanical and electrical residuals of very different scales leads to
>    ill-conditioned, unstable optimization [refs].
> 3. **Beam-specific or energy-based formulations.** Many piezoelectric studies
>    rely on reduced beam theories or the deep energy method (DEM) rather than
>    the full coupled PDE system, limiting generality [refs].
> 4. **Computational cost / surrogate trade-offs.** The cost of training versus
>    a single FEM solve is rarely quantified [refs].
> 5. **Scarcity of piezoelectric PINN studies.** Compared with fluid/elastic
>    applications, PINN treatments of the fully coupled piezoelectric problem
>    — and especially the *direct* effect — remain rare [refs].
>
> **How this work addresses the gap.** We solve the *fully coupled* piezoelectric
> PDE system (not a beam-specific or energy-based reduction) with a standard
> collocation PINN, and make stress and electric displacement explicit network
> outputs so the constitutive and balance laws are enforced with **first-order**
> derivatives only — avoiding the high-order-derivative and stability issues
> above. We treat both the converse (voltage-driven) and the direct
> (force-driven) effects, validate the latter against an independent FEM
> reference, and report runtime and accuracy trade-offs explicitly.

## Novelty table (Cluster 1 / R1.1 / R2.1 / R3.3)

Place this near the end of the Introduction or start of the Methods.

| Feature | Standard PINN | Rao et al. (2020) [33] | **This work** |
|---|---|---|---|
| Coupled piezoelectric PDEs | No | No | **Yes** |
| Explicit stress outputs | No | Yes | **Yes** |
| Explicit dielectric (D) outputs | No | No | **Yes** |
| PDE enforcement order | 2nd order | 1st order (mechanical only) | **1st order, fully coupled electromechanical** |
| Direct + converse effects | — | mechanical only | **Both** |

> The ablation in [`A_architecture_ablation.ipynb`](A_architecture_ablation.ipynb)
> provides quantitative support for the rows of this table (Case A vs Case B).

## Five-recent-references checklist (R3 #3)

- [ ] DOI: 10.3390/buildings15213960 (required by the reviewer)
- [ ] recent PINN in structural mechanics (≤3 yrs)
- [ ] recent PINN in multiphysics / coupled problems (≤3 yrs)
- [ ] recent PINN in computational mechanics (≤3 yrs)
- [ ] recent PINN in engineering design / optimization (≤3 yrs)

For each, add one sentence on what they do and one on how this work differs.
