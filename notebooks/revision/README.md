# Reviewer-revision notebooks

Colab notebooks that produce the results for the reviewer clusters assigned
to this part of the rebuttal. They are thin orchestrators on top of the
`pinn_piezo` package; the reusable logic lives in the package
(`fem.py`, `metrics.py`, `indirect/standard.py`).

| Notebook | Cluster(s) | Reviewer comments | Produces |
|----------|-----------|-------------------|----------|
| [`00_fem_reference.ipynb`](00_fem_reference.ipynb) | 5 (backbone for 6 & 8) | R1.5, R3.4, R1.6, R2.5, R3.2 | Validated FEM solver; **independent direct-effect reference**; FEM runtimes; reference CSVs |
| [`A_architecture_ablation.ipynb`](A_architecture_ablation.ipynb) | 3 | R1.3, R2.2, R3.3 (+ novelty R1.1/R2.1) | Width/depth sweep **and** Case A (conventional, 2nd-order) vs Case B (explicit, 1st-order) |
| [`B_hyperparameters.ipynb`](B_hyperparameters.ipynb) | 4 | R1.4, R3.7 | Collocation-point sweep (5k/10k/22.5k), boundary-point sweep, LR note |
| [`C_metrics_and_loss_curves.ipynb`](C_metrics_and_loss_curves.ipynb) | 7 & 11 | R1.6, R2.6, R2.4 | RMSE/MAE/max/L2 table; PDE/BC/total loss curves; better error maps |
| [`D_efficiency.ipynb`](D_efficiency.ipynb) | 6 | R1.6, R2.5, R3.2 | Honest FEM-vs-PINN runtime/memory table |
| [`E_generalization.ipynb`](E_generalization.ipynb) | 8 | R1.7, R3.5 | FEM load sweep + linearity check + optional retraining |

**Cluster 2** (research gap / literature review) is a *writing* task, not an
experiment — see [`CLUSTER2_literature_gap.md`](CLUSTER2_literature_gap.md).

## How to run on Colab

Each notebook's first cell clones this repo and installs it. **The new code
must be pushed to GitHub first** (the notebooks `git clone` from `main`):

```bash
git add src/pinn_piezo/fem.py src/pinn_piezo/metrics.py \
        src/pinn_piezo/indirect/standard.py src/pinn_piezo/geometry.py \
        src/pinn_piezo/indirect/model.py src/pinn_piezo/direct/losses.py \
        notebooks/revision/
git commit -m "Add reviewer-revision FEM solver, metrics, ablation baseline + notebooks"
git push
```

Then open any notebook in Colab (Runtime → GPU recommended) and run all cells.

### FEM reference

The notebooks need **no external FEM file**. All references come from the
built-in `scikit-fem` solver (`pinn_piezo.fem`), which `00` validates against
the analytical bimorph / cantilever solutions and which reproduces the paper's
reported indirect errors (≈0.19 L2 for `u`/`v`, ≈1e-3 for `phi`) when the PINN
is scored against it.

### Quick vs paper-quality runs

Training epochs default to a short setting (`QUICK = True`). For paper-quality
numbers set `QUICK = False`, or override per run with environment variables:

```python
import os; os.environ['REV_ADAM'] = '1000'; os.environ['REV_LBFGS'] = '200'
```

## ⚠️ Key finding to check first (Cluster 5)

The FEM solver predicts a **generated open-circuit voltage of tens of volts**
for a 1 N tip load (consistent with the analytical estimate
`V ≈ g31·σ·t ≈ 60 V`), whereas the PINN reports `~1e-3 V` — a discrepancy of
several orders of magnitude. Before stating the direct effect is "successfully
simulated", reconcile this in `00_fem_reference.ipynb` (check the applied force
actually used — the code default is `0.1 N`, not `1 N` — and the permittivity
sign/scale in the PINN). This FEM **is** the independent validation R1.5 / R3.4
ask for.

## Regenerating the notebooks

The notebooks are generated from a single script so their content is reviewable
as plain Python:

```bash
python notebooks/revision/_build_notebooks.py
```
