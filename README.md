# PINNs for Piezoelectricity

Source code, trained models and metrics for the paper *"Enhanced Multiphysics
Simulation via a Tailored PINN Architecture for Piezoelectric Cantilever Beam
Energy Harvesters"*.

A mixed eight-field PINN (outputs `u, v, φ, σxx, σyy, τxy, Dx, Dy`) is trained
for a two-layer PVDF parallel bimorph cantilever in two settings:

* **Converse effect (voltage-driven).** A potential difference is imposed
  between the electrodes and the beam deformation is predicted
  ([src/pinn_piezo/indirect/](src/pinn_piezo/indirect/)).
* **Direct effect (force-driven).** A tip traction is applied and the
  resulting electric potential is recovered
  ([src/pinn_piezo/direct/](src/pinn_piezo/direct/)).

Every result reported in the paper comes from a single campaign: **20 000 Adam
epochs from random initialization, seed 20260728, float64, no L-BFGS**,
evaluated against a P2 FEM reference (scikit-fem) on a common 201 × 21 grid.

## Repository layout

```
src/pinn_piezo/
    config.py, materials.py   # geometry constants, PVDF coefficients (Tables 1-2)
    geometry.py, scaling.py   # coefficient fields and residual scales
    fem.py                    # P2 FEM reference solver
    evaluation.py, metrics.py, plotting.py
    indirect/                 # converse effect: model, losses, sampling, training
    direct/                   # direct effect: model, losses, training

scripts/
    run_paper20k_campaign.py  # definition and launcher of the 17 paper runs
    build_paper20k_campaign.py# FEM evaluation, tables and figure package
    train_indirect.py, train_direct.py                     # mixed 8-field PINN
    train_indirect_three_field.py, train_direct_three_field.py  # 3-field baseline
    build_results_package.py, evaluate.py, aggregate_stable_*.py  # used by the builder
    figures/                  # Methods figures (sampling sets, FEM mesh)

notebooks/
    paper20k_full_campaign_colab.ipynb  # notebook that ran the paper campaign
    original/                 # first-version notebooks (historical reference only)

results/runs/
    paper20k_<experiment>/    # config.json, models/*.pt, loss histories, metrics.json
    paper20k_summary/         # all_metrics.json, ablation_results.csv, load sweeps
```

## Installation

```bash
pip install -r requirements.txt   # or: uv sync
```

## Reproducing the paper

### 1. Training (17 runs)

The campaign was run on Google Colab (GPU, float64) with
[notebooks/paper20k_full_campaign_colab.ipynb](notebooks/paper20k_full_campaign_colab.ipynb).
The notebook expects a zip of this repository (`src/`, `scripts/`,
`pyproject.toml`, `requirements.txt`) at
`MyDrive/pinn_piezo_colab_bundle.zip`.

The same runs can be launched locally:

```bash
export PYTHONPATH=src:.
export PINN_PIEZO_OUTPUTS_DIR=outputs
python -m scripts.run_paper20k_campaign --list
python -m scripts.run_paper20k_campaign --epochs 20000 --device cuda \
    --dtype float64 --experiments indirect_mixed direct_mixed
```

| Group | Experiments |
|---|---|
| Baselines | `direct_mixed` (0.1 N), `indirect_mixed` (100 V) |
| Mixed vs three-field | `direct_three_field`, `indirect_three_field` |
| Architecture (converse) | `indirect_arch_2x50`, `indirect_arch_4x50`, `indirect_arch_3x100` |
| Activation | `indirect_tanh_3x50` |
| Gradient routing | `indirect_routing_off` |
| Interior points | `indirect_interior_512`, `indirect_interior_1024` |
| Force sweep | `direct_force_0p05N`, `direct_force_0p2N` |
| Voltage sweep | `indirect_voltage_{200,300,400,500}V` |

### 2. Evaluation, tables and figures

The trained models of all 17 runs are included in `results/`. To recompute
the FEM comparison and regenerate every Results figure and table:

```bash
PINN_PIEZO_OUTPUTS_DIR=results PYTHONPATH=src:. \
    python -m scripts.build_paper20k_campaign
```

This writes `results/paper20k_results_package/` (field maps, error maps,
ablation and load-scaling plots, loss curves, `tables/error_metrics.md`) and
refreshes `results/runs/paper20k_summary/`. Re-evaluating on a different
machine reproduces the stored metrics to within 0.1 % relative.

### 3. Methods figures

```bash
cd results && python ../scripts/figures/make_sampling_figures.py   # Figs. 4, 5, 7
python ../scripts/figures/make_fem_mesh_figure.py                   # Fig. 6
```
