# PINNs for Piezoelectricity

Source code for the paper *"Physics-Informed Neural Networks applied to
a 2D piezoelectric beam"*. Two formulations are implemented:

* **Indirect (voltage-driven).** A potential difference is imposed
  between the top and bottom electrodes and the beam deformation is the
  output of the network. Lives in
  [src/pinn_piezo/indirect/](src/pinn_piezo/indirect/).
* **Direct (force-driven).** A traction is applied on the right end of
  the beam and the resulting electric potential is recovered. Lives in
  [src/pinn_piezo/direct/](src/pinn_piezo/direct/).

The original development happened in three Jupyter notebooks, kept for
reference under [notebooks/](notebooks/):
[geom_creation.ipynb](notebooks/geom_creation.ipynb),
[PINN_pz_v3.ipynb](notebooks/PINN_pz_v3.ipynb),
[PINN_pz_v3_directo.ipynb](notebooks/PINN_pz_v3_directo.ipynb). The
runnable code now lives in [src/](src/) and [scripts/](scripts/).

## Repository layout

```
src/pinn_piezo/
    config.py            # geometric constants and configurable paths
    materials.py         # piezoelectric material coefficients
    geometry.py          # boundary / collocation point sampling
    plotting.py          # shared matplotlib helpers
    evaluation.py        # FEM ground-truth comparison
    indirect/
        model.py         # FCNPyramid / FCNUniform with hard constraints
        losses.py        # physics + boundary losses (voltage-driven)
        train.py         # Adam + L-BFGS training driver
    direct/
        model.py         # FCN with hard constraints
        losses.py        # physics + boundary losses (force-driven)
        train.py         # Adam (+ optional L-BFGS) training driver

scripts/
    generate_geometry.py # create .npy data files
    train_indirect.py    # train the indirect formulation
    train_direct.py      # train the direct formulation
    evaluate.py          # field plots and FEM comparison
    run_all.py           # full pipeline: data -> train -> evaluate

notebooks/               # original Colab notebooks (reference only)
    geom_creation.ipynb
    PINN_pz_v3.ipynb
    PINN_pz_v3_directo.ipynb

models/                  # paper-quality trained weights (committed)
    indirect/model_PINN_indirect_paper_3.pt
    direct/model_PINN_direct_paper_3.pt

data/                    # .npy / FEM.csv files (generated)
outputs/
    runs/<run_name>/     # one self-contained directory per script invocation
                         #   models/, figures/, checkpoints/, loss_*.npy,
                         #   summary.json (run_all only)
```

All scripts (`run_all`, `train_indirect`, `train_direct`, `evaluate`)
write everything they produce into `outputs/runs/<run_name>/`. The
`--run-name` flag controls the directory name; if omitted, each script
generates one with its own timestamp.

## Installation

Both `uv` (recommended) and a plain `python -m venv` workflow are
supported. Pick one.

### With [`uv`](https://github.com/astral-sh/uv)

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
```

### With `pip` and `venv`

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

> **Note.** The scripts also work without the editable install (each
> entry-point under [scripts/](scripts/) bootstraps `src/` onto
> `sys.path`), so `python -m scripts.run_all` works as long as you run
> it from the repository root.

## Quick start: full pipeline

The fastest path from a clean checkout to a complete set of artefacts
is [scripts/run_all.py](scripts/run_all.py). It generates the geometry
datasets, trains the selected formulation(s), evaluates them on the
test grid (and, optionally, against an FEM ground truth) and bundles
everything into a single timestamped directory under
`outputs/runs/<run_name>/`.

```bash
# Both formulations with the default hyperparameters
python -m scripts.run_all

# Only the indirect PINN, with a shorter Adam stage and a custom run id
python -m scripts.run_all \
    --formulations indirect \
    --epochs-adam-indirect 500 \
    --epochs-lbfgs-indirect 100 \
    --run-name indirect_quick

# Skip training and just regenerate plots from the provided checkpoints
python -m scripts.run_all --use-pretrained --skip-data

# Compare against FEM ground truth
python -m scripts.run_all --fem data/FEM.csv --run-name vs_fem
```

Each `run_all` invocation produces:

```
outputs/runs/<run_name>/
    summary.json                   # metrics, paths, hyperparameters
    loss_<formulation>.npy         # training loss history (per epoch)
    figures/
        loss_<formulation>.png
        <formulation>/
            u_displacement_plot.png
            v_displacement_plot.png
            phi_plot.png
            beam_deformation.png
            (and *_FEM_plot.png / *_error_plot.png if --fem was set)
    models/
        model_PINN_<formulation>.pt
    checkpoints/                   # indirect formulation only
        indirect_ADAM/...
        indirect_LBFGS/...
```

The standalone `train_indirect`, `train_direct` and `evaluate` scripts
use the same `outputs/runs/<run_name>/` convention but only populate the
sub-directories they need (`models/`, `checkpoints/`, `figures/`).

### `run_all.py` flags

| Flag                              | Default                  | Meaning                                                       |
|-----------------------------------|--------------------------|---------------------------------------------------------------|
| `--formulations`                  | `indirect direct`        | Subset of formulations to execute.                            |
| `--run-name NAME`                 | `<UTC timestamp>`        | Identifier for the per-run output directory.                  |
| `--n-points / --n-collocation /`<br>`--n-collocation-test` | `400 / 150 / 200`        | Geometry sample counts.                                       |
| `--skip-data`                     | off                      | Reuse the `.npy` files already in `data/`.                    |
| `--epochs-adam-indirect / --epochs-lbfgs-indirect` | `1000 / 200`         | Indirect-PINN optimiser budget.                               |
| `--epochs-adam-direct / --epochs-lbfgs-direct`     | `3000 / 0`           | Direct-PINN optimiser budget.                                 |
| `--use-pretrained`                | off                      | Skip training and copy the `.pt` files from `models/`.        |
| `--seed N`                        | unset                    | Seed `numpy` and `torch` for reproducibility.                 |
| `--fem PATH`                      | unset                    | Compute L2 errors / error maps against an FEM CSV.            |

## Running the individual steps

If you prefer to drive each stage separately:

```bash
# 1. Generate the .npy geometry / collocation datasets
python -m scripts.generate_geometry           # both suffixes (_m1 and _m1_d)

# 2. Train one (or both) of the PINNs
python -m scripts.train_indirect
python -m scripts.train_direct

# 3. Evaluate against the test grid (and, optionally, FEM data)
python -m scripts.evaluate \
    --formulation indirect \
    --state models/indirect/model_PINN_indirect_paper_3.pt

python -m scripts.evaluate \
    --formulation direct \
    --state models/direct/model_PINN_direct_paper_3.pt \
    --fem data/FEM.csv
```

`scripts/evaluate.py` saves figures by default (use `--no-save-figs`
to disable). Pass `--show` to also pop up interactive windows; without
it the script uses the non-interactive `Agg` backend.

## Pre-trained models

The two `.pt` files under [models/](models/) are **the trained weights
reported in the paper** and are checked into the repository on purpose
(they are not ignored by `.gitignore`):

* [models/indirect/model_PINN_indirect_paper_3.pt](models/indirect/model_PINN_indirect_paper_3.pt)
* [models/direct/model_PINN_direct_paper_3.pt](models/direct/model_PINN_direct_paper_3.pt)

Newly trained models from `train_indirect.py`, `train_direct.py` or
`run_all.py` go into `outputs/runs/<run_name>/models/` instead, so the
paper artefacts are never accidentally overwritten. Use `--use-pretrained`
in `run_all.py` (or pass the paper paths to `--state` in
`evaluate.py`) to reproduce the figures from those weights.

## Configurable paths

Paths can be overridden through environment variables so the same code
runs locally, in CI, or on Colab:

| Variable                  | Default              |
|---------------------------|----------------------|
| `PINN_PIEZO_DATA_DIR`     | `<repo>/data`        |
| `PINN_PIEZO_MODELS_DIR`   | `<repo>/models`      |
| `PINN_PIEZO_OUTPUTS_DIR`  | `<repo>/outputs`     |

