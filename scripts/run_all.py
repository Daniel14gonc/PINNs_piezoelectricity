"""End-to-end pipeline: generate geometry, train, evaluate, save artefacts.

Runs the full simulation pipeline for either or both formulations and
collects every artefact (datasets, trained weights, loss history, field
plots, FEM error plots) under ``outputs/runs/<timestamp>/`` so each run
is self-contained and reproducible.

Usage:
    python -m scripts.run_all                           # both formulations
    python -m scripts.run_all --formulations indirect
    python -m scripts.run_all --skip-data --use-pretrained
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from pinn_piezo import evaluation, geometry, plotting
from pinn_piezo.config import (
    DATA_DIR, FIGURES_DIR, MODELS_DIR, OUTPUTS_DIR, get_device,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--formulations", nargs="+",
                   choices=["indirect", "direct"],
                   default=["indirect", "direct"])
    p.add_argument("--run-name", type=str, default=None,
                   help="Optional run identifier (defaults to a timestamp).")

    # Geometry
    p.add_argument("--n-points", type=int, default=400)
    p.add_argument("--n-collocation", type=int, default=150)
    p.add_argument("--n-collocation-test", type=int, default=200)
    p.add_argument("--skip-data", action="store_true",
                   help="Reuse the .npy files already in data/.")

    # Training
    p.add_argument("--epochs-adam-indirect", type=int, default=1000)
    p.add_argument("--epochs-lbfgs-indirect", type=int, default=200)
    p.add_argument("--epochs-adam-direct", type=int, default=3000)
    p.add_argument("--epochs-lbfgs-direct", type=int, default=0)
    p.add_argument("--use-pretrained", action="store_true",
                   help="Skip training and reuse the .pt files in models/.")
    p.add_argument("--seed", type=int, default=None)

    # Evaluation
    p.add_argument("--fem", type=str, default=None,
                   help="Optional FEM.csv ground-truth (applied to all "
                        "formulations).")
    return p.parse_args()


def _set_seed(seed: int | None):
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_run_dir(name: str | None) -> Path:
    stamp = name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = OUTPUTS_DIR / "runs" / stamp
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    return run_dir


def _generate_data(args, suffixes):
    print(f"\n[1/3] Generating geometry datasets: {suffixes}")
    for suffix in suffixes:
        geometry.generate_and_save(
            n_points=args.n_points,
            n_collocation=args.n_collocation,
            n_collocation_test=args.n_collocation_test,
            suffix=suffix,
            data_dir=DATA_DIR,
        )


def _train_indirect(args, run_dir: Path):
    from pinn_piezo.indirect import model as model_mod
    from pinn_piezo.indirect import train as train_mod

    torch.set_default_dtype(torch.float64)
    device = get_device()
    print(f"  device={device}")

    model = model_mod.build_default_model(device=device)
    arrays = train_mod.load_dataset(DATA_DIR, suffix="_m1", fraction=1.0)
    tensors = train_mod.to_device(arrays, device, dtype=torch.float64)

    ckpt_adam = run_dir / "checkpoints" / "indirect_ADAM"
    ckpt_lbfgs = run_dir / "checkpoints" / "indirect_LBFGS"
    ckpt_adam.mkdir(parents=True, exist_ok=True)
    ckpt_lbfgs.mkdir(parents=True, exist_ok=True)

    result = train_mod.train(
        model, tensors,
        epochs_adam=args.epochs_adam_indirect,
        epochs_lbfgs=args.epochs_lbfgs_indirect,
        checkpoints_adam_dir=ckpt_adam,
        checkpoints_lbfgs_dir=ckpt_lbfgs,
    )

    weights_path = run_dir / "models" / "model_PINN_indirect.pt"
    torch.save(model.state_dict(), weights_path)
    np.save(run_dir / "loss_indirect.npy", np.array(result["loss_list"]))
    plotting.plot_loss_curve(result["loss_list"], save=True,
                             save_dir=run_dir / "figures",
                             filename="loss_indirect.png", show=False)
    return model, weights_path, result


def _train_direct(args, run_dir: Path):
    from pinn_piezo.direct import model as model_mod
    from pinn_piezo.direct import train as train_mod

    torch.set_default_dtype(torch.float32)
    device = get_device()
    print(f"  device={device}")

    model = model_mod.build_default_model(device=device)
    arrays = train_mod.load_dataset(DATA_DIR, suffix="_m1_d", fraction=0.75)
    tensors = train_mod.to_device(arrays, device, dtype=torch.float32)

    result = train_mod.train(
        model, tensors,
        epochs_adam=args.epochs_adam_direct,
        epochs_lbfgs=args.epochs_lbfgs_direct,
    )

    weights_path = run_dir / "models" / "model_PINN_direct.pt"
    torch.save(model.state_dict(), weights_path)
    np.save(run_dir / "loss_direct.npy", np.array(result["loss_list"]))
    plotting.plot_loss_curve(result["loss_list"], save=True,
                             save_dir=run_dir / "figures",
                             filename="loss_direct.png", show=False)
    return model, weights_path, result


def _load_pretrained(formulation: str, run_dir: Path):
    if formulation == "indirect":
        from pinn_piezo.indirect import model as model_mod
        torch.set_default_dtype(torch.float64)
        src = MODELS_DIR / "indirect" / "model_PINN_indirect_paper_3.pt"
    else:
        from pinn_piezo.direct import model as model_mod
        torch.set_default_dtype(torch.float32)
        src = MODELS_DIR / "direct" / "model_PINN_direct_paper_3.pt"

    device = get_device()
    model = model_mod.build_default_model(device=device)
    model.load_state_dict(torch.load(src, map_location=device))

    dst = run_dir / "models" / src.name
    shutil.copy2(src, dst)
    return model, dst


def _evaluate(formulation: str, model, run_dir: Path, fem_csv: str | None):
    if formulation == "indirect":
        from pinn_piezo.indirect.train import tensorize as _t
        dtype = torch.float64
        suffix = "_m1"
        torch.set_default_dtype(torch.float64)
    else:
        from pinn_piezo.direct.train import tensorize as _t
        dtype = torch.float32
        suffix = "_m1_d"

    device = get_device()

    def tensorize(x):
        return _t(x, device, dtype=dtype)

    figures_dir = run_dir / "figures" / formulation
    figures_dir.mkdir(parents=True, exist_ok=True)

    x_test = np.load(DATA_DIR / f"x_collocation_test_non_normalized{suffix}.npy")
    x_test = x_test[:, :2]
    x_test_tensor = tensorize(x_test)
    preds = model(x_test_tensor).detach().cpu().numpy()
    u_pred, v_pred, phi_pred = preds[:, 0], preds[:, 1], preds[:, 2]
    x_test_np = x_test_tensor.detach().cpu().numpy()

    for field, label in [(u_pred, ("u_displacement_plot.png",
                                   "Deflection in x (u) piezoelectric beam",
                                   "u(m)")),
                         (v_pred, ("v_displacement_plot.png",
                                   "Deflection in y (v) piezoelectric beam",
                                   "v(m)")),
                         (phi_pred, ("phi_plot.png",
                                     "Electric potential (phi) piezoelectric beam",
                                     "phi(V)"))]:
        filename, title, cbar = label
        plotting.plot_results(x_test_np[:, 0], x_test_np[:, 1], field,
                              title=title, filename=filename,
                              xlabel='x(m)', ylabel='y(m)',
                              colorbar_label=cbar,
                              save=True, save_dir=figures_dir, show=False)

    plotting.plot_beam_deformation(x_test_np[:, 0], x_test_np[:, 1],
                                   u_pred, v_pred,
                                   save=True, save_dir=figures_dir,
                                   filename="beam_deformation.png", show=False)

    metrics = {}
    if fem_csv:
        X_gt, U, V, Phi = evaluation.load_FEM_ground_truth(fem_csv)
        report = evaluation.evaluate_against_FEM(model, X_gt, U, V, Phi,
                                                 tensorize)
        metrics = {k: float(report[k]) for k in ('l2_u', 'l2_v', 'l2_phi')}

        u_pred_gr = report['u_pred']
        v_pred_gr = report['v_pred']
        phi_pred_gr = report['phi_pred']

        for field, label in [(U, ("u_FEM_plot.png",
                                  "Deflection in x (u) FEM", "u(m)")),
                             (V, ("v_FEM_plot.png",
                                  "Deflection in y (v) FEM", "v(m)")),
                             (Phi, ("phi_FEM_plot.png",
                                    "Electric potential (phi) FEM", "phi(V)"))]:
            filename, title, cbar = label
            plotting.plot_results(X_gt[:, 0], X_gt[:, 1], field,
                                  title=title, filename=filename,
                                  xlabel='x(m)', ylabel='y(m)',
                                  colorbar_label=cbar,
                                  save=True, save_dir=figures_dir, show=False)

        if formulation == 'indirect':
            eps = 1e-25
            u_error = np.abs(U - u_pred_gr) / (np.abs(U) + eps)
            v_error = np.abs(V - v_pred_gr) / (np.abs(V) + eps)
            phi_error = np.abs(Phi - phi_pred_gr) / (np.abs(Phi) + eps)
            err_label = 'Relative error'
        else:
            u_error = np.abs(U - u_pred_gr)
            v_error = np.abs(V - v_pred_gr)
            phi_error = np.abs(Phi - phi_pred_gr)
            err_label = 'Absolute error'

        for field, label in [(u_error, ("u_error_plot.png",
                                        f"{err_label} deflection in x (u)")),
                             (v_error, ("v_error_plot.png",
                                        f"{err_label} deflection in y (v)")),
                             (phi_error, ("phi_error_plot.png",
                                          f"{err_label} electric potential "
                                          "(phi)"))]:
            filename, title = label
            plotting.plot_results(X_gt[:, 0], X_gt[:, 1], field,
                                  title=title, filename=filename,
                                  xlabel='x(m)', ylabel='y(m)',
                                  colorbar_label=err_label,
                                  save=True, save_dir=figures_dir, show=False)

    return metrics


def main():
    args = parse_args()
    _set_seed(args.seed)

    # Force non-interactive matplotlib for the whole pipeline.
    import matplotlib
    matplotlib.use("Agg")

    run_dir = _resolve_run_dir(args.run_name)
    print(f"Run directory: {run_dir}")

    suffixes = []
    if "indirect" in args.formulations:
        suffixes.append("_m1")
    if "direct" in args.formulations:
        suffixes.append("_m1_d")

    started = time.time()

    if not args.skip_data:
        _generate_data(args, suffixes)
    else:
        print("\n[1/3] Skipping geometry generation (--skip-data).")

    print("\n[2/3] Training / loading models.")
    trained = {}
    for formulation in args.formulations:
        print(f"  -> {formulation}")
        if args.use_pretrained:
            model, weights_path = _load_pretrained(formulation, run_dir)
            train_result = None
        elif formulation == "indirect":
            model, weights_path, train_result = _train_indirect(args, run_dir)
        else:
            model, weights_path, train_result = _train_direct(args, run_dir)
        trained[formulation] = (model, weights_path, train_result)

    print("\n[3/3] Evaluating + saving figures.")
    summary = {
        "run_dir": str(run_dir),
        "formulations": args.formulations,
        "use_pretrained": args.use_pretrained,
        "seed": args.seed,
        "fem_csv": args.fem,
        "results": {},
    }
    for formulation, (model, weights_path, train_result) in trained.items():
        print(f"  -> {formulation}")
        metrics = _evaluate(formulation, model, run_dir, args.fem)
        summary["results"][formulation] = {
            "weights": str(weights_path.relative_to(run_dir)),
            "metrics": metrics,
            "training": (
                None if train_result is None
                else {k: (v if not hasattr(v, 'item') else float(v))
                      for k, v in train_result.items() if k != "loss_list"}
            ),
        }

    summary["total_time_seconds"] = time.time() - started
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nDone. Summary at {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
