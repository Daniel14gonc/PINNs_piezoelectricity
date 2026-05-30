"""Evaluate a trained PINN model: produce field plots and FEM comparison.

Usage:
    python -m scripts.evaluate --formulation indirect \
        --state models/indirect/model_PINN_indirect_paper_3.pt
    python -m scripts.evaluate --formulation direct \
        --state models/direct/model_PINN_direct_paper_3.pt --fem data/FEM.csv
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from pinn_piezo import evaluation, plotting
from pinn_piezo.config import DATA_DIR, MODELS_DIR, RUNS_DIR, get_device


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--formulation", choices=["indirect", "direct"],
                   required=True)
    p.add_argument("--state", type=str, required=True,
                   help="Path to a torch state_dict (.pt) checkpoint.")
    p.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    p.add_argument("--suffix", type=str,
                   help="Override the dataset suffix. Defaults to _m1 "
                        "(indirect) or _m1_d (direct).")
    p.add_argument("--fem", type=str, default=None,
                   help="Optional path to FEM.csv for L2 ground-truth.")
    p.add_argument("--save-figs", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Write figures to outputs/runs/eval_<id>/figures/ "
                        "(default: on).")
    p.add_argument("--run-name", type=str, default=None,
                   help="Run identifier under outputs/runs/. Defaults to "
                        "'eval_<formulation>_<timestamp>'.")
    p.add_argument("--show", action="store_true",
                   help="Also display the figures interactively.")
    return p.parse_args()


def _select_model_and_tensorize(formulation, device):
    if formulation == "indirect":
        torch.set_default_dtype(torch.float64)
        from pinn_piezo.indirect import model as model_mod
        from pinn_piezo.indirect.train import tensorize as _t
        dtype = torch.float64
        default_suffix = "_m1"
        model = model_mod.build_default_model(device=device)
    else:
        from pinn_piezo.direct import model as model_mod
        from pinn_piezo.direct.train import tensorize as _t
        dtype = torch.float32
        default_suffix = "_m1_d"
        model = model_mod.build_default_model(device=device)

    def tensorize(x):
        return _t(x, device, dtype=dtype)

    return model, tensorize, default_suffix


def main():
    args = parse_args()

    if not args.show:
        import matplotlib
        matplotlib.use("Agg")

    device = get_device()
    print(f"Using device: {device}")

    model, tensorize, default_suffix = _select_model_and_tensorize(
        args.formulation, device,
    )
    suffix = args.suffix or default_suffix
    data_dir = Path(args.data_dir)

    run_name = args.run_name or (
        f"eval_{args.formulation}_"
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    figures_dir = RUNS_DIR / run_name / "figures"
    if args.save_figs:
        figures_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving figures to {figures_dir}")

    # --- Load trained weights ----------------------------------------------
    state = torch.load(args.state, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # --- Visualise on test collocation grid --------------------------------
    x_test = np.load(data_dir / f"x_collocation_test_non_normalized{suffix}.npy")
    x_test = x_test[:, :2]
    x_test_tensor = tensorize(x_test)

    preds = model(x_test_tensor).detach().cpu().numpy()
    u_pred, v_pred, phi_pred = preds[:, 0], preds[:, 1], preds[:, 2]

    x_test = x_test_tensor.detach().cpu().numpy()

    plotting.plot_results(x_test[:, 0], x_test[:, 1], u_pred,
                          title='Deflection in x (u) piezoelectric beam',
                          filename='u_displacement_plot.png',
                          xlabel='x(m)', ylabel='y(m)',
                          colorbar_label='u(m)',
                          save=args.save_figs, save_dir=figures_dir, show=args.show)
    plotting.plot_results(x_test[:, 0], x_test[:, 1], v_pred,
                          title='Deflection in y (v) piezoelectric beam',
                          filename='v_displacement_plot.png',
                          xlabel='x(m)', ylabel='y(m)',
                          colorbar_label='v(m)',
                          save=args.save_figs, save_dir=figures_dir, show=args.show)
    plotting.plot_results(x_test[:, 0], x_test[:, 1], phi_pred,
                          title='Electric potential (phi) piezoelectric beam',
                          filename='phi_plot.png',
                          xlabel='x(m)', ylabel='y(m)',
                          colorbar_label='phi(V)',
                          save=args.save_figs, save_dir=figures_dir, show=args.show)

    plotting.plot_beam_deformation(x_test[:, 0], x_test[:, 1], u_pred, v_pred,
                                   save=args.save_figs, save_dir=figures_dir,
                                   show=args.show)

    # --- FEM comparison ----------------------------------------------------
    if args.fem:
        X_gt, U, V, Phi = evaluation.load_FEM_ground_truth(args.fem)
        report = evaluation.evaluate_against_FEM(model, X_gt, U, V, Phi,
                                                 tensorize)

        u_pred_gr = report['u_pred']
        v_pred_gr = report['v_pred']
        phi_pred_gr = report['phi_pred']

        plotting.plot_results(X_gt[:, 0], X_gt[:, 1], U,
                              title='Deflection in x (u) FEM',
                              filename='u_FEM_plot.png',
                              xlabel='x(m)', ylabel='y(m)',
                              colorbar_label='u(m)',
                              save=args.save_figs, save_dir=figures_dir, show=args.show)
        plotting.plot_results(X_gt[:, 0], X_gt[:, 1], V,
                              title='Deflection in y (v) FEM',
                              filename='v_FEM_plot.png',
                              xlabel='x(m)', ylabel='y(m)',
                              colorbar_label='v(m)',
                              save=args.save_figs, save_dir=figures_dir, show=args.show)
        plotting.plot_results(X_gt[:, 0], X_gt[:, 1], Phi,
                              title='Electric potential (phi) FEM',
                              filename='phi_FEM_plot.png',
                              xlabel='x(m)', ylabel='y(m)',
                              colorbar_label='phi(V)',
                              save=args.save_figs, save_dir=figures_dir, show=args.show)

        eps = 1e-25 if args.formulation == 'indirect' else 0.0
        if args.formulation == 'indirect':
            u_error = np.abs(U - u_pred_gr) / (np.abs(U) + eps)
            v_error = np.abs(V - v_pred_gr) / (np.abs(V) + eps)
            phi_error = np.abs(Phi - phi_pred_gr) / (np.abs(Phi) + eps)
            err_label = 'Relative error'
        else:
            u_error = np.abs(U - u_pred_gr)
            v_error = np.abs(V - v_pred_gr)
            phi_error = np.abs(Phi - phi_pred_gr)
            err_label = 'Absolute error'

        plotting.plot_results(X_gt[:, 0], X_gt[:, 1], u_error,
                              title=f'{err_label} deflection in x (u)',
                              filename='u_error_plot.png',
                              xlabel='x(m)', ylabel='y(m)',
                              colorbar_label=err_label,
                              save=args.save_figs, save_dir=figures_dir, show=args.show)
        plotting.plot_results(X_gt[:, 0], X_gt[:, 1], v_error,
                              title=f'{err_label} deflection in y (v)',
                              filename='v_error_plot.png',
                              xlabel='x(m)', ylabel='y(m)',
                              colorbar_label=err_label,
                              save=args.save_figs, save_dir=figures_dir, show=args.show)
        plotting.plot_results(X_gt[:, 0], X_gt[:, 1], phi_error,
                              title=f'{err_label} electric potential (phi)',
                              filename='phi_error_plot.png',
                              xlabel='x(m)', ylabel='y(m)',
                              colorbar_label=err_label,
                              save=args.save_figs, save_dir=figures_dir, show=args.show)


if __name__ == "__main__":
    main()
