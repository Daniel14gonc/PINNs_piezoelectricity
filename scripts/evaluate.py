"""Evaluate a trained PINN model: produce field plots and FEM comparison.

The paper results are evaluated by ``scripts.build_paper20k_campaign``,
which reuses ``_select_model_and_tensorize`` from this module.
"""


from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from pinn_piezo import config as project_config
from pinn_piezo import evaluation, plotting
from pinn_piezo.config import DATA_DIR, REFERENCE_FORCE, RUNS_DIR, get_device


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--formulation", choices=["indirect", "direct"],
                   required=True)
    p.add_argument("--state", type=str, required=True,
                   help="Path to a torch state_dict (.pt) checkpoint.")
    p.add_argument(
        "--phase-enriched", action=argparse.BooleanOptionalAction,
        default=False,
        help="Build the indirect one-network material-phase architecture.",
    )
    p.add_argument(
        "--split-trunks", action=argparse.BooleanOptionalAction,
        default=False,
        help="Build the mixed model with separate primal/flux trunks.",
    )
    p.add_argument(
        "--constitutive-bridge", action=argparse.BooleanOptionalAction,
        default=False,
        help="Checkpoint used the direct mixed constitutive bridge.",
    )
    p.add_argument(
        "--hard-axial-constitutive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Checkpoint reconstructed sigma_xx exactly from u, v and phi."
        ),
    )
    p.add_argument(
        "--bending-basis", action=argparse.BooleanOptionalAction,
        default=False,
        help="Checkpoint used the direct polynomial bending skip.",
    )
    p.add_argument(
        "--beam-kinematics", action=argparse.BooleanOptionalAction,
        default=False,
        help="Checkpoint used the direct slender-beam kinematic lift.",
    )
    p.add_argument(
        "--bridge-correction-limit", type=float, default=0.1,
        help="Relative correction limit used by the constitutive bridge.",
    )
    p.add_argument(
        "--beam-static-lift", action=argparse.BooleanOptionalAction,
        default=False,
        help="Checkpoint used the equilibrated direct stress lift.",
    )
    p.add_argument(
        "--three-field", action="store_true",
        help="Load the primal (u,v,phi) strong-form model.",
    )
    p.add_argument(
        "--interface-enriched", action=argparse.BooleanOptionalAction,
        default=True,
        help="Three-field model used the abs(y/H-0.5) interface feature.",
    )
    p.add_argument("--activation", choices=["tanh", "silu"], default="tanh")
    p.add_argument(
        "--hidden-sizes", type=int, nargs="+", default=[100, 250],
        help="Hidden widths used by an indirect pyramid checkpoint.",
    )
    p.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    p.add_argument("--suffix", type=str,
                   help="Override the dataset suffix. Defaults to _m1 "
                        "(indirect) or _m1_d (direct).")
    p.add_argument("--fem", type=str, default=None,
                   help="FEM reference: path to FEM.csv, or 'internal' to "
                        "solve the aligned in-project FEM benchmark.")
    p.add_argument("--fem-nx", type=int, default=200,
                   help="Internal FEM divisions along the beam (default: 200).")
    p.add_argument("--fem-ny", type=int, default=8,
                   help="Internal FEM divisions through thickness (default: 8).")
    p.add_argument("--voltage", type=float, default=None,
                   help="Indirect case only: target voltage. The complete "
                        "100 V reference prediction is scaled by V/100.")
    p.add_argument("--force", type=float, default=REFERENCE_FORCE,
                   help="Direct case only: tip-force resultant in N.")
    p.add_argument(
        "--normalization-force", type=float, default=None,
        help=(
            "Direct checkpoint's force scale. Defaults to abs(--force); set "
            "this when training used a decoupled --normalization-force."
        ),
    )
    p.add_argument(
        "--direct-electrical-bc",
        choices=["floating_electrode", "insulated"],
        default="floating_electrode",
    )
    p.add_argument("--hard-natural-bcs", action="store_true")
    p.add_argument("--hard-floating-electrode", action="store_true")
    p.add_argument("--slender-warping", action="store_true")
    p.add_argument("--legacy-direct", action="store_true",
                   help="Direct only: use raw-coordinate/raw-output model.")
    p.add_argument(
        "--slender-transverse-scaling", action="store_true",
        help="Indirect only: use the slender transverse units used in training.",
    )
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


def _select_model_and_tensorize(formulation, device, *, phase_enriched=False,
                                activation="tanh", force=REFERENCE_FORCE,
                                normalization_force=None,
                                hard_natural_bcs=False,
                                hard_floating_electrode=False,
                                slender_warping=False, three_field=False,
                                interface_enriched=True,
                                hidden_sizes=(100, 250),
                                split_trunks=False,
                                constitutive_bridge=False,
                                hard_axial_constitutive=False,
                                bending_basis=False,
                                beam_kinematics=False,
                                bridge_correction_limit=0.1,
                                beam_static_lift=False,
                                legacy_direct=False):
    activation_cls = {"tanh": torch.nn.Tanh, "silu": torch.nn.SiLU}[
        activation
    ]
    if formulation == "indirect":
        torch.set_default_dtype(torch.float64)
        from pinn_piezo.indirect.train import tensorize as _t
        dtype = torch.float64
        default_suffix = "_m1"
        if three_field:
            from pinn_piezo.indirect import standard
            model = standard.build_standard_model(
                device=device,
                hidden_sizes=hidden_sizes,
                activation=activation_cls,
                interface_enriched=interface_enriched,
                phase_enriched=phase_enriched,
            ).double()
        else:
            from pinn_piezo.indirect import model as model_mod
            model = model_mod.build_default_model(
                device=device, phase_enriched=phase_enriched,
                hard_natural_bcs=hard_natural_bcs,
                activation=activation_cls, hidden_sizes=hidden_sizes,
                split_trunks=split_trunks,
                constitutive_bridge=constitutive_bridge,
                bridge_correction_limit=bridge_correction_limit,
            )
    else:
        torch.set_default_dtype(torch.float64)
        from pinn_piezo.direct import losses as losses_mod
        from pinn_piezo.direct.train import tensorize as _t
        losses_mod.APPLIED_FORCE_Y = force
        dtype = torch.float64
        default_suffix = "_m1_d"
        if three_field:
            from pinn_piezo.direct import standard
            model = standard.build_standard_model(
                device=device,
                hidden_sizes=hidden_sizes,
                reference_force=force,
                activation=activation_cls,
                interface_enriched=interface_enriched,
                phase_enriched=phase_enriched,
                slender_warping=slender_warping,
                hard_floating_electrode=hard_floating_electrode,
                output_init_gain=0.0,
            ).double()
        else:
            from pinn_piezo.direct import model as model_mod
            model = model_mod.build_default_model(
                device=device,
                hidden_sizes=tuple(hidden_sizes),
                reference_force=force,
                normalization_force=normalization_force,
                legacy=legacy_direct,
                hard_natural_bcs=hard_natural_bcs,
                hard_floating_electrode=hard_floating_electrode,
                slender_warping=slender_warping,
                phase_enriched=phase_enriched,
                split_trunks=split_trunks,
                constitutive_bridge=constitutive_bridge,
                hard_axial_constitutive=hard_axial_constitutive,
                bending_basis=bending_basis,
                beam_kinematics=beam_kinematics,
                bridge_correction_limit=bridge_correction_limit,
                beam_static_lift=beam_static_lift,
                activation=activation_cls,
            ).double()

    def tensorize(x):
        return _t(x, device, dtype=dtype)

    return model, tensorize, default_suffix


def main():
    args = parse_args()

    if (
        args.normalization_force is not None
        and args.normalization_force <= 0.0
    ):
        raise ValueError("--normalization-force must be positive")

    project_config.INDIRECT_SLENDER_TRANSVERSE_SCALING = (
        args.slender_transverse_scaling
    )

    if args.voltage is not None:
        if args.formulation != "indirect":
            raise ValueError("--voltage is only valid for --formulation indirect")
        project_config.VOLTAGE = args.voltage

    if not args.show:
        import matplotlib
        matplotlib.use("Agg")

    device = get_device()
    print(f"Using device: {device}")

    model, tensorize, default_suffix = _select_model_and_tensorize(
        args.formulation, device,
        phase_enriched=args.phase_enriched,
        activation=args.activation,
        force=args.force,
        normalization_force=args.normalization_force,
        hard_natural_bcs=args.hard_natural_bcs,
        hard_floating_electrode=args.hard_floating_electrode,
        slender_warping=args.slender_warping,
        three_field=args.three_field,
        interface_enriched=args.interface_enriched,
        hidden_sizes=tuple(args.hidden_sizes),
        split_trunks=args.split_trunks,
        constitutive_bridge=args.constitutive_bridge,
        hard_axial_constitutive=args.hard_axial_constitutive,
        bending_basis=args.bending_basis,
        beam_kinematics=args.beam_kinematics,
        bridge_correction_limit=args.bridge_correction_limit,
        beam_static_lift=args.beam_static_lift,
        legacy_direct=args.legacy_direct,
    )
    suffix = args.suffix or default_suffix
    data_dir = Path(args.data_dir)

    run_name = args.run_name or (
        f"eval_{args.formulation}_"
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = run_dir / "figures"
    if args.save_figs:
        figures_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving figures to {figures_dir}")

    # --- Load trained weights ----------------------------------------------
    state = torch.load(args.state, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
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
        reference_poling_sign = None
        if args.fem == "internal":
            from pinn_piezo.fem import solve_piezo

            fem_kwargs = dict(
                case=args.formulation,
                nx=args.fem_nx,
                ny=args.fem_ny,
                poling_sign=project_config.CANONICAL_POLING_SIGN,
                element_order=2,
                eval_points=x_test,
            )
            if args.formulation == "indirect":
                fem_kwargs["voltage"] = project_config.VOLTAGE
            else:
                fem_kwargs["force"] = args.force
                fem_kwargs["direct_electrical_bc"] = args.direct_electrical_bc
            reference = solve_piezo(**fem_kwargs)
            X_gt = x_test
            U = reference.eval["u"]
            V = reference.eval["v"]
            Phi = reference.eval["phi"]
            reference_poling_sign = reference.poling_sign
            load_description = (
                f"voltage={reference.voltage:g} V"
                if args.formulation == "indirect"
                else f"force={reference.force:g} N, "
                     f"electrical_bc={reference.direct_electrical_bc}"
            )
            print(
                "Internal FEM reference: "
                f"poling_sign={reference.poling_sign:+.0f}, "
                f"{load_description}, "
                f"mesh={args.fem_nx}x{args.fem_ny} P2"
            )
        else:
            X_gt, U, V, Phi = evaluation.load_FEM_ground_truth(args.fem)
        report = evaluation.evaluate_against_FEM(model, X_gt, U, V, Phi,
                                                 tensorize)

        metrics_payload = {
            "formulation": args.formulation,
            "activation": args.activation,
            "hidden_sizes": (
                args.hidden_sizes
            ),
            "slender_transverse_scaling": (
                args.slender_transverse_scaling
                if args.formulation == "indirect" else None
            ),
            "phase_enriched": args.phase_enriched,
            "split_trunks": args.split_trunks,
            "constitutive_bridge": args.constitutive_bridge,
            "hard_axial_constitutive": args.hard_axial_constitutive,
            "bending_basis": args.bending_basis,
            "beam_kinematics": args.beam_kinematics,
            "bridge_correction_limit": args.bridge_correction_limit,
            "beam_static_lift": args.beam_static_lift,
            "three_field": args.three_field,
            "interface_enriched": (
                args.interface_enriched if args.three_field else None
            ),
            "checkpoint": str(Path(args.state).resolve()),
            "reference": args.fem,
            "voltage_V": (
                project_config.VOLTAGE
                if args.formulation == "indirect" else None
            ),
            "force_N": args.force if args.formulation == "direct" else None,
            "normalization_force_N": (
                (
                    abs(args.force)
                    if args.normalization_force is None
                    else args.normalization_force
                )
                if args.formulation == "direct" else None
            ),
            "direct_electrical_bc": (
                args.direct_electrical_bc
                if args.formulation == "direct" else None
            ),
            "poling_sign": reference_poling_sign,
            "metrics": report["field_metrics"],
        }
        with (run_dir / "metrics.json").open("w", encoding="utf-8") as fh:
            json.dump(metrics_payload, fh, indent=2)
        print(f"Metrics saved to {run_dir / 'metrics.json'}")

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
