"""Train the direct (force-driven) PINN.

Direct counterpart of ``PINN_pz_v3_directo.ipynb``.

Usage:
    python -m scripts.train_direct
    python -m scripts.train_direct --epochs-adam 3000
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import numpy as np
import torch
from torchsummary import summary

from pinn_piezo.config import DATA_DIR, REFERENCE_FORCE, RUNS_DIR, get_device
from pinn_piezo.direct import model as model_mod
from pinn_piezo.direct import losses as losses_mod
from pinn_piezo.direct import train as train_mod
from pinn_piezo.indirect.sampling import sample_training_tensors


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs-adam", type=int, default=4000)
    p.add_argument("--epochs-lbfgs", type=int, default=100)
    p.add_argument("--lr-adam", type=float, default=1e-3)
    p.add_argument("--epochs-bc-warmup", type=int, default=0)
    p.add_argument("--lr-bc-warmup", type=float, default=1e-3)
    p.add_argument("--lr-lbfgs", type=float, default=1.0)
    p.add_argument("--activation", choices=["tanh", "silu"], default="silu")
    p.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[50, 50, 50],
        help="Hidden-layer widths (validated default: 50 50 50).",
    )
    p.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu",
    )
    p.add_argument(
        "--dtype", choices=["float64", "float32"], default="float64",
        help="Training precision. Apple MPS requires float32.",
    )
    p.add_argument(
        "--output-init-gain", type=float, default=1.0,
        help="Xavier gain for the final layer; 0 preserves zero initialization.",
    )
    p.add_argument("--fraction", type=float, default=0.75)
    p.add_argument(
        "--random-collocation", action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample both layers and boundaries instead of the legacy arrays.",
    )
    p.add_argument("--interior-per-layer", type=int, default=2048)
    p.add_argument("--boundary-points", type=int, default=256)
    p.add_argument("--interface-points", type=int, default=256)
    p.add_argument("--lbfgs-interior-per-layer", type=int, default=2048)
    p.add_argument("--lbfgs-boundary-points", type=int, default=256)
    p.add_argument("--lbfgs-interface-points", type=int, default=256)
    p.add_argument("--resample-every", type=int, default=200)
    p.add_argument("--force", type=float, default=-REFERENCE_FORCE,
                   help="Tip-force resultant in N (paper benchmark: 0.1 N).")
    p.add_argument(
        "--normalization-force", type=float, default=None,
        help=(
            "Positive force magnitude used only for output/residual scales. "
            "Defaults to abs(--force). Keeping 0.1 while training at 1 N is "
            "a controlled load-signal experiment; the physical BVP remains "
            "set by --force."
        ),
    )
    p.add_argument(
        "--traction-profile", choices=["uniform", "parabolic", "point"],
        default="uniform",
        help=(
            "Right-face traction distribution with the prescribed resultant. "
            "'point' uses a normalized half-Gaussian at (L,H)."
        ),
    )
    p.add_argument(
        "--point-load-width-ratio", type=float, default=1.0 / 16.0,
        help=(
            "Regularization width epsilon/H for --traction-profile point. "
            "Its right-face integral remains exactly --force."
        ),
    )
    p.add_argument(
        "--point-load-focus-fraction", type=float, default=0.5,
        help=(
            "Fraction of right-face samples placed inside four point-load "
            "widths of y=H; used only with random point-load collocation."
        ),
    )
    p.add_argument("--electrical-bc",
                   choices=["floating_electrode", "insulated"],
                   default="floating_electrode")
    p.add_argument(
        "--hard-natural-bcs", action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--hard-floating-electrode", action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--slender-warping", action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--phase-enriched", action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--split-trunks", action=argparse.BooleanOptionalAction,
        default=False,
        help="Use separate hidden trunks for (u,v,phi) and the five fluxes.",
    )
    p.add_argument(
        "--constitutive-bridge", action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Make the flux trunk a correction around the constitutive flux. "
            "This gives a cold force-driven run a direct gradient path into "
            "the (u,v,phi) trunk; requires --split-trunks and soft tractions."
        ),
    )
    p.add_argument(
        "--hard-axial-constitutive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Set sigma_xx identically to its constitutive value reconstructed "
            "from u, v and phi; the other four fluxes remain independent."
        ),
    )
    p.add_argument(
        "--bending-basis", action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add a zero-initialized polynomial bending skip to the primary "
            "trunk; requires --split-trunks."
        ),
    )
    p.add_argument(
        "--beam-kinematics", action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Interpret the first two primary outputs as section rotation and "
            "centerline deflection with a shear-warped slender-beam lift."
        ),
    )
    p.add_argument(
        "--bridge-correction-limit", type=float, default=0.1,
        help="Maximum relative flux correction used by --constitutive-bridge.",
    )
    p.add_argument(
        "--beam-static-lift", action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Lift the direct mixed stresses by the equilibrated Saint-Venant "
            "field for the selected force; requires parabolic traction."
        ),
    )
    p.add_argument("--legacy", action="store_true",
                   help="Use raw physical coordinates and raw network outputs.")
    p.add_argument("--unscaled-loss", action="store_true",
                   help="Do not divide PDE or BC residuals by characteristic units.")
    p.add_argument("--balance-weights", action="store_true")
    p.add_argument(
        "--adjust-weights", action="store_true",
        help=(
            "Use the original two-family gradient balance between the complete "
            "PDE and BC losses every --f epochs."
        ),
    )
    p.add_argument("--balance-every", type=int, default=100)
    p.add_argument("--balance-rate", type=float, default=0.15)
    p.add_argument(
        "--equation-minimax", action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Adapt fixed-sum weights across the eight pointwise strong-form "
            "PDE MSE terms during Adam, then freeze them for L-BFGS."
        ),
    )
    p.add_argument("--equation-weight-lr", type=float, default=1e-2)
    p.add_argument("--equation-weight-entropy", type=float, default=0.05)
    p.add_argument(
        "--pcgrad", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "Project conflicting gradients among constitutive, balance, and "
            "boundary/interface loss families during Adam. The scalar loss, "
            "PDEs, BCs, and subsequent L-BFGS objective remain unchanged."
        ),
    )
    p.add_argument("--pde-weight", type=float, default=1.0)
    p.add_argument("--bc-weight", type=float, default=10.0)
    p.add_argument("--bc-stress-weight", type=float, default=10.0)
    p.add_argument("--bc-electric-weight", type=float, default=10.0)
    p.add_argument("--interface-weight", type=float, default=1.0)
    p.add_argument(
        "--constitutive-traction-bcs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also impose the physical Neumann tractions on stresses "
            "reconstructed from (u,v,phi), giving the load a direct primal "
            "gradient path while retaining all eight strong-form outputs."
        ),
    )
    p.add_argument("--drop-shear-constitutive", action="store_true")
    p.add_argument("--midplane-shear-constitutive", action="store_true")
    p.add_argument(
        "--constitutive-weight", type=float, default=1.0,
        help=(
            "Common multiplier for the five normalized constitutive MSEs; "
            "equilibrium and Gauss retain unit weight."
        ),
    )
    p.add_argument(
        "--normal-constitutive-weight", type=float, default=1.0,
        help="Fixed multiplier for sigma_xx/sigma_yy constitutive MSE terms.",
    )
    p.add_argument(
        "--strict-transverse-constitutive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Normalize the sigma_yy constitutive residual by its own "
            "transverse-stress scale instead of the axial bending scale."
        ),
    )
    p.add_argument(
        "--constitutive-normalization",
        choices=("represented", "dominant"), default="dominant",
        help=(
            "How each constitutive residual is normalized. 'represented' "
            "divides by the scale of the quantity the equation represents "
            "(historical). 'dominant' divides by the largest term of the "
            "equation, which removes the ~2.1e4 over-weighting of the shear "
            "law and the ~174 over-weighting of the strict transverse law in "
            "a 100:1 beam. Same equations; only the constants differ."
        ),
    )
    p.add_argument(
        "--section-equilibrium-weight", type=float, default=0.0,
        help=(
            "Weight for the thickness-integrated equilibrium residuals "
            "dM/dx - Q = 0 and dQ/dx = 0.  Both targets are zero, so no known "
            "moment or shear distribution is injected; the terms are redundant "
            "for the exact solution.  They remove the spurious minimum in which "
            "the tip traction is absorbed by a thin boundary layer and the load "
            "never propagates along the beam.  0 disables."
        ),
    )
    p.add_argument(
        "--section-shear-weight", type=float, default=0.0,
        help=(
            "Weight for the thickness-integrated shear constitutive law "
            "int (tau - G(u_y+v_x)) dy = 0.  Same equation as the pointwise "
            "term, but its integrals are value-level quantities, so it "
            "determines v_x with unit sensitivity instead of ~2e4.  0 disables."
        ),
    )
    p.add_argument("--stress-warmup-epochs", type=int, default=0)
    p.add_argument(
        "--freeze-flux-epochs", type=int, default=0,
        help=(
            "For a split-trunk model, initially hold the complete stress/D "
            "branch fixed so (u,v,phi) must fit the imposed loaded flux field."
        ),
    )
    p.add_argument("--stress-warmup-lbfgs", action="store_true")
    p.add_argument(
        "--lbfgs-grad-clip", type=float, default=0.0,
        help="L-BFGS gradient-norm cap; set 0 to preserve strong-Wolfe gradients.",
    )
    p.add_argument("--f", type=int, default=200)
    p.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    p.add_argument("--suffix", type=str, default="_m1_d")
    p.add_argument("--run-name", type=str, default=None,
                   help="Run identifier under outputs/runs/. Defaults to "
                        "'train_direct_<timestamp>'.")
    p.add_argument("--init-state", type=str, default=None)
    p.add_argument(
        "--snapshot-epochs", type=int, nargs="*", default=[],
        help="Completed Adam epochs at which to save exact milestone states.",
    )
    p.add_argument(
        "--periodic-checkpoints", action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the legacy every-100-epoch checkpoints during training.",
    )
    p.add_argument("--seed", type=int, default=20260728)
    return p.parse_args()


def main():
    args = parse_args()

    if sum((
        bool(args.adjust_weights),
        bool(args.balance_weights),
        bool(args.equation_minimax),
        bool(args.pcgrad),
    )) > 1:
        raise ValueError(
            "Select at most one adaptive weighting scheme"
        )
    if args.beam_static_lift and args.traction_profile != "parabolic":
        raise ValueError("--beam-static-lift requires --traction-profile parabolic")
    if not 0.0 < args.point_load_width_ratio <= 1.0:
        raise ValueError("--point-load-width-ratio must lie in (0, 1]")
    if not 0.0 <= args.point_load_focus_fraction < 1.0:
        raise ValueError("--point-load-focus-fraction must lie in [0, 1)")
    if args.constitutive_weight <= 0.0:
        raise ValueError("--constitutive-weight must be positive")
    if (
        args.normalization_force is not None
        and args.normalization_force <= 0.0
    ):
        raise ValueError("--normalization-force must be positive")

    dtype = {
        "float64": torch.float64,
        "float32": torch.float32,
    }[args.dtype]
    if args.device == "mps" and dtype == torch.float64:
        raise ValueError("Apple MPS does not support float64; use --dtype float32")
    torch.set_default_dtype(dtype)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device() if args.device == "auto" else torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    print(f"Using device: {device}")
    print(f"Training dtype: {dtype}")

    run_name = args.run_name or (
        f"train_direct_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    run_dir = RUNS_DIR / run_name
    models_dir = run_dir / "models"
    checkpoints_adam = run_dir / "checkpoints" / "ADAM"
    checkpoints_lbfgs = run_dir / "checkpoints" / "LBFGS"
    snapshots_dir = run_dir / "snapshots"
    for directory in (
        models_dir, checkpoints_adam, checkpoints_lbfgs, snapshots_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")
    with (run_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, indent=2)

    losses_mod.APPLIED_FORCE_Y = args.force
    losses_mod.DIRECT_TRACTION_PROFILE = args.traction_profile
    losses_mod.DIRECT_POINT_LOAD_WIDTH_RATIO = args.point_load_width_ratio
    activation = {"tanh": torch.nn.Tanh, "silu": torch.nn.SiLU}[
        args.activation
    ]
    model = model_mod.build_default_model(
        device=device, reference_force=args.force,
        hidden_sizes=tuple(args.hidden_sizes),
        normalization_force=args.normalization_force,
        legacy=args.legacy,
        hard_natural_bcs=args.hard_natural_bcs,
        hard_floating_electrode=args.hard_floating_electrode,
        slender_warping=args.slender_warping,
        phase_enriched=args.phase_enriched,
        split_trunks=args.split_trunks,
        constitutive_bridge=args.constitutive_bridge,
        hard_axial_constitutive=args.hard_axial_constitutive,
        bending_basis=args.bending_basis,
        beam_kinematics=args.beam_kinematics,
        bridge_correction_limit=args.bridge_correction_limit,
        beam_static_lift=args.beam_static_lift,
        traction_profile=args.traction_profile,
        point_load_width_ratio=args.point_load_width_ratio,
        output_init_gain=args.output_init_gain,
        activation=activation,
    ).to(device=device, dtype=dtype)
    if args.init_state is not None:
        state = torch.load(args.init_state, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        print(f"Initialized model from: {args.init_state}")
    try:
        summary(model, (2,))
    except Exception:
        pass

    resample_fn = None
    lbfgs_tensors = None
    if args.random_collocation:
        def sample_adam(batch_index):
            point_profile = args.traction_profile == "point"
            return sample_training_tensors(
                n_interior_per_layer=args.interior_per_layer,
                n_boundary=args.boundary_points,
                n_interface=args.interface_points,
                seed=args.seed + batch_index,
                device=device,
                dtype=dtype,
                right_boundary_focus_fraction=(
                    args.point_load_focus_fraction if point_profile else 0.0
                ),
                right_boundary_focus_width=(
                    min(
                        args.point_load_width_ratio * 4.0,
                        1.0,
                    ) * float(losses_mod.HEIGHT)
                    if point_profile else 0.0
                ),
            )

        tensors = sample_adam(0)
        resample_fn = sample_adam
        lbfgs_tensors = sample_training_tensors(
            n_interior_per_layer=args.lbfgs_interior_per_layer,
            n_boundary=args.lbfgs_boundary_points,
            n_interface=args.lbfgs_interface_points,
            seed=args.seed + 1_000_000,
            device=device,
            dtype=dtype,
            right_boundary_focus_fraction=(
                args.point_load_focus_fraction
                if args.traction_profile == "point" else 0.0
            ),
            right_boundary_focus_width=(
                min(args.point_load_width_ratio * 4.0, 1.0)
                * float(losses_mod.HEIGHT)
                if args.traction_profile == "point" else 0.0
            ),
        )
    else:
        arrays = train_mod.load_dataset(
            args.data_dir, suffix=args.suffix, fraction=args.fraction,
        )
        tensors = train_mod.to_device(arrays, device, dtype=dtype)
    print("Collocation shapes:",
          tensors["x_collocation"].shape, tensors["y_collocation"].shape)

    loss_weights = {
        "pde": args.pde_weight,
        "bc": args.bc_weight,
    }
    if args.balance_weights:
        loss_weights = {
            "pde": args.pde_weight,
            "bc_stress": args.bc_stress_weight,
            "bc_electric": args.bc_electric_weight,
        }
    result = train_mod.train(
        model, tensors,
        epochs_adam=args.epochs_adam, epochs_lbfgs=args.epochs_lbfgs,
        lr_adam=args.lr_adam, lr_lbfgs=args.lr_lbfgs,
        loss_weights=loss_weights,
        epochs_bc_warmup=args.epochs_bc_warmup,
        lr_bc_warmup=args.lr_bc_warmup,
        f=args.f,
        electrical_mode=args.electrical_bc,
        include_shear_constitutive=not args.drop_shear_constitutive,
        shear_constitutive_mode=(
            "midplane" if args.midplane_shear_constitutive else "full"
        ),
        constitutive_weight=args.constitutive_weight,
        normal_constitutive_weight=args.normal_constitutive_weight,
        strict_transverse_constitutive=args.strict_transverse_constitutive,
        constitutive_normalization=args.constitutive_normalization,
        section_equilibrium_weight=args.section_equilibrium_weight,
        section_shear_weight=args.section_shear_weight,
        stress_warmup_epochs=args.stress_warmup_epochs,
        freeze_flux_epochs=args.freeze_flux_epochs,
        stress_warmup_lbfgs=args.stress_warmup_lbfgs,
        interface_weight=args.interface_weight,
        normalize_residuals=not args.unscaled_loss,
        constitutive_traction_bcs=args.constitutive_traction_bcs,
        adjust_weights=args.adjust_weights,
        balance_weights=args.balance_weights,
        balance_every=args.balance_every,
        balance_rate=args.balance_rate,
        equation_minimax=args.equation_minimax,
        equation_weight_lr=args.equation_weight_lr,
        equation_weight_entropy=args.equation_weight_entropy,
        pcgrad=args.pcgrad,
        lbfgs_grad_clip=args.lbfgs_grad_clip,
        resample_fn=resample_fn,
        resample_every=args.resample_every,
        lbfgs_tensors=lbfgs_tensors,
        checkpoints_adam_dir=(
            checkpoints_adam if args.periodic_checkpoints else None
        ),
        checkpoints_lbfgs_dir=(
            checkpoints_lbfgs if args.periodic_checkpoints else None
        ),
        snapshots_dir=snapshots_dir,
        snapshot_epochs=set(args.snapshot_epochs),
    )

    adam_save_path = models_dir / "model_PINN_direct_adam.pt"
    torch.save(result["adam_state_dict"], adam_save_path)
    print(f"Adam state_dict saved to {adam_save_path}")

    save_path = models_dir / "model_PINN_direct.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model state_dict saved to {save_path}")

    np.save(run_dir / "loss_direct.npy", np.array(result["loss_list"]))
    np.savez(
        run_dir / "loss_components.npz",
        **{
            name: np.asarray(values, dtype=float)
            for name, values in result["loss_components"].items()
        },
    )
    with (run_dir / "training_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump({
            "total_time_seconds": result["total_time"],
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "best_loss_adam": min(
                result["loss_list"][:args.epochs_adam], default=None
            ),
            "best_loss_lbfgs": min(
                result["loss_list"][args.epochs_adam:], default=None
            ),
        }, handle, indent=2)


if __name__ == "__main__":
    main()
