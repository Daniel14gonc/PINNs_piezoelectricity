"""Train the indirect (voltage-driven) PINN.

Maintained counterpart of ``PINN_pz_v3.ipynb``.

Usage:
    python -m scripts.train_indirect
    python -m scripts.train_indirect --epochs-adam 1000 --epochs-lbfgs 200
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import numpy as np
import torch
from torchsummary import summary

from pinn_piezo import config as project_config
from pinn_piezo.config import DATA_DIR, RUNS_DIR, get_device
from pinn_piezo.indirect import model as model_mod
from pinn_piezo.indirect.sampling import sample_training_tensors
from pinn_piezo.indirect import train as train_mod


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs-adam", type=int, default=8000)
    p.add_argument("--epochs-lbfgs", type=int, default=0)
    p.add_argument("--lr-adam", type=float, default=0.001)
    p.add_argument("--lr-lbfgs", type=float, default=1.0)
    p.add_argument("--fraction", type=float, default=1.0,
                   help="Fraction of collocation points used.")
    p.add_argument("--f", type=int, default=500,
                   help="Adaptive-weight update interval.")
    p.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    p.add_argument("--suffix", type=str, default="_m1")
    p.add_argument("--model-type", choices=["pyramid", "uniform"],
                   default="pyramid")
    p.add_argument(
        "--hidden-sizes", type=int, nargs="+", default=[50, 50, 50],
        help="Hidden widths for the pyramid model.",
    )
    p.add_argument("--activation", choices=["tanh", "silu"], default="silu")
    p.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu",
        help="Training device. CPU float64 is the validated paper recipe.",
    )
    p.add_argument(
        "--dtype", choices=["float64", "float32"], default="float64",
        help="Training precision. Apple MPS requires float32.",
    )
    p.add_argument(
        "--phase-enriched", action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep one 8-output network but add the known +/- material phase "
             "as an input and enforce the six physical interface conditions.",
    )
    p.add_argument(
        "--split-trunks", action=argparse.BooleanOptionalAction,
        default=False,
        help="Use independent hidden trunks for (u,v,phi) and "
             "(sigma_xx,sigma_yy,tau,Dx,Dy), while retaining one 8-output "
             "mixed PINN and the same residual equations.",
    )
    p.add_argument(
        "--constitutive-bridge",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Represent the five public flux fields as bounded corrections "
            "around the constitutive flux reconstructed from (u,v,phi). "
            "This preserves eight outputs and the same PDEs while routing "
            "equilibrium/Gauss/Neumann gradients to the primary fields."
        ),
    )
    p.add_argument(
        "--bridge-correction-limit",
        type=float,
        default=0.1,
        help="Maximum relative mixed-flux correction used by the bridge.",
    )
    p.add_argument(
        "--hard-natural-bcs", action=argparse.BooleanOptionalAction,
        default=False,
        help="Embed free-face stress/insulation factors in the outputs. "
             "Disable to enforce the same physical BCs through MSE terms.",
    )
    p.add_argument("--interior-per-layer", type=int, default=2048)
    p.add_argument("--boundary-points", type=int, default=256)
    p.add_argument("--interface-points", type=int, default=256)
    p.add_argument(
        "--interior-x-margin", type=float, default=0.0,
        help="Exclude this physical distance from x=0,L in PDE sampling; "
             "boundary conditions remain sampled separately.",
    )
    p.add_argument(
        "--interior-y-margin", type=float, default=0.0,
        help="Exclude this distance from each layer edge in PDE sampling.",
    )
    p.add_argument(
        "--singularity-radius", type=float, default=0.0,
        help="Exclude PDE points inside this physical radius around the six "
             "beam corners/interface endpoints; BC points are unchanged.",
    )
    p.add_argument("--lbfgs-interior-per-layer", type=int, default=12000)
    p.add_argument(
        "--lbfgs-boundary-points", type=int, default=None,
        help="Points on each boundary in the fixed L-BFGS cloud. Defaults "
             "to max(--boundary-points, 1000).",
    )
    p.add_argument(
        "--lbfgs-interface-points", type=int, default=None,
        help="Interface points in the fixed L-BFGS cloud. Defaults to "
             "max(--interface-points, 1000).",
    )
    p.add_argument("--resample-every", type=int, default=200)
    p.add_argument("--interface-weight", type=float, default=1.0)
    p.add_argument(
        "--slender-transverse-scaling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Normalize tau~sigma_x H/L and sigma_y~sigma_x(H/L)^2. "
             "This changes units of residuals/outputs, not the PDEs.",
    )
    p.add_argument(
        "--normal-constitutive-weight", type=float, default=1.0,
        help="Fixed multiplier applied only to the sigma_xx and sigma_yy "
             "constitutive MSE terms.",
    )
    p.add_argument(
        "--constitutive-normalization",
        choices=["represented", "dominant"],
        default="represented",
        help=(
            "Normalize mixed constitutive residuals by the represented field "
            "scale (historical) or by the largest term in each equation. "
            "The latter removes slender shear-cancellation locking without "
            "changing the PDE or its full gradient."
        ),
    )
    p.add_argument(
        "--constitutive-traction-bcs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also impose each free-face traction through stresses reconstructed "
            "from (u,v,phi), giving the primal fields a direct Neumann path."
        ),
    )
    p.add_argument(
        "--constitutive-traction-weight", type=float, default=1.0,
        help="Multiplier for the redundant primal constitutive-traction loss.",
    )
    p.add_argument(
        "--constitutive-flux-stopgrad",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep each mixed constitutive residual value unchanged but block "
            "its backward path through sigma,D, routing it toward u,v,phi. "
            "Requires --epochs-lbfgs 0 because the routed gradient is not "
            "compatible with quasi-Newton line search."
        ),
    )
    p.add_argument(
        "--mechanical-constitutive-stopgrad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Route only the three mechanical constitutive equations toward "
            "u,v,phi. The electrical constitutive equations keep their full "
            "gradient through Dx,Dy, avoiding the constant-D null mode. "
            "Requires --epochs-lbfgs 0."
        ),
    )
    p.add_argument(
        "--lbfgs-grad-clip", type=float, default=0.5,
        help="L-BFGS gradient-norm cap. Set 0 to disable clipping.",
    )
    p.add_argument(
        "--stress-warmup-epochs", type=int, default=0,
        help="Initially hold sigma_xx,sigma_yy,tau fixed while u,v,phi,D "
             "learn, then release all eight fields under the same loss.",
    )
    p.add_argument(
        "--stress-warmup-lbfgs", action="store_true",
        help="Keep stress output rows fixed during the L-BFGS phase too.",
    )
    p.add_argument("--seed", type=int, default=20260728)
    p.add_argument("--voltage", type=float,
                   default=project_config.REFERENCE_VOLTAGE,
                   help="Applied voltage. Fields are scaled from the 100 V "
                        "reference solution by linear superposition.")
    p.add_argument("--run-name", type=str, default=None,
                   help="Run identifier under outputs/runs/. Defaults to "
                        "'train_indirect_<timestamp>'.")
    p.add_argument(
        "--init-state", type=str, default=None,
        help="Optional state_dict used to continue a staged training run.",
    )
    p.add_argument(
        "--snapshot-epochs", type=int, nargs="*", default=[],
        help="Completed Adam epochs at which to save exact milestone states.",
    )
    p.add_argument(
        "--periodic-checkpoints", action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the legacy every-100-epoch checkpoints during training.",
    )
    p.add_argument(
        "--zero-output", action=argparse.BooleanOptionalAction, default=False,
        help="Zero the last layer after Xavier initialization (legacy default). "
             "Use --no-zero-output for a fully random cold start.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    project_config.VOLTAGE = args.voltage
    project_config.INDIRECT_SLENDER_TRANSVERSE_SCALING = (
        args.slender_transverse_scaling
    )

    dtype = {
        "float64": torch.float64,
        "float32": torch.float32,
    }[args.dtype]
    if args.device == "mps" and dtype == torch.float64:
        raise ValueError("Apple MPS does not support float64; use --dtype float32")
    if args.constitutive_traction_weight < 0.0:
        raise ValueError("--constitutive-traction-weight must be non-negative")
    if not 0.0 < args.bridge_correction_limit <= 1.0:
        raise ValueError("--bridge-correction-limit must lie in (0, 1]")
    if args.constitutive_bridge and args.split_trunks:
        raise ValueError(
            "--constitutive-bridge currently requires --no-split-trunks"
        )
    if (
        args.constitutive_flux_stopgrad
        and args.mechanical_constitutive_stopgrad
    ):
        raise ValueError(
            "Choose only one constitutive-gradient routing option"
        )
    if (
        args.constitutive_flux_stopgrad
        or args.mechanical_constitutive_stopgrad
    ) and args.epochs_lbfgs > 0:
        raise ValueError(
            "Constitutive gradient routing requires --epochs-lbfgs 0"
        )
    torch.set_default_dtype(dtype)

    device = get_device() if args.device == "auto" else torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    print(f"Using device: {device}")
    print(f"Training dtype: {dtype}")

    run_name = args.run_name or (
        f"train_indirect_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    run_dir = RUNS_DIR / run_name
    ckpt_adam = run_dir / "checkpoints" / "ADAM"
    ckpt_lbfgs = run_dir / "checkpoints" / "LBFGS"
    models_dir = run_dir / "models"
    snapshots_dir = run_dir / "snapshots"
    for d in (ckpt_adam, ckpt_lbfgs, models_dir, snapshots_dir):
        d.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")
    with (run_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, indent=2)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    activation = {"tanh": torch.nn.Tanh, "silu": torch.nn.SiLU}[
        args.activation
    ]
    model = model_mod.build_default_model(
        device=device,
        model_type=args.model_type,
        hidden_sizes=tuple(args.hidden_sizes),
        phase_enriched=args.phase_enriched,
        hard_natural_bcs=args.phase_enriched and args.hard_natural_bcs,
        activation=activation,
        zero_output=args.zero_output,
        split_trunks=args.split_trunks,
        constitutive_bridge=args.constitutive_bridge,
        bridge_correction_limit=args.bridge_correction_limit,
    )
    if args.init_state is not None:
        state = torch.load(args.init_state, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        print(f"Initialized model from: {args.init_state}")
    try:
        summary(model, (2,))
    except Exception:
        # torchsummary on CPU sometimes fails on float64 inputs; non-critical.
        pass

    resample_fn = None
    lbfgs_tensors = None
    if args.phase_enriched:
        def sample_adam(batch_index):
            return sample_training_tensors(
                n_interior_per_layer=args.interior_per_layer,
                n_boundary=args.boundary_points,
                n_interface=args.interface_points,
                seed=args.seed + batch_index,
                device=device,
                dtype=dtype,
                interior_x_margin=args.interior_x_margin,
                interior_y_margin=args.interior_y_margin,
                singularity_radius=args.singularity_radius,
            )

        tensors = sample_adam(0)
        resample_fn = sample_adam
        lbfgs_tensors = sample_training_tensors(
            n_interior_per_layer=args.lbfgs_interior_per_layer,
            n_boundary=(
                args.lbfgs_boundary_points
                if args.lbfgs_boundary_points is not None
                else max(args.boundary_points, 1000)
            ),
            n_interface=(
                args.lbfgs_interface_points
                if args.lbfgs_interface_points is not None
                else max(args.interface_points, 1000)
            ),
            seed=args.seed + 1_000_000,
            device=device,
            dtype=dtype,
            interior_x_margin=args.interior_x_margin,
            interior_y_margin=args.interior_y_margin,
            singularity_radius=args.singularity_radius,
        )
    else:
        arrays = train_mod.load_dataset(args.data_dir,
                                        suffix=args.suffix,
                                        fraction=args.fraction)
        tensors = train_mod.to_device(arrays, device, dtype=dtype)
    print("Collocation shapes:",
          tensors["x_collocation"].shape, tensors["y_collocation"].shape)

    result = train_mod.train(
        model, tensors,
        epochs_adam=args.epochs_adam, epochs_lbfgs=args.epochs_lbfgs,
        lr_adam=args.lr_adam, lr_lbfgs=args.lr_lbfgs,
        f=args.f,
        checkpoints_adam_dir=(ckpt_adam if args.periodic_checkpoints else None),
        checkpoints_lbfgs_dir=(
            ckpt_lbfgs if args.periodic_checkpoints else None
        ),
        resample_fn=resample_fn,
        resample_every=args.resample_every,
        lbfgs_tensors=lbfgs_tensors,
        interface_weight=args.interface_weight,
        stress_warmup_epochs=args.stress_warmup_epochs,
        stress_warmup_lbfgs=args.stress_warmup_lbfgs,
        normal_constitutive_weight=args.normal_constitutive_weight,
        constitutive_normalization=args.constitutive_normalization,
        constitutive_traction_bcs=args.constitutive_traction_bcs,
        constitutive_traction_weight=args.constitutive_traction_weight,
        constitutive_flux_stopgrad=args.constitutive_flux_stopgrad,
        mechanical_constitutive_stopgrad=(
            args.mechanical_constitutive_stopgrad
        ),
        lbfgs_grad_clip=args.lbfgs_grad_clip,
        snapshots_dir=snapshots_dir,
        snapshot_epochs=set(args.snapshot_epochs),
    )

    print("Best LBFGS loss:", result["best_loss_lbfgs"])

    save_path = models_dir / "model_PINN_indirect.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model state_dict saved to {save_path}")

    np.save(run_dir / "loss_indirect.npy", np.array(result["loss_list"]))
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
            "best_loss_adam": result["best_loss_adam"],
            "best_loss_lbfgs": (
                result["best_loss_lbfgs"]
                if args.epochs_lbfgs else None
            ),
        }, handle, indent=2)


if __name__ == "__main__":
    main()
