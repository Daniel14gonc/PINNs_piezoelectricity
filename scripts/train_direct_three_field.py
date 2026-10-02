"""Cold-train the direct strong-form PINN with only ``(u, v, phi)``."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import torch

from pinn_piezo.config import REFERENCE_FORCE, RUNS_DIR, get_device
from pinn_piezo.direct import standard
from pinn_piezo.indirect.sampling import sample_training_tensors


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs-adam", type=int, default=5000)
    parser.add_argument("--epochs-lbfgs", type=int, default=20)
    parser.add_argument("--lr-adam", type=float, default=1e-3)
    parser.add_argument("--lr-lbfgs", type=float, default=1.0)
    parser.add_argument("--activation", choices=("tanh", "silu"), default="tanh")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"),
                        default="auto")
    parser.add_argument("--dtype", choices=("float64", "float32"),
                        default="float64")
    parser.add_argument(
        "--num-threads", type=int, default=1,
        help="Number of PyTorch intra-op CPU threads used for timing.",
    )
    parser.add_argument("--force", type=float, default=REFERENCE_FORCE)
    parser.add_argument("--interior-per-layer", type=int, default=2048)
    parser.add_argument("--boundary-points", type=int, default=512)
    parser.add_argument("--interface-points", type=int, default=512)
    parser.add_argument("--lbfgs-interior-per-layer", type=int, default=4096)
    parser.add_argument("--lbfgs-boundary-points", type=int, default=2048)
    parser.add_argument("--lbfgs-interface-points", type=int, default=2048)
    parser.add_argument("--resample-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[100, 250])
    parser.add_argument("--output-init-gain", type=float, default=1e-4)
    parser.add_argument(
        "--interface-enriched", action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--phase-enriched", action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--slender-warping", action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--hard-floating-electrode",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--periodic-checkpoints", action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the legacy every-100-epoch checkpoints.",
    )
    return parser.parse_args()


def _write_json(path: Path, payload):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main():
    args = parse_args()
    if args.num_threads <= 0:
        raise ValueError("--num-threads must be positive")
    torch.set_num_threads(args.num_threads)
    torch.set_num_interop_threads(1)
    dtype = {"float64": torch.float64, "float32": torch.float32}[args.dtype]
    if args.device == "mps" and dtype == torch.float64:
        raise ValueError("Apple MPS requires --dtype float32")
    torch.set_default_dtype(dtype)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")

    run_name = args.run_name or (
        "train_direct_three_field_"
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    run_dir = RUNS_DIR / run_name
    models_dir = run_dir / "models"
    adam_dir = run_dir / "checkpoints" / "ADAM"
    lbfgs_dir = run_dir / "checkpoints" / "LBFGS"
    for directory in (models_dir, adam_dir, lbfgs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "config.json", vars(args))
    print(
        f"Using device: {device}; dtype: {dtype}; "
        f"threads: {torch.get_num_threads()}"
    )
    print(f"Run directory: {run_dir}")

    activation = {"tanh": torch.nn.Tanh, "silu": torch.nn.SiLU}[
        args.activation
    ]
    model = standard.build_standard_model(
        device=device,
        hidden_sizes=tuple(args.hidden_sizes),
        activation=activation,
        reference_force=args.force,
        interface_enriched=args.interface_enriched,
        phase_enriched=args.phase_enriched,
        slender_warping=args.slender_warping,
        hard_floating_electrode=args.hard_floating_electrode,
        output_init_gain=args.output_init_gain,
    ).to(device=device, dtype=dtype)

    def sample_adam(batch_index):
        return sample_training_tensors(
            n_interior_per_layer=args.interior_per_layer,
            n_boundary=args.boundary_points,
            n_interface=args.interface_points,
            seed=args.seed + batch_index,
            device=device,
            dtype=dtype,
        )

    tensors = sample_adam(0)
    lbfgs_tensors = sample_training_tensors(
        n_interior_per_layer=args.lbfgs_interior_per_layer,
        n_boundary=args.lbfgs_boundary_points,
        n_interface=args.lbfgs_interface_points,
        seed=args.seed + 1_000_000,
        device=device,
        dtype=dtype,
    )
    print(
        "Interior points:", tensors["x_collocation"].shape[0],
        "L-BFGS:", lbfgs_tensors["x_collocation"].shape[0],
    )
    result = standard.train_standard(
        model, tensors,
        epochs_adam=args.epochs_adam,
        epochs_lbfgs=args.epochs_lbfgs,
        lr_adam=args.lr_adam,
        lr_lbfgs=args.lr_lbfgs,
        resample_fn=sample_adam,
        resample_every=args.resample_every,
        lbfgs_tensors=lbfgs_tensors,
        adam_checkpoints_dir=(adam_dir if args.periodic_checkpoints else None),
        lbfgs_checkpoints_dir=(
            lbfgs_dir if args.periodic_checkpoints else None
        ),
    )

    torch.save(
        result["adam_state_dict"],
        models_dir / "model_PINN_direct_three_field_adam.pt",
    )
    torch.save(
        model.state_dict(),
        models_dir / "model_PINN_direct_three_field.pt",
    )
    np.save(run_dir / "loss.npy", np.asarray(result["loss_list"]))
    np.savez(
        run_dir / "loss_components.npz",
        **{
            name: np.asarray(values, dtype=float)
            for name, values in result["loss_components"].items()
        },
    )
    _write_json(run_dir / "final_terms.json", result["final_terms"])
    _write_json(run_dir / "training_summary.json", {
        "total_time_seconds": result["total_time"],
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
    })
    print(f"Final loss: {result['final_terms']['total']:.6e}")
    print(f"Elapsed: {result['total_time']:.1f} s")


if __name__ == "__main__":
    main()
