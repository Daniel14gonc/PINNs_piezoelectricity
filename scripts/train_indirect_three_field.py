"""Cold-train the indirect strong-form PINN with only (u, v, phi).

Stress and electric displacement are reconstructed from the constitutive law.
No energy functional, pretraining, frozen rows, or checkpoint initialization is
used.  The default is SiLU with Adam followed by deterministic strong-Wolfe
L-BFGS.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import torch

from pinn_piezo import config as project_config
from pinn_piezo.config import RUNS_DIR, get_device
from pinn_piezo.indirect.sampling import sample_training_tensors
from pinn_piezo.indirect import standard


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs-adam", type=int, default=1000)
    parser.add_argument("--epochs-lbfgs", type=int, default=20)
    parser.add_argument("--lr-adam", type=float, default=1e-3)
    parser.add_argument("--lr-lbfgs", type=float, default=1.0)
    parser.add_argument("--activation", choices=("tanh", "silu"), default="silu")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"),
                        default="auto")
    parser.add_argument("--dtype", choices=("float64", "float32"),
                        default="float64")
    parser.add_argument(
        "--num-threads", type=int, default=1,
        help="Number of PyTorch intra-op CPU threads used for timing.",
    )
    parser.add_argument("--interior-per-layer", type=int, default=512)
    parser.add_argument("--boundary-points", type=int, default=128)
    parser.add_argument("--interface-points", type=int, default=128)
    parser.add_argument("--lbfgs-interior-per-layer", type=int, default=1024)
    parser.add_argument("--resample-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--voltage", type=float, default=100.0)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[100, 250])
    parser.add_argument(
        "--interface-enriched",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add abs(y/H-0.5), a fixed interface-distance feature.",
    )
    parser.add_argument(
        "--phase-enriched",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Match the mixed model's cusp-plus-phase input and average the "
            "two material traces for continuous u, v and phi."
        ),
    )
    parser.add_argument(
        "--zero-output",
        action="store_true",
        help="Zero the last layer. Off by default for genuinely cold training.",
    )
    parser.add_argument("--run-name", type=str, default=None)
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
    project_config.VOLTAGE = args.voltage
    dtype = {"float64": torch.float64, "float32": torch.float32}[args.dtype]
    if args.device == "mps" and dtype == torch.float64:
        raise ValueError("Apple MPS requires --dtype float32")
    torch.set_default_dtype(dtype)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)

    run_name = args.run_name or (
        "train_indirect_three_field_"
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    run_dir = RUNS_DIR / run_name
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "config.json", vars(args))
    print(f"Using device: {device}; threads: {torch.get_num_threads()}")
    print(f"Run directory: {run_dir}")
    print("Initialization: cold random weights")

    activation = {"tanh": torch.nn.Tanh, "silu": torch.nn.SiLU}[
        args.activation
    ]
    model = standard.build_standard_model(
        device=device,
        hidden_sizes=tuple(args.hidden_sizes),
        activation=activation,
        interface_enriched=args.interface_enriched,
        phase_enriched=args.phase_enriched,
        zero_output=args.zero_output,
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
        n_boundary=max(args.boundary_points, 256),
        n_interface=max(args.interface_points, 256),
        seed=args.seed + 1_000_000,
        device=device,
        dtype=dtype,
    )
    result = standard.train_standard(
        model,
        tensors,
        epochs_adam=args.epochs_adam,
        epochs_lbfgs=args.epochs_lbfgs,
        lr_adam=args.lr_adam,
        lr_lbfgs=args.lr_lbfgs,
        resample_fn=sample_adam,
        resample_every=args.resample_every,
        lbfgs_tensors=lbfgs_tensors,
    )

    state_path = models_dir / "model_PINN_indirect_three_field.pt"
    torch.save(model.state_dict(), state_path)
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
    print(f"Model saved to {state_path}")


if __name__ == "__main__":
    main()
