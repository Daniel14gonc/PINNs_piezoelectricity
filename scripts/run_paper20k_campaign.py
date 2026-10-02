"""Run the paper's matched 20k-Adam PINN campaign, one job at a time.

All experiments start from the same fixed seed and random initialization.
The campaign deliberately uses Adam only so that the optimization budget is
matched across direct/converse and routing-on/routing-off models.  Outputs are
written under ``PINN_PIEZO_OUTPUTS_DIR/runs`` when that environment variable
is set (the Colab notebook points it at Google Drive).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys

from pinn_piezo.config import RUNS_DIR


SEED = 20260728


@dataclass(frozen=True)
class Experiment:
    name: str
    group: str
    module: str
    model_name: str
    arguments: tuple[str, ...]

    @property
    def run_name(self) -> str:
        return f"paper20k_{self.name}"


def indirect_mixed(
    *, hidden=(50, 50, 50), activation="silu", routing=True,
    interior=2048, voltage=100,
):
    return (
        "--epochs-lbfgs", "0",
        "--lr-adam", "1e-3",
        "--model-type", "pyramid",
        "--hidden-sizes", *(str(value) for value in hidden),
        "--activation", activation,
        "--phase-enriched",
        "--no-split-trunks",
        "--no-hard-natural-bcs",
        "--interior-per-layer", str(interior),
        "--boundary-points", "256",
        "--interface-points", "256",
        "--resample-every", "200",
        "--interface-weight", "1.0",
        "--no-slender-transverse-scaling",
        "--normal-constitutive-weight", "1.0",
        "--constitutive-normalization", "represented",
        "--no-constitutive-traction-bcs",
        "--no-constitutive-flux-stopgrad",
        (
            "--mechanical-constitutive-stopgrad"
            if routing else "--no-mechanical-constitutive-stopgrad"
        ),
        "--stress-warmup-epochs", "0",
        "--seed", str(SEED),
        "--voltage", str(voltage),
        "--no-zero-output",
        "--no-periodic-checkpoints",
    )


def direct_mixed(*, force=-0.1):
    return (
        "--epochs-lbfgs", "0",
        "--lr-adam", "1e-3",
        "--activation", "silu",
        "--hidden-sizes", "50", "50", "50",
        "--output-init-gain", "1.0",
        "--random-collocation",
        "--interior-per-layer", "2048",
        "--boundary-points", "256",
        "--interface-points", "256",
        "--resample-every", "200",
        "--force", str(force),
        "--traction-profile", "uniform",
        "--electrical-bc", "floating_electrode",
        "--hard-natural-bcs",
        "--hard-floating-electrode",
        "--slender-warping",
        "--phase-enriched",
        "--no-split-trunks",
        "--pde-weight", "1.0",
        "--bc-weight", "10.0",
        "--interface-weight", "1.0",
        "--no-constitutive-traction-bcs",
        "--normal-constitutive-weight", "1.0",
        "--strict-transverse-constitutive",
        "--constitutive-normalization", "dominant",
        "--lbfgs-grad-clip", "0.0",
        "--seed", str(SEED),
        "--no-periodic-checkpoints",
    )


def indirect_three_field():
    return (
        "--epochs-lbfgs", "0",
        "--lr-adam", "1e-3",
        "--activation", "silu",
        "--hidden-sizes", "50", "50", "50",
        "--phase-enriched",
        "--interface-enriched",
        "--interior-per-layer", "2048",
        "--boundary-points", "256",
        "--interface-points", "256",
        "--resample-every", "200",
        "--seed", str(SEED),
        "--voltage", "100",
    )


def direct_three_field():
    return (
        "--epochs-lbfgs", "0",
        "--lr-adam", "1e-3",
        "--activation", "silu",
        "--hidden-sizes", "50", "50", "50",
        "--output-init-gain", "1.0",
        "--phase-enriched",
        "--interface-enriched",
        "--slender-warping",
        "--hard-floating-electrode",
        "--interior-per-layer", "2048",
        "--boundary-points", "256",
        "--interface-points", "256",
        "--resample-every", "200",
        "--force", "-0.1",
        "--seed", str(SEED),
        "--no-periodic-checkpoints",
    )


def build_experiments() -> dict[str, Experiment]:
    items = [
        Experiment("direct_mixed", "baseline", "scripts.train_direct",
                   "model_PINN_direct.pt", direct_mixed()),
        Experiment("indirect_mixed", "baseline", "scripts.train_indirect",
                   "model_PINN_indirect.pt", indirect_mixed()),
        Experiment("direct_three_field", "mixed_vs_three_field",
                   "scripts.train_direct_three_field",
                   "model_PINN_direct_three_field.pt", direct_three_field()),
        Experiment("indirect_three_field", "mixed_vs_three_field",
                   "scripts.train_indirect_three_field",
                   "model_PINN_indirect_three_field.pt", indirect_three_field()),
        Experiment("indirect_arch_2x50", "architecture",
                   "scripts.train_indirect", "model_PINN_indirect.pt",
                   indirect_mixed(hidden=(50, 50))),
        Experiment("indirect_arch_4x50", "architecture",
                   "scripts.train_indirect", "model_PINN_indirect.pt",
                   indirect_mixed(hidden=(50, 50, 50, 50))),
        Experiment("indirect_arch_3x100", "architecture",
                   "scripts.train_indirect", "model_PINN_indirect.pt",
                   indirect_mixed(hidden=(100, 100, 100))),
        Experiment("indirect_tanh_3x50", "activation",
                   "scripts.train_indirect", "model_PINN_indirect.pt",
                   indirect_mixed(activation="tanh")),
        Experiment("indirect_routing_off", "gradient_routing",
                   "scripts.train_indirect", "model_PINN_indirect.pt",
                   indirect_mixed(routing=False)),
        Experiment("indirect_interior_512", "collocation_sensitivity",
                   "scripts.train_indirect", "model_PINN_indirect.pt",
                   indirect_mixed(interior=512)),
        Experiment("indirect_interior_1024", "collocation_sensitivity",
                   "scripts.train_indirect", "model_PINN_indirect.pt",
                   indirect_mixed(interior=1024)),
        Experiment("direct_force_0p05N", "force_sweep",
                   "scripts.train_direct", "model_PINN_direct.pt",
                   direct_mixed(force=-0.05)),
        Experiment("direct_force_0p2N", "force_sweep",
                   "scripts.train_direct", "model_PINN_direct.pt",
                   direct_mixed(force=-0.2)),
    ]
    for voltage in (200, 300, 400, 500):
        items.append(Experiment(
            f"indirect_voltage_{voltage}V", "voltage_sweep",
            "scripts.train_indirect", "model_PINN_indirect.pt",
            indirect_mixed(voltage=voltage),
        ))
    return {item.name: item for item in items}


EXPERIMENTS = build_experiments()


def model_path(experiment: Experiment) -> Path:
    return (
        RUNS_DIR / experiment.run_name / "models" / experiment.model_name
    )


def run_complete(experiment: Experiment) -> bool:
    """Return true only after every artifact needed for evaluation exists."""
    run_dir = RUNS_DIR / experiment.run_name
    loss_name = (
        "loss.npy"
        if "three_field" in experiment.model_name
        else (
            "loss_direct.npy"
            if experiment.name.startswith("direct_")
            else "loss_indirect.npy"
        )
    )
    required = (
        model_path(experiment),
        run_dir / "config.json",
        run_dir / "training_summary.json",
        run_dir / loss_name,
        run_dir / "loss_components.npz",
    )
    return run_dir.is_dir() and all(path.is_file() for path in required)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="+", choices=sorted(EXPERIMENTS))
    parser.add_argument("--epochs", type=int, default=20_000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float64", "float32"),
                        default="float64")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        print(json.dumps({
            name: {"group": exp.group, "run_name": exp.run_name}
            for name, exp in EXPERIMENTS.items()
        }, indent=2))
        return
    if not args.experiments:
        parser.error("--experiments is required unless --list is used")
    if args.epochs <= 0:
        parser.error("--epochs must be positive")

    for name in args.experiments:
        experiment = EXPERIMENTS[name]
        final_model = model_path(experiment)
        if run_complete(experiment):
            print(f"[SKIP] {name}: completed model exists at {final_model}")
            continue
        run_dir = RUNS_DIR / experiment.run_name
        if run_dir.exists():
            raise RuntimeError(
                f"Incomplete run directory exists: {run_dir}. "
                "Rename or remove it explicitly before restarting from scratch."
            )
        command = [
            sys.executable, "-u", "-m", experiment.module,
            "--epochs-adam", str(args.epochs),
            "--device", args.device,
            "--dtype", args.dtype,
            *experiment.arguments,
            "--run-name", experiment.run_name,
        ]
        print(f"\n[START] {name} ({experiment.group})", flush=True)
        print(" ".join(command), flush=True)
        subprocess.run(command, check=True)
        if not final_model.exists():
            raise RuntimeError(f"Training ended without model: {final_model}")
        print(f"[DONE] {name}: {final_model}", flush=True)


if __name__ == "__main__":
    main()
