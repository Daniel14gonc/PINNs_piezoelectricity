"""Compare stable independently trained indirect PINNs with FEM over voltage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pinn_piezo import config
from pinn_piezo.fem import solve_piezo
from pinn_piezo.indirect import model as model_mod


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument(
        "--voltages", type=int, nargs="+", default=[100, 200, 300, 400, 500],
    )
    parser.add_argument(
        "--prefix", default="stable_indirect_voltage_sweep_",
    )
    parser.add_argument(
        "--base-100-run",
        default=(
            "indirect_single_trunk_silu_dense_softbc_"
            "mechanical_routing_adam8000_20260730"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/runs/eval_stable_indirect_voltage_sweep"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("indirect_voltage_sweep_stable.csv"),
    )
    return parser.parse_args()


def run_name(voltage, *, prefix, base_100_run):
    return base_100_run if voltage == 100 else f"{prefix}{voltage}V"


def load_model(run_dir, settings):
    activation = {"tanh": torch.nn.Tanh, "silu": torch.nn.SiLU}[
        settings["activation"]
    ]
    model = model_mod.build_default_model(
        device=torch.device("cpu"),
        model_type=settings["model_type"],
        hidden_sizes=tuple(settings["hidden_sizes"]),
        phase_enriched=settings["phase_enriched"],
        hard_natural_bcs=settings["hard_natural_bcs"],
        activation=activation,
        zero_output=settings["zero_output"],
        split_trunks=settings["split_trunks"],
        constitutive_bridge=settings.get("constitutive_bridge", False),
        bridge_correction_limit=settings.get("bridge_correction_limit", 0.1),
    ).double()
    state = torch.load(
        run_dir / "models" / "model_PINN_indirect.pt",
        map_location="cpu",
    )
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    return model.eval(), state


def state_digest(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(
            state[name].detach().cpu().contiguous().numpy().tobytes()
        )
    return digest.hexdigest()


def relative_l2(prediction, reference):
    return float(
        np.linalg.norm(prediction - reference)
        / np.linalg.norm(reference)
    )


def error_metrics(prediction, reference):
    difference = prediction - reference
    return {
        "rel_L2": relative_l2(prediction, reference),
        "RMSE": float(np.sqrt(np.mean(difference**2))),
        "MAE": float(np.mean(np.abs(difference))),
        "max_abs": float(np.max(np.abs(difference))),
    }


def collect(run_dir, voltage):
    settings = json.loads((run_dir / "config.json").read_text())
    config.VOLTAGE = float(voltage)
    model, state = load_model(run_dir, settings)

    x_grid = np.linspace(0.0, config.WIDTH, 201)
    y_grid = np.linspace(0.0, config.HEIGHT, 21)
    xx, yy = np.meshgrid(x_grid, y_grid)
    metric_points = np.column_stack((xx.ravel(), yy.ravel()))

    y_tip = np.linspace(0.0, config.HEIGHT, 101)
    tip_points = np.column_stack((
        np.full_like(y_tip, config.WIDTH), y_tip,
    ))
    centerline_points = np.column_stack((
        x_grid, np.full_like(x_grid, config.CENTER),
    ))
    probe_points = np.array([
        [config.WIDTH, config.CENTER],
        [0.5 * config.WIDTH, 0.25 * config.HEIGHT],
    ])
    points = np.vstack((
        metric_points, tip_points, centerline_points, probe_points,
    ))

    with torch.no_grad():
        prediction = model(
            torch.as_tensor(points, dtype=torch.float64)
        ).cpu().numpy()
    fem = solve_piezo(
        case="indirect",
        nx=200,
        ny=8,
        voltage=float(voltage),
        poling_sign=config.CANONICAL_POLING_SIGN,
        element_order=2,
        eval_points=points,
    )
    reference = np.column_stack((
        fem.eval["u"], fem.eval["v"], fem.eval["phi"],
    ))

    n_metric = len(metric_points)
    n_tip = len(tip_points)
    n_centerline = len(centerline_points)
    tip_slice = slice(n_metric, n_metric + n_tip)
    centerline_slice = slice(
        n_metric + n_tip,
        n_metric + n_tip + n_centerline,
    )
    tip_probe_index = n_metric + n_tip + n_centerline
    phi_probe_index = tip_probe_index + 1
    loss = np.load(run_dir / "loss_indirect.npy")
    field_metrics = {
        field: error_metrics(
            prediction[:n_metric, index],
            reference[:n_metric, index],
        )
        for index, field in enumerate(("u", "v", "phi"))
    }

    row = {
        "voltage_V": voltage,
        "run_name": run_dir.name,
        "state_sha256": state_digest(state),
        "u_tip_absmax_pinn_m": float(
            np.max(np.abs(prediction[tip_slice, 0]))
        ),
        "u_tip_absmax_fem_m": float(
            np.max(np.abs(reference[tip_slice, 0]))
        ),
        "v_tip_mid_pinn_m": float(prediction[tip_probe_index, 1]),
        "v_tip_mid_fem_m": float(reference[tip_probe_index, 1]),
        "phi_probe_pinn_V": float(prediction[phi_probe_index, 2]),
        "phi_probe_fem_V": float(reference[phi_probe_index, 2]),
        "rel_L2_u": field_metrics["u"]["rel_L2"],
        "RMSE_u_m": field_metrics["u"]["RMSE"],
        "MAE_u_m": field_metrics["u"]["MAE"],
        "max_abs_u_m": field_metrics["u"]["max_abs"],
        "rel_L2_v": field_metrics["v"]["rel_L2"],
        "RMSE_v_m": field_metrics["v"]["RMSE"],
        "MAE_v_m": field_metrics["v"]["MAE"],
        "max_abs_v_m": field_metrics["v"]["max_abs"],
        "rel_L2_phi": field_metrics["phi"]["rel_L2"],
        "RMSE_phi_V": field_metrics["phi"]["RMSE"],
        "MAE_phi_V": field_metrics["phi"]["MAE"],
        "max_abs_phi_V": field_metrics["phi"]["max_abs"],
        "final_loss": float(loss[-1]),
    }
    profile = {
        "voltage": voltage,
        "x_mm": 1e3 * x_grid,
        "v_pinn_um": 1e6 * prediction[centerline_slice, 1],
        "v_fem_um": 1e6 * reference[centerline_slice, 1],
    }
    return row, profile


def regression(rows, key):
    voltage = np.array([row["voltage_V"] for row in rows], dtype=float)
    values = np.array([row[key] for row in rows], dtype=float)
    slope, intercept = np.polyfit(voltage, values, 1)
    fitted = slope * voltage + intercept
    denominator = np.sum((values - np.mean(values)) ** 2)
    r_squared = 1.0 - np.sum((values - fitted) ** 2) / denominator
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "R2": float(r_squared),
    }


def make_summary_plot(rows, output):
    voltage = np.array([row["voltage_V"] for row in rows])
    fig, axes = plt.subplots(
        2, 2, figsize=(11.5, 7.8), constrained_layout=True,
    )
    comparisons = (
        (
            axes[0, 0], "Maximum tip $|u|$", "µm",
            "u_tip_absmax_fem_m", "u_tip_absmax_pinn_m", 1e6,
        ),
        (
            axes[0, 1], "Mid-plane tip $v$", "µm",
            "v_tip_mid_fem_m", "v_tip_mid_pinn_m", 1e6,
        ),
        (
            axes[1, 0], "Interior $\\phi$ at $(L/2,H/4)$", "V",
            "phi_probe_fem_V", "phi_probe_pinn_V", 1.0,
        ),
    )
    for axis, title, unit, fem_key, pinn_key, factor in comparisons:
        fem = factor * np.array([row[fem_key] for row in rows])
        pinn = factor * np.array([row[pinn_key] for row in rows])
        axis.plot(voltage, fem, "o-", linewidth=2, label="FEM P2")
        axis.plot(voltage, pinn, "s--", linewidth=2, label="PINN")
        axis.set_title(title)
        axis.set_xlabel("Applied voltage [V]")
        axis.set_ylabel(unit)
        axis.grid(alpha=0.25)
        axis.legend()

    for key, label, marker in (
        ("rel_L2_u", "$u$", "o"),
        ("rel_L2_v", "$v$", "s"),
        ("rel_L2_phi", "$\\phi$", "^"),
    ):
        axes[1, 1].plot(
            voltage, [row[key] for row in rows],
            marker=marker, linewidth=2, label=label,
        )
    axes[1, 1].set_title("Global error against FEM")
    axes[1, 1].set_xlabel("Applied voltage [V]")
    axes[1, 1].set_ylabel("Relative L2 error")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend()
    fig.suptitle(
        "Stable indirect effect: independent cold training\n"
        "Shared trunk · SiLU/MSE · 8000 Adam · routed mechanical gradient"
    )
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_deflection_plot(profiles, output):
    fig, axes = plt.subplots(
        2, 3, figsize=(12.5, 7.2), constrained_layout=True,
        sharex=True,
    )
    flat_axes = axes.ravel()
    for axis, profile in zip(flat_axes, profiles):
        axis.plot(
            profile["x_mm"], profile["v_fem_um"],
            linewidth=2.2, label="FEM P2",
        )
        axis.plot(
            profile["x_mm"], profile["v_pinn_um"],
            "--", linewidth=2.2, label="PINN",
        )
        axis.set_title(f'{profile["voltage"]} V')
        axis.set_xlabel("$x$ [mm]")
        axis.set_ylabel("$v(x,H/2)$ [µm]")
        axis.grid(alpha=0.25)
    for axis in flat_axes[len(profiles):]:
        axis.axis("off")
    flat_axes[0].legend()
    fig.suptitle(
        "Mid-plane deflection: PINN vs FEM at each voltage"
    )
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(rows, regressions, output):
    lines = [
        "# Barrido de voltaje — indirecto estable",
        "",
        "Entrenamientos cold independientes con la receta final de un tronco,",
        "SiLU, 8000 Adam y gradiente constitutivo dirigido solo mecánico.",
        "",
        "| V | v punta PINN [µm] | v punta FEM [µm] | L2 u | L2 v | L2 phi |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f'| {row["voltage_V"]} '
            f'| {1e6 * row["v_tip_mid_pinn_m"]:.3f} '
            f'| {1e6 * row["v_tip_mid_fem_m"]:.3f} '
            f'| {row["rel_L2_u"]:.4f} '
            f'| {row["rel_L2_v"]:.4f} '
            f'| {row["rel_L2_phi"]:.4f} |'
        )
    lines.extend([
        "",
        "## Linealidad",
        "",
        f'- `u_tip`: R² = {regressions["u_tip"]["R2"]:.8f}',
        f'- `v_tip`: R² = {regressions["v_tip"]["R2"]:.8f}',
        f'- `phi_probe`: R² = {regressions["phi_probe"]["R2"]:.8f}',
        "",
        "Los hashes de los modelos se incluyen en el CSV para verificar si los",
        "problemas adimensionales siguieron exactamente la misma trayectoria.",
        "",
    ])
    output.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    torch.set_default_dtype(torch.float64)
    if any(voltage <= 0 for voltage in args.voltages):
        raise ValueError("all voltages must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    profiles = []
    for voltage in args.voltages:
        directory = args.runs_root / run_name(
            voltage, prefix=args.prefix, base_100_run=args.base_100_run,
        )
        row, profile = collect(directory, voltage)
        rows.append(row)
        profiles.append(profile)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    regressions = {
        "u_tip": regression(rows, "u_tip_absmax_pinn_m"),
        "v_tip": regression(rows, "v_tip_mid_pinn_m"),
        "phi_probe": regression(rows, "phi_probe_pinn_V"),
    }
    summary_plot = args.output_dir / "voltage_sweep_fem_vs_pinn.png"
    deflection_plot = args.output_dir / "voltage_sweep_deflection_profiles.png"
    make_summary_plot(rows, summary_plot)
    make_deflection_plot(profiles, deflection_plot)
    (args.output_dir / "results.json").write_text(
        json.dumps(
            {"rows": rows, "linearity": regressions}, indent=2,
        ),
        encoding="utf-8",
    )
    write_report(
        rows, regressions, args.output_dir / "README.md",
    )
    print(args.csv.resolve())
    print(summary_plot.resolve())
    print(deflection_plot.resolve())


if __name__ == "__main__":
    main()
