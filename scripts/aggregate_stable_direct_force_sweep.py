"""Compare independently trained direct PINNs with FEM over applied force."""

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
from pinn_piezo.direct import model as model_mod
from pinn_piezo.fem import solve_piezo


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument(
        "--forces", type=float, nargs="+", default=[0.05, 0.1, 0.2],
    )
    parser.add_argument(
        "--prefix", default="stable_direct_force_sweep_",
    )
    parser.add_argument(
        "--base-run", default="direct_electromechanical_silu_dense_20260728",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/runs/eval_stable_direct_force_sweep"),
    )
    parser.add_argument(
        "--csv", type=Path, default=Path("direct_force_sweep_stable.csv"),
    )
    return parser.parse_args()


def force_tag(magnitude):
    return f"{magnitude:g}".replace(".", "p")


def run_name(magnitude, prefix, base_run):
    return base_run if np.isclose(magnitude, 0.1) else (
        f"{prefix}{force_tag(magnitude)}N"
    )


def load_model(run_dir, settings):
    activation = {"tanh": torch.nn.Tanh, "silu": torch.nn.SiLU}[
        settings["activation"]
    ]
    force = float(settings["force"])
    model = model_mod.build_default_model(
        device=torch.device("cpu"),
        hidden_sizes=tuple(settings["hidden_sizes"]),
        reference_force=force,
        normalization_force=settings.get("normalization_force"),
        legacy=settings.get("legacy", False),
        hard_natural_bcs=settings["hard_natural_bcs"],
        hard_floating_electrode=settings["hard_floating_electrode"],
        slender_warping=settings["slender_warping"],
        phase_enriched=settings["phase_enriched"],
        split_trunks=settings["split_trunks"],
        constitutive_bridge=settings.get("constitutive_bridge", False),
        hard_axial_constitutive=settings.get(
            "hard_axial_constitutive", False
        ),
        bending_basis=settings.get("bending_basis", False),
        beam_kinematics=settings.get("beam_kinematics", False),
        bridge_correction_limit=settings.get(
            "bridge_correction_limit", 0.1
        ),
        beam_static_lift=settings.get("beam_static_lift", False),
        traction_profile=settings.get("traction_profile", "uniform"),
        point_load_width_ratio=settings.get(
            "point_load_width_ratio", 1.0 / 16.0
        ),
        activation=activation,
    ).double()
    state = torch.load(
        run_dir / "models" / "model_PINN_direct.pt",
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


def error_metrics(prediction, reference):
    difference = prediction - reference
    return {
        "rel_L2": float(
            np.linalg.norm(difference) / np.linalg.norm(reference)
        ),
        "RMSE": float(np.sqrt(np.mean(difference**2))),
        "MAE": float(np.mean(np.abs(difference))),
        "max_abs": float(np.max(np.abs(difference))),
    }


def collect(run_dir, magnitude):
    settings = json.loads((run_dir / "config.json").read_text())
    force = -abs(float(magnitude))
    model, state = load_model(run_dir, settings)
    x_grid = np.linspace(0.0, config.WIDTH, 201)
    y_grid = np.linspace(0.0, config.HEIGHT, 21)
    xx, yy = np.meshgrid(x_grid, y_grid)
    metric_points = np.column_stack((xx.ravel(), yy.ravel()))
    y_tip = np.linspace(0.0, config.HEIGHT, 101)
    tip_points = np.column_stack((np.full_like(y_tip, config.WIDTH), y_tip))
    centerline_points = np.column_stack((
        x_grid, np.full_like(x_grid, config.CENTER),
    ))
    electrode_point = np.array([[0.5 * config.WIDTH, config.HEIGHT]])
    points = np.vstack((
        metric_points, tip_points, centerline_points, electrode_point,
    ))
    with torch.no_grad():
        prediction = model(
            torch.as_tensor(points, dtype=torch.float64)
        ).cpu().numpy()
    fem = solve_piezo(
        case="direct",
        nx=200,
        ny=8,
        force=force,
        poling_sign=config.CANONICAL_POLING_SIGN,
        direct_electrical_bc="floating_electrode",
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
        n_metric + n_tip, n_metric + n_tip + n_centerline
    )
    tip_mid_index = n_metric + n_tip // 2
    electrode_index = len(points) - 1
    metrics = {
        field: error_metrics(
            prediction[:n_metric, index], reference[:n_metric, index]
        )
        for index, field in enumerate(("u", "v", "phi"))
    }
    loss = np.load(run_dir / "loss_direct.npy")
    row = {
        "force_magnitude_N": magnitude,
        "run_name": run_dir.name,
        "state_sha256": state_digest(state),
        "u_tip_absmax_pinn_m": float(
            np.max(np.abs(prediction[tip_slice, 0]))
        ),
        "u_tip_absmax_fem_m": float(
            np.max(np.abs(reference[tip_slice, 0]))
        ),
        "v_tip_mid_pinn_m": float(prediction[tip_mid_index, 1]),
        "v_tip_mid_fem_m": float(reference[tip_mid_index, 1]),
        "phi_electrode_pinn_V": float(prediction[electrode_index, 2]),
        "phi_electrode_fem_V": float(reference[electrode_index, 2]),
        "final_loss": float(loss[-1]),
    }
    for field in ("u", "v", "phi"):
        for metric, value in metrics[field].items():
            row[f"{metric}_{field}"] = value
    profile = {
        "force": magnitude,
        "x_mm": 1e3 * x_grid,
        "v_pinn_um": 1e6 * prediction[centerline_slice, 1],
        "v_fem_um": 1e6 * reference[centerline_slice, 1],
    }
    return row, profile


def regression(rows, key):
    force = np.array([row["force_magnitude_N"] for row in rows])
    values = np.array([row[key] for row in rows])
    slope, intercept = np.polyfit(force, values, 1)
    fitted = slope * force + intercept
    denominator = np.sum((values - np.mean(values)) ** 2)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "R2": float(
            1.0 - np.sum((values - fitted) ** 2) / denominator
        ),
    }


def make_summary_plot(rows, output):
    force = np.array([row["force_magnitude_N"] for row in rows])
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
            axes[1, 0], "Floating-electrode potential", "V",
            "phi_electrode_fem_V", "phi_electrode_pinn_V", 1.0,
        ),
    )
    for axis, title, unit, fem_key, pinn_key, factor in comparisons:
        fem = factor * np.array([row[fem_key] for row in rows])
        pinn = factor * np.array([row[pinn_key] for row in rows])
        axis.plot(force, fem, "o-", linewidth=2, label="FEM P2")
        axis.plot(force, pinn, "s--", linewidth=2, label="PINN")
        axis.set_title(title)
        axis.set_xlabel("Applied force magnitude [N]")
        axis.set_ylabel(unit)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    for field, label, marker in (
        ("u", "$u$", "o"), ("v", "$v$", "s"),
        ("phi", "$\\phi$", "^"),
    ):
        axes[1, 1].plot(
            force, [row[f"rel_L2_{field}"] for row in rows],
            marker=marker, linewidth=2, label=label,
        )
    axes[1, 1].set_title("Global error against FEM")
    axes[1, 1].set_xlabel("Applied force magnitude [N]")
    axes[1, 1].set_ylabel("Relative L2 error")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(frameon=False)
    fig.suptitle("Direct effect: independent cold training at each force")
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_deflection_plot(profiles, output):
    fig, axes = plt.subplots(
        1, len(profiles), figsize=(12.6, 3.8),
        constrained_layout=True, sharex=True,
    )
    for axis, profile in zip(np.atleast_1d(axes), profiles):
        axis.plot(
            profile["x_mm"], profile["v_fem_um"],
            linewidth=2.2, label="FEM P2",
        )
        axis.plot(
            profile["x_mm"], profile["v_pinn_um"],
            "--", linewidth=2.2, label="PINN",
        )
        axis.set_title(f'{profile["force"]:g} N')
        axis.set_xlabel("$x$ [mm]")
        axis.set_ylabel("$v(x,H/2)$ [µm]")
        axis.grid(alpha=0.25)
    np.atleast_1d(axes)[0].legend(frameon=False)
    fig.suptitle("Mid-plane deflection across applied forces")
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    torch.set_default_dtype(torch.float64)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    profiles = []
    for magnitude in args.forces:
        directory = args.runs_root / run_name(
            magnitude, args.prefix, args.base_run
        )
        row, profile = collect(directory, magnitude)
        rows.append(row)
        profiles.append(profile)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    regressions = {
        "u_tip": regression(rows, "u_tip_absmax_pinn_m"),
        "v_tip": regression(rows, "v_tip_mid_pinn_m"),
        "phi_electrode": regression(rows, "phi_electrode_pinn_V"),
    }
    payload = {"rows": rows, "regressions": regressions}
    (args.output_dir / "results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    make_summary_plot(
        rows, args.output_dir / "force_sweep_fem_vs_pinn.png"
    )
    make_deflection_plot(
        profiles, args.output_dir / "force_sweep_deflection_profiles.png"
    )
    print(json.dumps(regressions, indent=2))


if __name__ == "__main__":
    main()
