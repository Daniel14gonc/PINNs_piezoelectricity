"""Build a publication-ready results package from completed PINN runs.

The script evaluates the mixed and three-field baselines on the common
201 x 21 grid, compares them with the aligned internal P2 FEM reference, and
collects the completed ablation, load-scaling, and loss-history studies.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pinn_piezo import config as project_config
from pinn_piezo.fem import solve_piezo
from pinn_piezo.metrics import field_metrics
from scripts.evaluate import _select_model_and_tensorize


ROOT = Path(__file__).resolve().parents[1]
RUNS = project_config.RUNS_DIR
OUT = Path(os.environ.get(
    "PINN_PIEZO_RESULTS_PACKAGE_DIR", ROOT / "output" / "results_package",
))


def configured_run(variable: str, default: str) -> Path:
    return RUNS / os.environ.get(variable, default)


DIRECT_MIXED = configured_run(
    "PINN_PIEZO_DIRECT_MIXED_RUN",
    "direct_electromechanical_silu_dense_20260728",
)
INDIRECT_MIXED = configured_run(
    "PINN_PIEZO_INDIRECT_MIXED_RUN",
    "indirect_single_trunk_silu_dense_softbc_mechanical_routing_adam8000_20260730",
)
INDIRECT_ROUTING_OFF = configured_run(
    "PINN_PIEZO_INDIRECT_ROUTING_OFF_RUN",
    "revision_ablation_indirect_mixed_routing_off",
)
INDIRECT_TANH = configured_run(
    "PINN_PIEZO_INDIRECT_TANH_RUN",
    "revision_ablation_indirect_mixed_tanh_3x50",
)
DIRECT_THREE = configured_run(
    "PINN_PIEZO_DIRECT_THREE_RUN",
    "revision_ablation_direct_three_field_seed20260728",
)
INDIRECT_THREE = configured_run(
    "PINN_PIEZO_INDIRECT_THREE_RUN",
    "revision_ablation_indirect_three_field_seed20260728",
)
ABLATION_CSV = Path(os.environ.get(
    "PINN_PIEZO_ABLATION_CSV",
    RUNS / "revision_ablation_summary" / "ablation_results.csv",
))
DIRECT_SWEEP_CSV = Path(os.environ.get(
    "PINN_PIEZO_DIRECT_SWEEP_CSV", ROOT / "direct_force_sweep_stable.csv",
))
INDIRECT_SWEEP_CSV = Path(os.environ.get(
    "PINN_PIEZO_INDIRECT_SWEEP_CSV",
    ROOT / "indirect_voltage_sweep_stable.csv",
))

COLORS = {
    "mixed": "#1f77b4",
    "three": "#d95f02",
    "u": "#1f77b4",
    "v": "#d95f02",
    "phi": "#2a9d3f",
    "fem": "#252525",
    "pinn": "#1f77b4",
}


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def activation(name: str):
    return {"silu": torch.nn.SiLU, "tanh": torch.nn.Tanh}[name]


def checkpoint(run_dir: Path, formulation: str, three_field: bool) -> Path:
    if three_field:
        return (
            run_dir
            / "models"
            / f"model_PINN_{formulation}_three_field.pt"
        )
    return run_dir / "models" / f"model_PINN_{formulation}.pt"


def evaluate_run(
    run_dir: Path,
    formulation: str,
    *,
    three_field: bool,
) -> dict:
    settings = json.loads((run_dir / "config.json").read_text())
    force = float(settings.get("force", -0.1))
    voltage = float(settings.get("voltage", 100.0))
    if formulation == "indirect":
        project_config.VOLTAGE = voltage

    model, tensorize, _ = _select_model_and_tensorize(
        formulation,
        torch.device("cpu"),
        phase_enriched=bool(settings.get("phase_enriched", False)),
        activation=settings.get("activation", "silu"),
        force=force,
        normalization_force=settings.get("normalization_force"),
        hard_natural_bcs=bool(settings.get("hard_natural_bcs", False)),
        hard_floating_electrode=bool(
            settings.get("hard_floating_electrode", False)
        ),
        slender_warping=bool(settings.get("slender_warping", False)),
        three_field=three_field,
        interface_enriched=bool(settings.get("interface_enriched", True)),
        hidden_sizes=tuple(settings.get("hidden_sizes", (50, 50, 50))),
        split_trunks=bool(settings.get("split_trunks", False)),
        constitutive_bridge=bool(settings.get("constitutive_bridge", False)),
        hard_axial_constitutive=bool(
            settings.get("hard_axial_constitutive", False)
        ),
        bending_basis=bool(settings.get("bending_basis", False)),
        beam_kinematics=bool(settings.get("beam_kinematics", False)),
        bridge_correction_limit=float(
            settings.get("bridge_correction_limit", 0.1)
        ),
        beam_static_lift=bool(settings.get("beam_static_lift", False)),
        legacy_direct=bool(settings.get("legacy", False)),
    )
    state = torch.load(
        checkpoint(run_dir, formulation, three_field),
        map_location="cpu",
    )
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    x = np.linspace(0.0, project_config.WIDTH, 201)
    y = np.linspace(0.0, project_config.HEIGHT, 21)
    grid_x, grid_y = np.meshgrid(x, y)
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    with torch.no_grad():
        prediction = model(tensorize(points))[:, :3].cpu().numpy()

    fem_kwargs = {
        "case": formulation,
        "nx": 200,
        "ny": 8,
        "poling_sign": project_config.CANONICAL_POLING_SIGN,
        "element_order": 2,
        "eval_points": points,
    }
    if formulation == "direct":
        fem_kwargs.update({
            "force": force,
            "direct_electrical_bc": settings.get(
                "electrical_bc", "floating_electrode"
            ),
        })
    else:
        fem_kwargs["voltage"] = voltage
    fem = solve_piezo(**fem_kwargs)
    reference = np.column_stack(
        (fem.eval["u"], fem.eval["v"], fem.eval["phi"])
    )

    metrics = {
        field: field_metrics(prediction[:, i], reference[:, i])
        for i, field in enumerate(("u", "v", "phi"))
    }
    return {
        "formulation": formulation,
        "model": "three-field" if three_field else "mixed",
        "x": x,
        "y": y,
        "shape": grid_x.shape,
        "points": points,
        "prediction": prediction,
        "reference": reference,
        "metrics": metrics,
        "force": force,
        "voltage": voltage,
    }


def field_definition(index: int):
    return (
        ("u", r"$u$", r"$\mu$m", 1e6, "viridis"),
        ("v", r"$v$", r"$\mu$m", 1e6, "viridis"),
        ("phi", r"$\varphi$", "V", 1.0, "viridis"),
    )[index]


def plot_primary_fields(data: dict, output: Path) -> None:
    shape = data["shape"]
    x = data["x"] * 1e3
    y = data["y"] * 1e3
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 4.9), constrained_layout=True)
    for col in range(3):
        field, symbol, unit, factor, cmap = field_definition(col)
        ref = data["reference"][:, col].reshape(shape) * factor
        pred = data["prediction"][:, col].reshape(shape) * factor
        lo = float(min(ref.min(), pred.min()))
        hi = float(max(ref.max(), pred.max()))
        if field in ("u", "v") and lo < 0 < hi:
            limit = max(abs(lo), abs(hi))
            lo, hi = -limit, limit
        for row, (values, label) in enumerate(((ref, "FEM P2"), (pred, "PINN"))):
            image = axes[row, col].pcolormesh(
                x, y, values, shading="auto", cmap=cmap, vmin=lo, vmax=hi
            )
            axes[row, col].set_box_aspect(0.23)
            axes[row, col].set_xlabel(r"$x$ [mm]")
            axes[row, col].set_ylabel(r"$y$ [mm]")
            axes[row, col].set_title(f"{symbol} - {label}")
            bar = fig.colorbar(image, ax=axes[row, col], pad=0.02)
            bar.set_label(unit)
    load = (
        f"Direct effect, |F| = {abs(data['force']):g} N"
        if data["formulation"] == "direct"
        else f"Indirect effect, V = {data['voltage']:g} V"
    )
    kind = "mixed eight-field PINN" if data["model"] == "mixed" else "three-field PINN"
    fig.suptitle(f"{load} - {kind}", fontsize=12)
    save_figure(fig, output)


def plot_error_maps(data: dict, output: Path) -> None:
    shape = data["shape"]
    x = data["x"] * 1e3
    y = data["y"] * 1e3
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 4.9), constrained_layout=True)
    for col in range(3):
        field, symbol, unit, factor, _ = field_definition(col)
        ref = data["reference"][:, col]
        pred = data["prediction"][:, col]
        abs_error = np.abs(pred - ref) * factor
        denominator = float(np.max(np.abs(ref)))
        normalized = np.abs(pred - ref) / denominator if denominator else np.nan
        maps = (
            (abs_error.reshape(shape), f"Absolute error in {symbol}", unit),
            (normalized.reshape(shape), f"Normalized error in {symbol}", "-"),
        )
        for row, (values, title, bar_label) in enumerate(maps):
            image = axes[row, col].pcolormesh(
                x, y, values, shading="auto", cmap="viridis"
            )
            axes[row, col].set_box_aspect(0.23)
            axes[row, col].set_xlabel(r"$x$ [mm]")
            axes[row, col].set_ylabel(r"$y$ [mm]")
            axes[row, col].set_title(title)
            bar = fig.colorbar(image, ax=axes[row, col], pad=0.02)
            bar.set_label(bar_label)
    load = (
        f"Direct effect, |F| = {abs(data['force']):g} N"
        if data["formulation"] == "direct"
        else f"Indirect effect, V = {data['voltage']:g} V"
    )
    kind = "mixed eight-field PINN" if data["model"] == "mixed" else "three-field PINN"
    fig.suptitle(f"PINN-FEM pointwise errors - {load} - {kind}", fontsize=12)
    save_figure(fig, output)


def plot_fem_pinn_normalized_error(data: dict, output: Path) -> None:
    """Plot FEM, PINN, and normalized pointwise error in one 3 x 3 figure."""
    shape = data["shape"]
    x = data["x"] * 1e3
    y = data["y"] * 1e3
    fig, axes = plt.subplots(3, 3, figsize=(12.6, 7.0), constrained_layout=True)
    for col in range(3):
        field, symbol, unit, factor, cmap = field_definition(col)
        reference_raw = data["reference"][:, col]
        prediction_raw = data["prediction"][:, col]
        reference = reference_raw.reshape(shape) * factor
        prediction = prediction_raw.reshape(shape) * factor

        lo = float(min(reference.min(), prediction.min()))
        hi = float(max(reference.max(), prediction.max()))
        if field in ("u", "v") and lo < 0 < hi:
            limit = max(abs(lo), abs(hi))
            lo, hi = -limit, limit

        denominator = float(np.max(np.abs(reference_raw)))
        normalized_error = (
            np.abs(prediction_raw - reference_raw) / denominator
            if denominator
            else np.zeros_like(reference_raw)
        ).reshape(shape)

        rows = (
            (reference, "FEM", lo, hi, unit),
            (prediction, "PINN", lo, hi, unit),
            (normalized_error, "Normalized error", 0.0, None, r"$\widehat{e}$"),
        )
        for row, (values, label, vmin, vmax, bar_label) in enumerate(rows):
            image = axes[row, col].pcolormesh(
                x,
                y,
                values,
                shading="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            axes[row, col].set_box_aspect(0.23)
            axes[row, col].set_xlabel(r"$x$ [mm]")
            axes[row, col].set_ylabel(r"$y$ [mm]")
            axes[row, col].set_title(f"{symbol} - {label}")
            bar = fig.colorbar(image, ax=axes[row, col], pad=0.02)
            bar.set_label(bar_label)

    load = (
        f"Direct effect, |F| = {abs(data['force']):g} N"
        if data["formulation"] == "direct"
        else f"Indirect effect, V = {data['voltage']:g} V"
    )
    fig.suptitle(f"FEM-PINN comparison and normalized pointwise error - {load.replace('Indirect', 'Converse')}", fontsize=12)
    save_figure(fig, output)


def plot_routing_fields(routing_on: dict, routing_off: dict, output: Path) -> None:
    """Compare converse-effect primary fields with matched color limits."""
    shape = routing_on["shape"]
    x = routing_on["x"] * 1e3
    y = routing_on["y"] * 1e3
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 4.9), constrained_layout=True)
    rows = ((routing_on, "Routing on"), (routing_off, "Routing off"))
    for col in range(3):
        field, symbol, unit, factor, cmap = field_definition(col)
        all_values = np.concatenate([
            data["prediction"][:, col] * factor for data, _ in rows
        ])
        lo = float(all_values.min())
        hi = float(all_values.max())
        if field in ("u", "v") and lo < 0 < hi:
            limit = max(abs(lo), abs(hi))
            lo, hi = -limit, limit
        for row, (data, label) in enumerate(rows):
            values = data["prediction"][:, col].reshape(shape) * factor
            image = axes[row, col].pcolormesh(
                x, y, values, shading="auto", cmap=cmap, vmin=lo, vmax=hi
            )
            axes[row, col].set_box_aspect(0.23)
            axes[row, col].set_xlabel(r"$x$ [mm]")
            axes[row, col].set_ylabel(r"$y$ [mm]")
            axes[row, col].set_title(f"{symbol} - {label}")
            bar = fig.colorbar(image, ax=axes[row, col], pad=0.02)
            bar.set_label(unit)
    fig.suptitle(
        "Effect of mechanical gradient routing - converse effect, V = 100 V",
        fontsize=12,
    )
    save_figure(fig, output)


def plot_mixed_three_metrics(results: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True, constrained_layout=True)
    fields = ("u", "v", "phi")
    x = np.arange(3)
    width = 0.34
    for axis, formulation, title in zip(
        axes,
        ("direct", "indirect"),
        ("Direct effect", "Converse effect"),
    ):
        mixed = [results[(formulation, "mixed")]["metrics"][f]["rel_L2"] for f in fields]
        three = [results[(formulation, "three-field")]["metrics"][f]["rel_L2"] for f in fields]
        axis.bar(x - width / 2, mixed, width, label="Mixed eight-field", color=COLORS["mixed"])
        axis.bar(x + width / 2, three, width, label="Three-field", color=COLORS["three"])
        axis.set_xticks(x, (r"$u$", r"$v$", r"$\varphi$"))
        axis.set_title(title)
        axis.set_ylabel(r"Relative $L^2$ error")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
    save_figure(fig, output)


def plot_mixed_three_fields(
    mixed: dict,
    three_field: dict,
    output: Path,
) -> None:
    """Compare mixed and three-field PINN predictions on matched color scales."""
    shape = mixed["shape"]
    x = mixed["x"] * 1e3
    y = mixed["y"] * 1e3
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 4.9), constrained_layout=True)
    rows = ((mixed, "Mixed PINN"), (three_field, "Three-field PINN"))
    for col in range(3):
        field, symbol, unit, factor, cmap = field_definition(col)
        # Use the exact FEM-plus-mixed limits used by the baseline validation
        # figure.  This ensures that the same mixed prediction has the same
        # color in every figure; the three-field prediction is then shown on
        # that fixed physical reference scale.
        scale_values = np.concatenate((
            mixed["reference"][:, col] * factor,
            mixed["prediction"][:, col] * factor,
        ))
        lo = float(scale_values.min())
        hi = float(scale_values.max())
        if field in ("u", "v") and lo < 0 < hi:
            limit = max(abs(lo), abs(hi))
            lo, hi = -limit, limit
        for row, (data, label) in enumerate(rows):
            values = data["prediction"][:, col].reshape(shape) * factor
            image = axes[row, col].pcolormesh(
                x,
                y,
                values,
                shading="auto",
                cmap=cmap,
                vmin=lo,
                vmax=hi,
            )
            axes[row, col].set_box_aspect(0.23)
            axes[row, col].set_xlabel(r"$x$ [mm]")
            axes[row, col].set_ylabel(r"$y$ [mm]")
            axes[row, col].set_title(f"{symbol} - {label}")
            bar = fig.colorbar(image, ax=axes[row, col], pad=0.02)
            bar.set_label(unit)

    if mixed["formulation"] == "direct":
        title = f"Direct effect, |F| = {abs(mixed['force']):g} N"
    else:
        title = f"Converse effect, V = {mixed['voltage']:g} V"
    fig.suptitle(f"Mixed versus three-field PINN - {title}", fontsize=12)
    save_figure(fig, output)


def read_ablation_rows() -> list[dict]:
    with ABLATION_CSV.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metric_triplet(row: dict) -> list[float]:
    return [float(row[f"{field}_rel_L2"]) for field in ("u", "v", "phi")]


def grouped_bars(axis, labels: list[str], values: list[list[float]], title: str) -> None:
    x = np.arange(len(labels))
    width = 0.24
    for offset, field in zip((-1, 0, 1), ("u", "v", "phi")):
        field_label = r"$\varphi$" if field == "phi" else f"${field}$"
        axis.bar(
            x + offset * width,
            [row[("u", "v", "phi").index(field)] for row in values],
            width,
            label=field_label,
            color=COLORS[field],
        )
    axis.set_xticks(x, labels, rotation=18, ha="right")
    axis.set_title(title)
    axis.set_ylabel(r"Relative $L^2$ error")
    axis.grid(axis="y", alpha=0.25)


def plot_indirect_ablations(rows: list[dict], output: Path) -> None:
    baseline = next(row for row in rows if row["formulation"] == "indirect" and row["group"] == "baseline")
    architecture = [
        next(row for row in rows if row["run"].endswith("arch_2x50")),
        baseline,
        next(row for row in rows if row["run"].endswith("arch_3x100")),
        next(row for row in rows if row["run"].endswith("arch_4x50")),
    ]
    activation_rows = [
        baseline,
        next(row for row in rows if "tanh_3x50" in row["run"]),
    ]
    routing_rows = [
        baseline,
        next(row for row in rows if "routing_off" in row["run"]),
    ]
    collocation_rows = [
        next(row for row in rows if row["run"].endswith("interior_512")),
        next(row for row in rows if row["run"].endswith("interior_1024")),
        baseline,
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.2), constrained_layout=True)
    grouped_bars(axes[0, 0], ["2 x 50", "3 x 50", "3 x 100", "4 x 50"], [metric_triplet(r) for r in architecture], "Network architecture")
    grouped_bars(axes[0, 1], ["SiLU", "tanh"], [metric_triplet(r) for r in activation_rows], "Activation function")
    grouped_bars(axes[1, 0], ["On", "Off"], [metric_triplet(r) for r in routing_rows], "Mechanical gradient routing")
    grouped_bars(axes[1, 1], ["512", "1024", "2048"], [metric_triplet(r) for r in collocation_rows], "Interior points per layer")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
    )
    fig.suptitle("Indirect-effect ablation studies", fontsize=12, y=1.035)
    save_figure(fig, output)


def regression(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    coefficient = np.polyfit(x, y, 1)
    fit = np.polyval(coefficient, x)
    residual = float(np.sum((y - fit) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - residual / total if total else 1.0
    return fit, r2


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_load_sweep(path: Path, formulation: str, output: Path) -> None:
    rows = load_csv(path)
    if formulation == "direct":
        x = np.array([float(row["force_magnitude_N"]) for row in rows])
        definitions = (
            ("u_tip_absmax_pinn_m", "u_tip_absmax_fem_m", 1e6, r"Maximum tip $|u|$ [$\mu$m]"),
            ("v_tip_mid_pinn_m", "v_tip_mid_fem_m", 1e6, r"Mid-plane tip $v$ [$\mu$m]"),
            ("phi_electrode_pinn_V", "phi_electrode_fem_V", 1.0, r"Floating-electrode potential [V]"),
        )
        xlabel = "Applied force magnitude [N]"
        title = "Direct-effect load scaling"
    else:
        x = np.array([float(row["voltage_V"]) for row in rows])
        definitions = (
            ("u_tip_absmax_pinn_m", "u_tip_absmax_fem_m", 1e6, r"Maximum tip $|u|$ [$\mu$m]"),
            ("v_tip_mid_pinn_m", "v_tip_mid_fem_m", 1e6, r"Mid-plane tip $v$ [$\mu$m]"),
            ("phi_probe_pinn_V", "phi_probe_fem_V", 1.0, r"Interior $\varphi$ [V]"),
        )
        xlabel = "Applied voltage [V]"
        title = "Converse-effect load scaling"
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.7), constrained_layout=True)
    for axis, (pinn_key, fem_key, factor, ylabel) in zip(axes, definitions):
        pinn = np.array([float(row[pinn_key]) for row in rows]) * factor
        fem = np.array([float(row[fem_key]) for row in rows]) * factor
        pinn_fit, pinn_r2 = regression(x, pinn)
        fem_fit, fem_r2 = regression(x, fem)
        axis.scatter(x, fem, color=COLORS["fem"], marker="o", label=f"FEM P2 ($R^2={fem_r2:.4f}$)")
        axis.plot(x, fem_fit, color=COLORS["fem"], linewidth=1.2)
        axis.scatter(x, pinn, color=COLORS["pinn"], marker="s", label=f"PINN ($R^2={pinn_r2:.4f}$)")
        axis.plot(x, pinn_fit, color=COLORS["pinn"], linewidth=1.2, linestyle="--")
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=12)
    save_figure(fig, output)


def load_loss(run_dir: Path) -> np.ndarray:
    for name in ("loss.npy", "loss_direct.npy", "loss_indirect.npy"):
        path = run_dir / name
        if path.exists():
            return np.asarray(np.load(path), dtype=float).reshape(-1)
    raise FileNotFoundError(f"No loss history in {run_dir}")


def plot_direct_baseline_loss(output: Path) -> None:
    loss = load_loss(DIRECT_MIXED)
    fig, axis = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    axis.semilogy(np.arange(1, loss.size + 1), loss, color=COLORS["mixed"])
    axis.axvline(4000, color="0.3", linestyle="--", linewidth=1)
    axis.text(
        4000,
        axis.get_ylim()[1],
        " L-BFGS",
        va="top",
        ha="left",
        fontsize=8,
    )
    axis.set_xlabel("Optimization step")
    axis.set_ylabel("Total physics loss")
    axis.set_title("Direct-effect training history")
    axis.grid(alpha=0.25)
    save_figure(fig, output)


def plot_indirect_baseline_loss(output: Path) -> None:
    loss = load_loss(INDIRECT_MIXED)
    fig, axis = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    axis.semilogy(np.arange(1, loss.size + 1), loss, color=COLORS["mixed"])
    axis.set_xlabel("Optimization step")
    axis.set_ylabel("Total physics loss")
    axis.set_title("Converse-effect training loss")
    axis.grid(alpha=0.25)
    save_figure(fig, output)


def plot_loss_components(run_dir: Path, title: str, output: Path) -> None:
    """Plot the same weighted component families recorded during training."""
    path = run_dir / "loss_components.npz"
    if not path.exists():
        return
    histories = np.load(path)
    definitions = (
        ("constitutive", "Constitutive", "#1f77b4"),
        ("balance", "Equilibrium + Gauss", "#d95f02"),
        ("boundary", "Boundary", "#2a9d3f"),
        ("interface", "Interface", "#9467bd"),
        ("pde_total", "PDE", "#d95f02"),
        ("bc_total", "Boundary", "#2a9d3f"),
        ("interface_total", "Interface", "#9467bd"),
        ("total", "Total", "#252525"),
    )
    fig, axis = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)
    for key, label, color in definitions:
        if key not in histories:
            continue
        values = np.asarray(histories[key], dtype=float).reshape(-1)
        values = np.where(values > 0.0, values, np.nan)
        if np.all(np.isnan(values)):
            continue
        axis.semilogy(
            np.arange(1, values.size + 1), values,
            label=label, color=color, linewidth=1.1,
        )
    axis.set_xlabel("Adam epoch")
    axis.set_ylabel("Weighted loss component")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    save_figure(fig, output)


def normalized_loss(loss: np.ndarray) -> np.ndarray:
    finite = loss[np.isfinite(loss) & (loss > 0)]
    scale = finite[0] if finite.size else 1.0
    return loss / scale


def plot_three_field_losses(output: Path) -> None:
    definitions = (
        ("Direct effect", DIRECT_MIXED, DIRECT_THREE),
        ("Converse effect", INDIRECT_MIXED, INDIRECT_THREE),
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8), constrained_layout=True)
    for axis, (title, mixed_dir, three_dir) in zip(axes, definitions):
        mixed = normalized_loss(load_loss(mixed_dir))
        three = normalized_loss(load_loss(three_dir))
        axis.semilogy(np.arange(1, mixed.size + 1), mixed, label="Mixed eight-field", color=COLORS["mixed"])
        axis.semilogy(np.arange(1, three.size + 1), three, label="Three-field", color=COLORS["three"])
        axis.set_title(title)
        axis.set_xlabel("Optimization step")
        axis.set_ylabel(r"Normalized loss $\mathcal{L}/\mathcal{L}_0$")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.suptitle("Training histories", fontsize=12)
    save_figure(fig, output)


def architecture_cost_table(rows: list[dict]) -> str:
    definitions = (
        (
            "Direct",
            "Mixed eight-field",
            next(row for row in rows if row["run"] == DIRECT_MIXED.name),
        ),
        (
            "Direct",
            "Three-field",
            next(row for row in rows if row["run"] == DIRECT_THREE.name),
        ),
        (
            "Converse",
            "Mixed eight-field",
            next(row for row in rows if row["run"] == INDIRECT_MIXED.name),
        ),
        (
            "Converse",
            "Three-field",
            next(row for row in rows if row["run"] == INDIRECT_THREE.name),
        ),
    )
    mixed_times = {
        effect: float(row["training_time_s"])
        for effect, model, row in definitions
        if model == "Mixed eight-field"
    }
    lines = [
        "# Mixed and three-field computational cost",
        "",
        "| Effect | Formulation | Outputs | Trainable parameters | Training time [s] | Training time [h] | Time relative to mixed |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for effect, model, row in definitions:
        seconds = float(row["training_time_s"])
        ratio = seconds / mixed_times[effect]
        outputs = 8 if model == "Mixed eight-field" else 3
        lines.append(
            f"| {effect} | {model} | {outputs} | "
            f"{int(row['parameter_count']):,} | {seconds:,.1f} | "
            f"{seconds / 3600.0:.3f} | {ratio:.2f}x |"
        )
    lines.extend([
        "",
        "All runs used the same 3 x 50 hidden architecture, SiLU activation, float64 precision, collocation counts, and matched optimizer budget within each effect.",
        "The three-field formulation has fewer trainable parameters, but its strong-form residuals require higher-order automatic differentiation and therefore take longer to evaluate.",
        "",
    ])
    return "\n".join(lines)


def plot_indirect_loss_ablations(output: Path) -> None:
    baseline = load_loss(INDIRECT_MIXED)
    routing_off = load_loss(INDIRECT_ROUTING_OFF)
    tanh = load_loss(INDIRECT_TANH)
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8), constrained_layout=True)
    comparisons = (
        ("Mechanical gradient routing", baseline, routing_off, "Routing on", "Routing off"),
        ("Activation function", baseline, tanh, "SiLU", "tanh"),
    )
    for axis, (title, first, second, first_label, second_label) in zip(axes, comparisons):
        axis.semilogy(np.arange(1, first.size + 1), first, label=first_label, color=COLORS["mixed"])
        axis.semilogy(np.arange(1, second.size + 1), second, label=second_label, color=COLORS["three"])
        axis.set_title(title)
        axis.set_xlabel("Optimization step")
        axis.set_ylabel("Total physics loss")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.suptitle("Indirect-effect loss ablations", fontsize=12)
    save_figure(fig, output)


def markdown_metrics(results: dict, routing_off: dict) -> str:
    lines = [
        "# PINN-FEM error metrics",
        "",
        "All values use the common 201 x 21 evaluation grid and the same internal P2 FEM reference.",
        "Displacement RMSE and MAE are reported in micrometers; potential errors are reported in volts.",
        "",
        "| Effect | Model | Field | RMSE | MAE | Maximum absolute error | nRMSE | Relative L2 error |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {"direct": "Direct, 0.1 N", "indirect": "Converse, 100 V"}
    model_labels = {"mixed": "Mixed eight-field", "three-field": "Three-field"}
    for formulation in ("direct", "indirect"):
        for model in ("mixed", "three-field"):
            data = results[(formulation, model)]
            for field in ("u", "v", "phi"):
                metric = data["metrics"][field]
                factor = 1e6 if field in ("u", "v") else 1.0
                display_field = "φ" if field == "phi" else field
                lines.append(
                    f"| {labels[formulation]} | {model_labels[model]} | {display_field} | "
                    f"{metric['RMSE'] * factor:.6g} | {metric['MAE'] * factor:.6g} | "
                    f"{metric['max_abs'] * factor:.6g} | {metric['nRMSE']:.6f} | "
                    f"{metric['rel_L2']:.6f} |"
                )
    lines.extend([
        "",
        "## Gradient-routing ablation",
        "",
        "Converse effect at 100 V. Displacement RMSE and MAE are reported in micrometers; potential errors are reported in volts.",
        "",
        "| Gradient routing | Field | RMSE | MAE | Maximum absolute error | nRMSE | Relative L2 error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    routing_cases = (
        ("On", results[("indirect", "mixed")]),
        ("Off", routing_off),
    )
    for routing_label, data in routing_cases:
        for field in ("u", "v", "phi"):
            metric = data["metrics"][field]
            factor = 1e6 if field in ("u", "v") else 1.0
            display_field = "φ" if field == "phi" else field
            lines.append(
                f"| {routing_label} | {display_field} | "
                f"{metric['RMSE'] * factor:.6g} | "
                f"{metric['MAE'] * factor:.6g} | "
                f"{metric['max_abs'] * factor:.6g} | "
                f"{metric['nRMSE']:.6f} | "
                f"{metric['rel_L2']:.6f} |"
            )
    lines.extend([
        "",
        "## Pointwise error maps",
        "",
        "The absolute map is e_k = |psi_PINN(x_k) - psi_FEM(x_k)|. The normalized map divides this value by max_j |psi_FEM(x_j)|. Both maps therefore require a FEM reference and are not training losses.",
        "",
    ])
    return "\n".join(lines)


def write_readme() -> None:
    text = """# Results figure package

This folder contains the completed figures that can be used in the Results section.

## Recommended main-text figures

1. `01_baseline_fields/`: use `direct_mixed_fem_pinn_normalized_error` and `indirect_mixed_fem_pinn_normalized_error` as the main validation figures. Each combines the FEM field, PINN prediction, and normalized pointwise error for u, v, and phi. No aggregate error metric is printed inside the panels because those values are reported in the error table.
2. `02_error_maps/`: separate absolute and globally normalized pointwise PINN-FEM error maps, retained only as optional supporting outputs.
3. `04_ablation_summary/`: mixed-versus-three-field relative-L2 metrics and matched field maps for both effects, together with the completed converse-effect architecture, activation, routing, and collocation studies.
4. `05_load_scaling/`: independent force and voltage sweeps with linear fits and R-squared values. These are load-scaling studies, not ablations.
5. `tables/error_metrics.md`: RMSE, MAE, and relative L2 errors on the common 201 x 21 grid.
6. `tables/mixed_vs_three_field_training_cost.md`: trainable-parameter counts and measured training times for the matched mixed and three-field runs.

## Recommended supplementary figures

- `03_three_field_fields/`: full FEM-PINN maps for the three-field baselines.
- `06_loss_curves/`: baseline and ablation training histories.
- `07_supplementary/`: runtime comparison and existing tip-deflection profiles when available.

## Important interpretation notes

- The normalized error is divided by the maximum absolute FEM value of the complete field, not by the local FEM value. This avoids singular-looking errors near zero crossings.
- Absolute total losses should not be used to rank the mixed and three-field formulations because they minimize different residual sets. Their normalized histories show convergence behavior only; FEM error metrics provide the comparison of accuracy.
- The force and voltage sweeps use independently trained models. They test linear load scaling and do not demonstrate parametric generalization.

## Results that are still scientifically useful but not fully covered

- Validation of the mixed auxiliary outputs (stress and electric displacement) against FEM.
- Explicit interface-jump diagnostics for transmitted quantities.
- Repeated-seed statistics or uncertainty bars. The current comparisons use one matched seed.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def copy_supplementary() -> None:
    target = OUT / "07_supplementary"
    target.mkdir(parents=True, exist_ok=True)
    candidates = {
        RUNS / "runtime_pinn_vs_fem" / "runtime_comparison.png": "runtime_comparison.png",
        RUNS / "eval_stable_direct_force_sweep" / "force_sweep_deflection_profiles.png": "direct_force_sweep_deflection_profiles.png",
        RUNS / "eval_stable_indirect_voltage_sweep" / "voltage_sweep_deflection_profiles.png": "indirect_voltage_sweep_deflection_profiles.png",
    }
    for source, name in candidates.items():
        if source.exists():
            shutil.copy2(source, target / name)


def remove_obsolete_generated_files() -> None:
    obsolete = (
        OUT / "04_ablation_summary" / "converse_ablation_relative_l2.png",
        OUT / "04_ablation_summary" / "converse_ablation_relative_l2.pdf",
        OUT / "05_load_scaling" / "converse_voltage_scaling.png",
        OUT / "05_load_scaling" / "converse_voltage_scaling.pdf",
        OUT / "05_load_scaling" / "indirect_voltage_scaling.png",
        OUT / "05_load_scaling" / "indirect_voltage_scaling.pdf",
        OUT / "06_loss_curves" / "converse_loss_ablations.png",
        OUT / "06_loss_curves" / "converse_loss_ablations.pdf",
        OUT / "06_loss_curves" / "mixed_baseline_losses.png",
        OUT / "06_loss_curves" / "mixed_baseline_losses.pdf",
        OUT / "07_supplementary" / "converse_voltage_sweep_deflection_profiles.png",
    )
    for path in obsolete:
        if path.exists():
            path.unlink()


def main() -> None:
    torch.set_default_dtype(torch.float64)
    set_style()
    OUT.mkdir(parents=True, exist_ok=True)
    remove_obsolete_generated_files()

    results = {}
    for formulation, model, run_dir, three_field in (
        ("direct", "mixed", DIRECT_MIXED, False),
        ("indirect", "mixed", INDIRECT_MIXED, False),
        ("direct", "three-field", DIRECT_THREE, True),
        ("indirect", "three-field", INDIRECT_THREE, True),
    ):
        data = evaluate_run(run_dir, formulation, three_field=three_field)
        results[(formulation, model)] = data
        folder = "01_baseline_fields" if model == "mixed" else "03_three_field_fields"
        stem = f"{formulation}_{'mixed' if model == 'mixed' else 'three_field'}_primary_fields"
        plot_primary_fields(data, OUT / folder / stem)
        error_folder = "02_error_maps" if model == "mixed" else "03_three_field_fields"
        plot_error_maps(data, OUT / error_folder / f"{formulation}_{model.replace('-', '_')}_error_maps")
        if model == "mixed":
            plot_fem_pinn_normalized_error(
                data,
                OUT
                / "01_baseline_fields"
                / f"{formulation}_mixed_fem_pinn_normalized_error",
            )

    routing_off = evaluate_run(
        INDIRECT_ROUTING_OFF,
        "indirect",
        three_field=False,
    )
    plot_routing_fields(
        results[("indirect", "mixed")],
        routing_off,
        OUT / "04_ablation_summary" / "gradient_routing_primary_fields",
    )

    plot_mixed_three_metrics(
        results,
        OUT / "04_ablation_summary" / "mixed_vs_three_field_relative_l2",
    )
    plot_mixed_three_fields(
        results[("direct", "mixed")],
        results[("direct", "three-field")],
        OUT / "04_ablation_summary" / "direct_mixed_vs_three_field_fields",
    )
    plot_mixed_three_fields(
        results[("indirect", "mixed")],
        results[("indirect", "three-field")],
        OUT / "04_ablation_summary" / "converse_mixed_vs_three_field_fields",
    )
    rows = read_ablation_rows()
    table_dir = OUT / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    (table_dir / "mixed_vs_three_field_training_cost.md").write_text(
        architecture_cost_table(rows),
        encoding="utf-8",
    )
    plot_indirect_ablations(
        rows,
        OUT / "04_ablation_summary" / "indirect_ablation_relative_l2",
    )
    plot_load_sweep(
        DIRECT_SWEEP_CSV,
        "direct",
        OUT / "05_load_scaling" / "direct_force_scaling",
    )
    plot_load_sweep(
        INDIRECT_SWEEP_CSV,
        "indirect",
        OUT / "05_load_scaling" / "converse_voltage_scaling",
    )
    plot_direct_baseline_loss(
        OUT / "06_loss_curves" / "direct_mixed_loss_curve"
    )
    plot_indirect_baseline_loss(
        OUT / "06_loss_curves" / "indirect_mixed_loss_curve"
    )
    plot_loss_components(
        DIRECT_MIXED,
        "Direct-effect loss components",
        OUT / "06_loss_curves" / "direct_loss_components",
    )
    plot_loss_components(
        INDIRECT_MIXED,
        "Converse-effect loss components",
        OUT / "06_loss_curves" / "indirect_loss_components",
    )
    plot_three_field_losses(OUT / "06_loss_curves" / "mixed_vs_three_field_losses")
    plot_indirect_loss_ablations(OUT / "06_loss_curves" / "indirect_loss_ablations")

    (table_dir / "error_metrics.md").write_text(
        markdown_metrics(results, routing_off), encoding="utf-8"
    )
    metrics_json = {
        f"{formulation}_{model}": data["metrics"]
        for (formulation, model), data in results.items()
    }
    (table_dir / "error_metrics.json").write_text(
        json.dumps(metrics_json, indent=2), encoding="utf-8"
    )
    write_readme()
    copy_supplementary()
    print(OUT)


if __name__ == "__main__":
    main()
