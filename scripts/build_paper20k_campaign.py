"""Evaluate the complete 20k campaign and build the publication package."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pinn_piezo.config import OUTPUTS_DIR, RUNS_DIR
from scripts.aggregate_stable_direct_force_sweep import collect as collect_force
from scripts.aggregate_stable_indirect_voltage_sweep import (
    collect as collect_voltage,
)
from scripts.build_results_package import (
    COLORS,
    evaluate_run,
    load_loss,
    plot_error_maps,
    plot_fem_pinn_normalized_error,
    plot_load_sweep,
    plot_loss_components,
    plot_mixed_three_fields,
    plot_primary_fields,
    plot_routing_fields,
    save_figure,
    set_style,
)
from scripts.run_paper20k_campaign import EXPERIMENTS, run_complete


SUMMARY = RUNS_DIR / "paper20k_summary"


def formulation(name: str) -> str:
    return "direct" if name.startswith("direct_") else "indirect"


def is_three_field(name: str) -> bool:
    return name.endswith("three_field")


def architecture_label(config: dict) -> str:
    widths = config["hidden_sizes"]
    if len(set(widths)) == 1:
        return f"{len(widths)}x{widths[0]}"
    return str(widths)


def evaluate_experiment(name: str) -> dict:
    experiment = EXPERIMENTS[name]
    run_dir = RUNS_DIR / experiment.run_name
    config = json.loads((run_dir / "config.json").read_text())
    summary = json.loads((run_dir / "training_summary.json").read_text())
    effect = formulation(name)
    three_field = is_three_field(name)
    data = evaluate_run(run_dir, effect, three_field=three_field)
    payload = {
        "run": experiment.run_name,
        "experiment": name,
        "group": experiment.group,
        "formulation": effect,
        "fields": "3 (u,v,phi)" if three_field else "8 (mixed)",
        "architecture": architecture_label(config),
        "activation": config["activation"],
        "interior_per_layer": config["interior_per_layer"],
        "seed": config["seed"],
        "gradient_routing": config.get(
            "mechanical_constitutive_stopgrad"
        ),
        "parameter_count": summary["parameter_count"],
        "training_time_s": summary["total_time_seconds"],
        "training_time_h": summary["total_time_seconds"] / 3600.0,
        "epochs_adam": config["epochs_adam"],
        "epochs_lbfgs": config["epochs_lbfgs"],
        "metrics": data["metrics"],
    }
    SUMMARY.mkdir(parents=True, exist_ok=True)
    (SUMMARY / f"{name}_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8",
    )
    return payload


def flatten(payload: dict) -> dict:
    row = {key: value for key, value in payload.items() if key != "metrics"}
    for field in ("u", "v", "phi"):
        for metric in ("rel_L2", "RMSE", "MAE", "max_abs", "nRMSE"):
            row[f"{field}_{metric}"] = payload["metrics"][field][metric]
    return row


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_sweep_csvs() -> tuple[Path, Path]:
    direct_definitions = (
        (0.05, "paper20k_direct_force_0p05N"),
        (0.10, "paper20k_direct_mixed"),
        (0.20, "paper20k_direct_force_0p2N"),
    )
    force_rows = [
        collect_force(RUNS_DIR / run_name, magnitude)[0]
        for magnitude, run_name in direct_definitions
    ]
    direct_csv = SUMMARY / "direct_force_sweep_20k.csv"
    write_csv(force_rows, direct_csv)

    voltage_definitions = (
        (100, "paper20k_indirect_mixed"),
        (200, "paper20k_indirect_voltage_200V"),
        (300, "paper20k_indirect_voltage_300V"),
        (400, "paper20k_indirect_voltage_400V"),
        (500, "paper20k_indirect_voltage_500V"),
    )
    voltage_rows = [
        collect_voltage(RUNS_DIR / run_name, voltage)[0]
        for voltage, run_name in voltage_definitions
    ]
    indirect_csv = SUMMARY / "indirect_voltage_sweep_20k.csv"
    write_csv(voltage_rows, indirect_csv)
    return direct_csv, indirect_csv


def build_package(ablation_csv: Path, direct_csv: Path,
                  indirect_csv: Path) -> Path:
    output = OUTPUTS_DIR / "paper20k_results_package"
    environment = os.environ.copy()
    environment.update({
        "PINN_PIEZO_DIRECT_MIXED_RUN": "paper20k_direct_mixed",
        "PINN_PIEZO_INDIRECT_MIXED_RUN": "paper20k_indirect_mixed",
        "PINN_PIEZO_DIRECT_THREE_RUN": "paper20k_direct_three_field",
        "PINN_PIEZO_INDIRECT_THREE_RUN": "paper20k_indirect_three_field",
        "PINN_PIEZO_INDIRECT_ROUTING_OFF_RUN": "paper20k_indirect_routing_off",
        "PINN_PIEZO_INDIRECT_TANH_RUN": "paper20k_indirect_tanh_3x50",
        "PINN_PIEZO_ABLATION_CSV": str(ablation_csv),
        "PINN_PIEZO_DIRECT_SWEEP_CSV": str(direct_csv),
        "PINN_PIEZO_INDIRECT_SWEEP_CSV": str(indirect_csv),
        "PINN_PIEZO_RESULTS_PACKAGE_DIR": str(output),
    })
    subprocess.run(
        [sys.executable, "-u", "-m", "scripts.build_results_package"],
        check=True, env=environment,
    )
    return output


def _experiment_label(name: str) -> str:
    return (
        name.replace("indirect_", "")
        .replace("direct_", "")
        .replace("_", " ")
        .replace("mixed", "Mixed")
        .replace("three field", "Three-field")
    )


def plot_partial_metrics(payloads: list[dict], group: str,
                         output: Path) -> None:
    selected = [item for item in payloads if item["group"] == group]
    if group != "baseline" and not selected:
        return
    baseline = next(
        (item for item in payloads if item["experiment"] == "indirect_mixed"),
        None,
    )
    direct_baseline = next(
        (item for item in payloads if item["experiment"] == "direct_mixed"),
        None,
    )
    if group in {
        "architecture", "activation", "gradient_routing",
        "collocation_sensitivity",
    } and baseline is not None:
        selected = [baseline, *selected]
    if group == "mixed_vs_three_field":
        selected = [
            item for item in (direct_baseline, baseline, *selected)
            if item is not None
        ]
    if not selected:
        return
    x = np.arange(len(selected))
    width = 0.24
    fig, axis = plt.subplots(
        figsize=(max(6.4, 1.35 * len(selected)), 4.2),
        constrained_layout=True,
    )
    for offset, field in zip((-1, 0, 1), ("u", "v", "phi")):
        label = r"$\varphi$" if field == "phi" else f"${field}$"
        axis.bar(
            x + offset * width,
            [item["metrics"][field]["rel_L2"] for item in selected],
            width,
            label=label,
            color=COLORS[field],
        )
    axis.set_xticks(
        x,
        [_experiment_label(item["experiment"]) for item in selected],
        rotation=20,
        ha="right",
    )
    axis.set_ylabel(r"Relative $L^2$ error")
    axis.set_title(f"Completed 20k runs — {group.replace('_', ' ')}")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=3)
    save_figure(fig, output)


def plot_total_loss(run_dir: Path, title: str, output: Path) -> None:
    loss = load_loss(run_dir)
    fig, axis = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    axis.semilogy(
        np.arange(1, loss.size + 1), loss,
        color=COLORS["mixed"], linewidth=0.9,
    )
    axis.set_xlabel("Adam epoch")
    axis.set_ylabel("Total physics loss")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    save_figure(fig, output)


def build_partial_sweeps(completed: list[str], output: Path) -> None:
    force_definitions = (
        ("direct_force_0p05N", 0.05, "paper20k_direct_force_0p05N"),
        ("direct_mixed", 0.10, "paper20k_direct_mixed"),
        ("direct_force_0p2N", 0.20, "paper20k_direct_force_0p2N"),
    )
    force_rows = [
        collect_force(RUNS_DIR / run_name, magnitude)[0]
        for name, magnitude, run_name in force_definitions
        if name in completed
    ]
    if force_rows:
        force_csv = SUMMARY / "direct_force_sweep_partial.csv"
        write_csv(force_rows, force_csv)
        if len(force_rows) >= 2:
            plot_load_sweep(
                force_csv, "direct",
                output / "05_load_scaling" / "direct_force_scaling_partial",
            )

    voltage_definitions = (
        ("indirect_mixed", 100, "paper20k_indirect_mixed"),
        ("indirect_voltage_200V", 200, "paper20k_indirect_voltage_200V"),
        ("indirect_voltage_300V", 300, "paper20k_indirect_voltage_300V"),
        ("indirect_voltage_400V", 400, "paper20k_indirect_voltage_400V"),
        ("indirect_voltage_500V", 500, "paper20k_indirect_voltage_500V"),
    )
    voltage_rows = [
        collect_voltage(RUNS_DIR / run_name, voltage)[0]
        for name, voltage, run_name in voltage_definitions
        if name in completed
    ]
    if voltage_rows:
        voltage_csv = SUMMARY / "indirect_voltage_sweep_partial.csv"
        write_csv(voltage_rows, voltage_csv)
        if len(voltage_rows) >= 2:
            plot_load_sweep(
                voltage_csv, "indirect",
                output / "05_load_scaling" / "converse_voltage_scaling_partial",
            )


def build_partial_package(payloads: list[dict], completed: list[str],
                          missing: list[str], ablation_csv: Path) -> Path:
    """Generate every figure currently supported by completed runs."""
    output = OUTPUTS_DIR / "paper20k_results_package"
    set_style()
    output.mkdir(parents=True, exist_ok=True)
    evaluated: dict[str, dict] = {}
    for payload in payloads:
        name = payload["experiment"]
        experiment = EXPERIMENTS[name]
        run_dir = RUNS_DIR / experiment.run_name
        data = evaluate_run(
            run_dir,
            payload["formulation"],
            three_field=is_three_field(name),
        )
        evaluated[name] = data
        stem = output / "01_completed_fields" / name
        plot_fem_pinn_normalized_error(
            data, stem.with_name(f"{name}_fem_pinn_normalized_error"),
        )
        plot_primary_fields(
            data, stem.with_name(f"{name}_primary_fields"),
        )
        plot_error_maps(
            data,
            output / "02_completed_error_maps" / f"{name}_error_maps",
        )
        plot_total_loss(
            run_dir,
            f"{_experiment_label(name)} — training loss",
            output / "06_loss_curves" / f"{name}_loss_curve",
        )
        plot_loss_components(
            run_dir,
            f"{_experiment_label(name)} — loss components",
            output / "06_loss_curves" / f"{name}_loss_components",
        )

    comparison_dir = output / "04_ablation_summary"
    mixed_three_pairs = (
        (
            "direct_mixed",
            "direct_three_field",
            "direct_mixed_vs_three_field_fields",
        ),
        (
            "indirect_mixed",
            "indirect_three_field",
            "converse_mixed_vs_three_field_fields",
        ),
    )
    for mixed_name, three_name, filename in mixed_three_pairs:
        if mixed_name in evaluated and three_name in evaluated:
            plot_mixed_three_fields(
                evaluated[mixed_name],
                evaluated[three_name],
                comparison_dir / filename,
            )
    if "indirect_mixed" in evaluated and "indirect_routing_off" in evaluated:
        plot_routing_fields(
            evaluated["indirect_mixed"],
            evaluated["indirect_routing_off"],
            comparison_dir / "gradient_routing_primary_fields",
        )

    for group in (
        "baseline", "mixed_vs_three_field", "architecture", "activation",
        "gradient_routing", "collocation_sensitivity",
    ):
        plot_partial_metrics(
            payloads,
            group,
            output / "04_ablation_summary" / f"{group}_partial",
        )
    build_partial_sweeps(completed, output)

    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ablation_csv, tables / "ablation_results_partial.csv")
    shutil.copy2(
        SUMMARY / "all_metrics.json", tables / "all_metrics_partial.json",
    )
    status = [
        "# Paper 20k — partial results", "",
        f"Completed: {len(completed)}/{len(EXPERIMENTS)}", "",
        "## Completed", "",
        *[f"- {name}" for name in completed], "",
        "## Pending", "",
        *[f"- {name}" for name in missing], "",
        "Regenerate with the same command as more runs finish; existing "
        "figures are updated and new comparisons are added.",
    ]
    (output / "PARTIAL_STATUS.md").write_text(
        "\n".join(status), encoding="utf-8",
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="Evaluate completed runs even when the campaign is incomplete.",
    )
    parser.add_argument(
        "--no-package", action="store_true",
        help="Only evaluate and aggregate; do not generate figures.",
    )
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    missing = [
        name for name, experiment in EXPERIMENTS.items()
        if not run_complete(experiment)
    ]
    if missing and not args.allow_partial:
        raise RuntimeError(
            "Campaign is incomplete. Missing: " + ", ".join(missing)
        )
    completed = [name for name in EXPERIMENTS if name not in missing]
    if not completed:
        raise RuntimeError("No completed campaign models were found")
    if missing:
        print(
            f"Partial campaign: evaluating {len(completed)}/{len(EXPERIMENTS)} "
            "completed runs."
        )
        print("Missing: " + ", ".join(missing))

    SUMMARY.mkdir(parents=True, exist_ok=True)
    payloads = [evaluate_experiment(name) for name in completed]
    rows = [flatten(payload) for payload in payloads]
    ablation_csv = SUMMARY / "ablation_results.csv"
    write_csv(rows, ablation_csv)
    (SUMMARY / "all_metrics.json").write_text(
        json.dumps(payloads, indent=2), encoding="utf-8",
    )
    print(f"Metrics: {SUMMARY / 'all_metrics.json'}")
    print(f"Table: {ablation_csv}")
    if missing and not args.no_package:
        output = build_partial_package(
            payloads, completed, missing, ablation_csv,
        )
        print(f"Partial figure package: {output}")
    elif not missing:
        direct_csv, indirect_csv = build_sweep_csvs()
    if not args.no_package and not missing:
        output = build_package(ablation_csv, direct_csv, indirect_csv)
        print(f"Publication package: {output}")


if __name__ == "__main__":
    main()
