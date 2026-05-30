"""Generate the .npy geometry / collocation files expected by training.

Reproduces the artefacts produced by ``geom_creation.ipynb`` for both the
indirect (``_m1``) and direct (``_m1_d``) PINN formulations.

Usage:
    python -m scripts.generate_geometry
    python -m scripts.generate_geometry --n-points 400 --n-collocation 150
"""

from __future__ import annotations

import argparse

from pinn_piezo import geometry
from pinn_piezo.config import DATA_DIR


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-points", type=int, default=400)
    p.add_argument("--n-collocation", type=int, default=150)
    p.add_argument("--n-collocation-test", type=int, default=200)
    p.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    p.add_argument("--suffix", choices=["_m1", "_m1_d", "both"], default="both",
                   help="Which dataset suffix to generate.")
    return p.parse_args()


def main():
    args = parse_args()
    suffixes = ["_m1", "_m1_d"] if args.suffix == "both" else [args.suffix]
    for suffix in suffixes:
        print(f"Generating dataset with suffix '{suffix}' ...")
        geometry.generate_and_save(
            n_points=args.n_points,
            n_collocation=args.n_collocation,
            n_collocation_test=args.n_collocation_test,
            suffix=suffix,
            data_dir=args.data_dir,
        )
    print(f"Done. Files written to {args.data_dir}")


if __name__ == "__main__":
    main()
