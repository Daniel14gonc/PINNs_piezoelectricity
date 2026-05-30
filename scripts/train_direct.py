"""Train the direct (force-driven) PINN.

Direct counterpart of ``PINN_pz_v3_directo.ipynb``.

Usage:
    python -m scripts.train_direct
    python -m scripts.train_direct --epochs-adam 3000
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torchsummary import summary

from pinn_piezo.config import DATA_DIR, RUNS_DIR, get_device
from pinn_piezo.direct import model as model_mod
from pinn_piezo.direct import train as train_mod


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs-adam", type=int, default=3000)
    p.add_argument("--epochs-lbfgs", type=int, default=0)
    p.add_argument("--lr-adam", type=float, default=0.005)
    p.add_argument("--lr-lbfgs", type=float, default=0.0001)
    p.add_argument("--fraction", type=float, default=0.75)
    p.add_argument("--f", type=int, default=200)
    p.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    p.add_argument("--suffix", type=str, default="_m1_d")
    p.add_argument("--run-name", type=str, default=None,
                   help="Run identifier under outputs/runs/. Defaults to "
                        "'train_direct_<timestamp>'.")
    return p.parse_args()


def main():
    args = parse_args()

    device = get_device()
    print(f"Using device: {device}")

    run_name = args.run_name or (
        f"train_direct_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    run_dir = RUNS_DIR / run_name
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    model = model_mod.build_default_model(device=device)
    try:
        summary(model, (2,))
    except Exception:
        pass

    arrays = train_mod.load_dataset(args.data_dir,
                                    suffix=args.suffix,
                                    fraction=args.fraction)
    tensors = train_mod.to_device(arrays, device, dtype=torch.float32)
    print("Collocation shapes:",
          tensors["x_collocation"].shape, tensors["y_collocation"].shape)

    result = train_mod.train(
        model, tensors,
        epochs_adam=args.epochs_adam, epochs_lbfgs=args.epochs_lbfgs,
        lr_adam=args.lr_adam, lr_lbfgs=args.lr_lbfgs,
        f=args.f,
    )

    save_path = models_dir / "model_PINN_direct.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model state_dict saved to {save_path}")

    np.save(run_dir / "loss_direct.npy", np.array(result["loss_list"]))


if __name__ == "__main__":
    main()
