"""Train the indirect (voltage-driven) PINN.

Direct counterpart of ``PINN_pz_v3.ipynb``.

Usage:
    python -m scripts.train_indirect
    python -m scripts.train_indirect --epochs-adam 1000 --epochs-lbfgs 200
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torchsummary import summary

from pinn_piezo.config import DATA_DIR, RUNS_DIR, get_device
from pinn_piezo.indirect import model as model_mod
from pinn_piezo.indirect import train as train_mod


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs-adam", type=int, default=1000)
    p.add_argument("--epochs-lbfgs", type=int, default=200)
    p.add_argument("--lr-adam", type=float, default=0.001)
    p.add_argument("--lr-lbfgs", type=float, default=0.01)
    p.add_argument("--fraction", type=float, default=1.0,
                   help="Fraction of collocation points used.")
    p.add_argument("--f", type=int, default=500,
                   help="Adaptive-weight update interval.")
    p.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    p.add_argument("--suffix", type=str, default="_m1")
    p.add_argument("--model-type", choices=["pyramid", "uniform"],
                   default="pyramid")
    p.add_argument("--run-name", type=str, default=None,
                   help="Run identifier under outputs/runs/. Defaults to "
                        "'train_indirect_<timestamp>'.")
    return p.parse_args()


def main():
    args = parse_args()

    torch.set_default_dtype(torch.float64)

    device = get_device()
    print(f"Using device: {device}")

    run_name = args.run_name or (
        f"train_indirect_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    run_dir = RUNS_DIR / run_name
    ckpt_adam = run_dir / "checkpoints" / "ADAM"
    ckpt_lbfgs = run_dir / "checkpoints" / "LBFGS"
    models_dir = run_dir / "models"
    for d in (ckpt_adam, ckpt_lbfgs, models_dir):
        d.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    model = model_mod.build_default_model(device=device,
                                          model_type=args.model_type)
    try:
        summary(model, (2,))
    except Exception:
        # torchsummary on CPU sometimes fails on float64 inputs; non-critical.
        pass

    arrays = train_mod.load_dataset(args.data_dir,
                                    suffix=args.suffix,
                                    fraction=args.fraction)
    tensors = train_mod.to_device(arrays, device, dtype=torch.float64)
    print("Collocation shapes:",
          tensors["x_collocation"].shape, tensors["y_collocation"].shape)

    result = train_mod.train(
        model, tensors,
        epochs_adam=args.epochs_adam, epochs_lbfgs=args.epochs_lbfgs,
        lr_adam=args.lr_adam, lr_lbfgs=args.lr_lbfgs,
        f=args.f,
        checkpoints_adam_dir=ckpt_adam,
        checkpoints_lbfgs_dir=ckpt_lbfgs,
    )

    print("Best LBFGS loss:", result["best_loss_lbfgs"])

    save_path = models_dir / "model_PINN_indirect.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model state_dict saved to {save_path}")

    np.save(run_dir / "loss_indirect.npy", np.array(result["loss_list"]))


if __name__ == "__main__":
    main()
