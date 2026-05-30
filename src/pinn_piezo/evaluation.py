"""FEM ground-truth comparison utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_FEM_ground_truth(csv_path: str | Path):
    df = pd.read_csv(csv_path)
    X = df['X_Coordinate'].values.reshape(-1, 1)
    Y = df['Y_Coordinate'].values.reshape(-1, 1)
    U = np.array(df['X_Deflection'].values)
    V = np.array(df['Y_Deflection'].values)
    Phi = np.array(df['Potential'].values)
    X_ground_truth = np.hstack((X, Y))
    return X_ground_truth, U, V, Phi


def relative_l2_error(u_pred, u_ref):
    return np.linalg.norm(u_pred - u_ref, 2) / np.linalg.norm(u_ref, 2)


def evaluate_against_FEM(model, X_ground_truth, U, V, Phi, tensorize):
    X_tensor = tensorize(X_ground_truth)
    preds_gr = model(X_tensor).detach().cpu().numpy()
    u_pred_gr = preds_gr[:, 0]
    v_pred_gr = preds_gr[:, 1]
    phi_pred_gr = preds_gr[:, 2]

    l2_u = np.linalg.norm(u_pred_gr - U) / np.linalg.norm(U)
    l2_v = np.linalg.norm(v_pred_gr - V) / np.linalg.norm(V)
    l2_phi = np.linalg.norm(phi_pred_gr - Phi) / np.linalg.norm(Phi)

    print(f"Relative L2 error (u): {l2_u:.4e}")
    print(f"Relative L2 error (v): {l2_v:.4e}")
    print(f"Relative L2 error (phi): {l2_phi:.4e}")

    return {
        'preds': preds_gr,
        'u_pred': u_pred_gr,
        'v_pred': v_pred_gr,
        'phi_pred': phi_pred_gr,
        'l2_u': l2_u,
        'l2_v': l2_v,
        'l2_phi': l2_phi,
    }
