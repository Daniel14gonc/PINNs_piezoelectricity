"""Error metrics for comparing PINN predictions against a reference field.

Added for the reviewer revision (Cluster 7 "additional metrics"). The
paper originally reported only the point-wise relative error and the
global relative L2 error; reviewers asked for RMSE / MAE / maximum
absolute error as well. These helpers keep every notebook reporting the
same, consistent set of numbers.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np


def _flat(a) -> np.ndarray:
    return np.asarray(a, dtype=float).reshape(-1)


def relative_l2_error(pred, ref) -> float:
    """Global relative L2 error ``||pred - ref|| / ||ref||`` (Eq. 39)."""
    pred, ref = _flat(pred), _flat(ref)
    denom = np.linalg.norm(ref)
    return float(np.linalg.norm(pred - ref) / denom) if denom > 0 else float("nan")


def rmse(pred, ref) -> float:
    pred, ref = _flat(pred), _flat(ref)
    return float(np.sqrt(np.mean((pred - ref) ** 2)))


def mae(pred, ref) -> float:
    pred, ref = _flat(pred), _flat(ref)
    return float(np.mean(np.abs(pred - ref)))


def max_abs_error(pred, ref) -> float:
    pred, ref = _flat(pred), _flat(ref)
    return float(np.max(np.abs(pred - ref)))


def normalized_rmse(pred, ref) -> float:
    """RMSE normalised by the peak-to-peak range of the reference field."""
    ref = _flat(ref)
    rng = float(np.max(ref) - np.min(ref))
    return float(rmse(pred, ref) / rng) if rng > 0 else float("nan")


def field_metrics(pred, ref) -> dict[str, float]:
    """Return every metric for a single field as a dictionary."""
    return {
        "rel_L2": relative_l2_error(pred, ref),
        "RMSE": rmse(pred, ref),
        "MAE": mae(pred, ref),
        "max_abs": max_abs_error(pred, ref),
        "nRMSE": normalized_rmse(pred, ref),
    }


def metrics_table(preds: Mapping[str, np.ndarray],
                  refs: Mapping[str, np.ndarray],
                  fields: Iterable[str] = ("u", "v", "phi")):
    """Build a tidy ``pandas.DataFrame`` (one row per field).

    ``preds`` / ``refs`` map a field name (e.g. ``"u"``) to a 1-D array.
    Importing pandas lazily keeps the module usable without it.
    """
    import pandas as pd

    rows = []
    for f in fields:
        if f not in preds or f not in refs:
            continue
        m = field_metrics(preds[f], refs[f])
        rows.append({"field": f, **m})
    return pd.DataFrame(rows).set_index("field")
