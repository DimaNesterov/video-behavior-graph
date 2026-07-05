"""Trajectory prediction metrics: ADE, FDE, and their min-over-K variants."""
import numpy as np


def ade(pred: np.ndarray, gt: np.ndarray) -> float:
    """Average Displacement Error. pred, gt: (T, 2) in meters."""
    return float(np.linalg.norm(pred - gt, axis=-1).mean())


def fde(pred: np.ndarray, gt: np.ndarray) -> float:
    """Final Displacement Error: error at the last predicted point."""
    return float(np.linalg.norm(pred[-1] - gt[-1]))


def min_ade_fde(preds_k: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Best-of-K metrics for multi-modal output. preds_k: (K, T, 2), gt: (T, 2).

    Standard protocol: report ADE/FDE of the sample closest to ground truth.
    """
    ades = np.linalg.norm(preds_k - gt[None], axis=-1).mean(axis=-1)  # (K,)
    best = int(ades.argmin())
    return float(ades[best]), fde(preds_k[best], gt)