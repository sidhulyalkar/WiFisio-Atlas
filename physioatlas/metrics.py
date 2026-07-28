from __future__ import annotations

import numpy as np


def masked_rmse(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    valid = np.broadcast_to(mask.astype(bool), target.shape)
    if not np.any(valid):
        return float("nan")
    return float(np.sqrt(np.mean((prediction[valid] - target[valid]) ** 2)))


def masked_mae(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    valid = np.broadcast_to(mask.astype(bool), target.shape)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.abs(prediction[valid] - target[valid])))


def masked_correlation(
    prediction: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> float:
    valid = np.broadcast_to(mask.astype(bool), target.shape)
    pred = prediction[valid]
    true = target[valid]
    if pred.size < 3 or np.std(pred) < 1e-12 or np.std(true) < 1e-12:
        return float("nan")
    return float(np.corrcoef(pred, true)[0, 1])


def prediction_interval_coverage(
    prediction: np.ndarray,
    target: np.ndarray,
    standard_deviation: np.ndarray,
    mask: np.ndarray,
    *,
    z: float = 1.645,
) -> float:
    valid = np.broadcast_to(mask.astype(bool), target.shape)
    if not np.any(valid):
        return float("nan")
    lower = prediction - z * standard_deviation
    upper = prediction + z * standard_deviation
    covered = (target >= lower) & (target <= upper)
    return float(np.mean(covered[valid]))


def binary_brier_score(probability: np.ndarray, target: np.ndarray) -> float:
    probability = np.asarray(probability, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if probability.shape != target.shape:
        target = np.broadcast_to(target, probability.shape)
    return float(np.mean((probability - target) ** 2))
