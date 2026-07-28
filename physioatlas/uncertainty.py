from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Union

import numpy as np


@dataclass(frozen=True)
class ConformalCalibration:
    alpha: float
    residual_quantile: float
    calibration_samples: int

    def to_dict(self) -> dict[str, Union[float, int]]:
        return asdict(self)


def fit_split_conformal(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    alpha: float = 0.1,
    observed: Optional[np.ndarray] = None,
) -> ConformalCalibration:
    """Fit a finite-sample split-conformal absolute-residual interval."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    truth_array = np.asarray(truth, dtype=np.float64).reshape(-1)
    prediction_array = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if truth_array.shape != prediction_array.shape:
        raise ValueError("truth and prediction must have the same shape")
    valid = np.isfinite(truth_array) & np.isfinite(prediction_array)
    if observed is not None:
        observed_array = np.asarray(observed).reshape(-1)
        if observed_array.shape != truth_array.shape:
            raise ValueError("observed mask must match truth")
        valid &= observed_array.astype(bool)
    residuals = np.abs(truth_array[valid] - prediction_array[valid])
    if residuals.size < 2:
        raise ValueError("At least two finite observed calibration samples are required")
    # Standard finite-sample conformal rank: ceil((n+1)*(1-alpha))/n.
    quantile_level = min(1.0, np.ceil((residuals.size + 1) * (1.0 - alpha)) / residuals.size)
    quantile = float(np.quantile(residuals, quantile_level, method="higher"))
    return ConformalCalibration(
        alpha=float(alpha),
        residual_quantile=quantile,
        calibration_samples=int(residuals.size),
    )


def conformal_interval(
    prediction: np.ndarray,
    calibration: ConformalCalibration,
) -> tuple[np.ndarray, np.ndarray]:
    prediction_array = np.asarray(prediction, dtype=np.float64)
    radius = calibration.residual_quantile
    return prediction_array - radius, prediction_array + radius


def interval_coverage(
    truth: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    observed: Optional[np.ndarray] = None,
) -> float:
    truth_array = np.asarray(truth, dtype=np.float64)
    lower_array = np.asarray(lower, dtype=np.float64)
    upper_array = np.asarray(upper, dtype=np.float64)
    if truth_array.shape != lower_array.shape or truth_array.shape != upper_array.shape:
        raise ValueError("truth, lower, and upper must have identical shapes")
    valid = np.isfinite(truth_array) & np.isfinite(lower_array) & np.isfinite(upper_array)
    if observed is not None:
        observed_array = np.asarray(observed)
        if observed_array.shape != truth_array.shape:
            raise ValueError("observed mask must match truth")
        valid &= observed_array.astype(bool)
    if not valid.any():
        return float("nan")
    covered = (truth_array[valid] >= lower_array[valid]) & (truth_array[valid] <= upper_array[valid])
    return float(covered.mean())


def should_abstain(
    *,
    observability: float,
    interval_width: float,
    observability_threshold: float = 0.6,
    max_interval_width: float = float("inf"),
    out_of_distribution_score: float = 0.0,
    max_ood_score: float = 1.0,
) -> bool:
    """Fail closed when observability, uncertainty, or OOD support is inadequate."""
    values = (observability, interval_width, out_of_distribution_score)
    if not all(np.isfinite(value) for value in values):
        return True
    return bool(
        observability < observability_threshold
        or interval_width > max_interval_width
        or out_of_distribution_score > max_ood_score
    )
