from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Union

import numpy as np

from .io import SignalStream


@dataclass(frozen=True)
class ClockCalibration:
    """Affine mapping from a sensor clock into the reference clock.

    ``reference_time = scale * sensor_time + offset_s``. Drift is reported in
    parts per million to make hardware instability visible in run records.
    """

    source_clock: str
    reference_clock: str
    scale: float
    offset_s: float
    drift_ppm: float
    rmse_s: float
    n_events: int
    inlier_fraction: float

    def transform(self, timestamps_s: np.ndarray) -> np.ndarray:
        return self.scale * np.asarray(timestamps_s, dtype=np.float64) + self.offset_s


@dataclass(frozen=True)
class RigidTransform:
    source_frame: str
    target_frame: str
    rotation: list[list[float]]
    translation_m: list[float]
    rmse_m: float
    n_points: int

    def apply(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        rotation = np.asarray(self.rotation, dtype=np.float64)
        translation = np.asarray(self.translation_m, dtype=np.float64)
        return points @ rotation.T + translation


def estimate_clock_calibration(
    source_events_s: Iterable[float],
    reference_events_s: Iterable[float],
    *,
    source_clock: str = "sensor",
    reference_clock: str = "host",
    max_iterations: int = 4,
    outlier_sigma: float = 4.0,
) -> ClockCalibration:
    source = np.asarray(list(source_events_s), dtype=np.float64)
    reference = np.asarray(list(reference_events_s), dtype=np.float64)
    if source.shape != reference.shape or source.ndim != 1:
        raise ValueError("source and reference events must be equal-length vectors")
    if source.size < 2:
        raise ValueError("At least two synchronization events are required")
    if np.any(np.diff(source) <= 0) or np.any(np.diff(reference) <= 0):
        raise ValueError("Synchronization events must be strictly increasing")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(reference)):
        raise ValueError("Synchronization events must be finite")

    mask = np.ones(source.size, dtype=bool)
    coefficients = np.asarray([1.0, reference[0] - source[0]], dtype=np.float64)
    for _ in range(max_iterations):
        if mask.sum() < 2:
            raise ValueError("Clock calibration rejected too many synchronization events")
        design = np.stack([source[mask], np.ones(mask.sum())], axis=1)
        coefficients, *_ = np.linalg.lstsq(design, reference[mask], rcond=None)
        residuals = reference - (coefficients[0] * source + coefficients[1])
        median = float(np.median(residuals[mask]))
        mad = float(np.median(np.abs(residuals[mask] - median)))
        robust_sigma = max(1.4826 * mad, 1e-9)
        new_mask = np.abs(residuals - median) <= outlier_sigma * robust_sigma
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask

    residuals = reference - (coefficients[0] * source + coefficients[1])
    rmse = float(np.sqrt(np.mean(residuals[mask] ** 2)))
    scale = float(coefficients[0])
    return ClockCalibration(
        source_clock=source_clock,
        reference_clock=reference_clock,
        scale=scale,
        offset_s=float(coefficients[1]),
        drift_ppm=(scale - 1.0) * 1e6,
        rmse_s=rmse,
        n_events=int(source.size),
        inlier_fraction=float(mask.mean()),
    )


def estimate_signal_offset(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    sample_rate_hz: float,
    max_lag_s: float,
) -> tuple[float, float]:
    """Estimate a constant lag with normalized cross-correlation.

    Returns ``(offset_s, peak_correlation)`` where a positive offset means the
    moving signal should be shifted forward in time to match the reference.
    """
    if sample_rate_hz <= 0 or max_lag_s <= 0:
        raise ValueError("sample_rate_hz and max_lag_s must be positive")
    a = np.asarray(reference, dtype=np.float64).reshape(-1)
    b = np.asarray(moving, dtype=np.float64).reshape(-1)
    n = min(a.size, b.size)
    if n < 4:
        raise ValueError("Signals require at least four samples")
    a = np.nan_to_num(a[:n] - np.nanmean(a[:n]))
    b = np.nan_to_num(b[:n] - np.nanmean(b[:n]))
    a_std = float(np.std(a))
    b_std = float(np.std(b))
    if a_std < 1e-12 or b_std < 1e-12:
        raise ValueError("Signals must have non-zero variance")
    max_lag = min(int(round(max_lag_s * sample_rate_hz)), n - 2)
    best_lag = 0
    best_corr = -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            ref_slice, mov_slice = a[lag:], b[: n - lag]
        else:
            ref_slice, mov_slice = a[: n + lag], b[-lag:]
        if ref_slice.size < 3:
            continue
        corr = float(np.dot(ref_slice, mov_slice) / (np.linalg.norm(ref_slice) * np.linalg.norm(mov_slice) + 1e-12))
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    return best_lag / sample_rate_hz, best_corr


def apply_clock_calibration(stream: SignalStream, calibration: ClockCalibration) -> SignalStream:
    calibrated = SignalStream(
        timestamps_s=calibration.transform(stream.timestamps_s),
        values=np.asarray(stream.values).copy(),
        quality=None if stream.quality is None else np.asarray(stream.quality).copy(),
    )
    calibrated.validate()
    return calibrated


def estimate_rigid_transform(
    source_points_m: np.ndarray,
    target_points_m: np.ndarray,
    *,
    source_frame: str = "sensor",
    target_frame: str = "room_meters",
) -> RigidTransform:
    source = np.asarray(source_points_m, dtype=np.float64)
    target = np.asarray(target_points_m, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target points must both have shape [N, 3]")
    if source.shape[0] < 3:
        raise ValueError("At least three point correspondences are required")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        raise ValueError("Calibration points must be finite")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    predicted = source @ rotation.T + translation
    rmse = float(np.sqrt(np.mean(np.sum((predicted - target) ** 2, axis=1))))
    return RigidTransform(
        source_frame=source_frame,
        target_frame=target_frame,
        rotation=rotation.tolist(),
        translation_m=translation.tolist(),
        rmse_m=rmse,
        n_points=int(source.shape[0]),
    )


def save_calibration(path: Union[str, Path], calibration: Union[ClockCalibration, RigidTransform]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(calibration)
    payload["kind"] = "clock" if isinstance(calibration, ClockCalibration) else "rigid"
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
