from __future__ import annotations

import math

import numpy as np

from .schema import AnatomyRegion, SensorGeometry


def _vec(sensor_position) -> np.ndarray:
    return np.asarray(sensor_position.as_tuple(), dtype=np.float32)


def point_to_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom <= 1e-12:
        return float(np.linalg.norm(point - start))
    projection = float(np.dot(point - start, segment) / denom)
    projection = min(1.0, max(0.0, projection))
    closest = start + projection * segment
    return float(np.linalg.norm(point - closest))


def link_feature_matrix(geometry: SensorGeometry) -> tuple[list[str], np.ndarray]:
    """Create deterministic physical features for each RF link.

    Features are TX position, RX position, unit direction, link length,
    log-frequency, and log-bandwidth. They are normalized only by simple units,
    keeping the transform auditable and independent of training data.
    """
    sensors = {sensor.sensor_id: sensor for sensor in geometry.sensors}
    rows: list[np.ndarray] = []
    ids: list[str] = []
    for link in geometry.rf_links:
        tx = sensors[link.tx_sensor_id]
        rx = sensors[link.rx_sensor_id]
        if tx.position_m is None or rx.position_m is None:
            raise ValueError(f"Link {link.link_id} requires positioned TX/RX sensors")
        tx_pos = _vec(tx.position_m)
        rx_pos = _vec(rx.position_m)
        delta = rx_pos - tx_pos
        length = float(np.linalg.norm(delta))
        direction = delta / max(length, 1e-6)
        row = np.concatenate(
            [
                tx_pos,
                rx_pos,
                direction,
                np.asarray(
                    [
                        length,
                        math.log10(link.center_frequency_hz),
                        math.log10(link.bandwidth_hz),
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        ids.append(link.link_id)
        rows.append(row.astype(np.float32))
    if not rows:
        return [], np.empty((0, 12), dtype=np.float32)
    return ids, np.stack(rows)


def anatomy_link_prior(
    geometry: SensorGeometry,
    *,
    sigma_m: float = 0.45,
) -> tuple[list[AnatomyRegion], np.ndarray]:
    """Soft prior connecting RF links to anatomical anchor locations.

    This is not an organ localization result. It is a geometric inductive prior
    based on operator-supplied body anchors and sensor positions.
    """
    if sigma_m <= 0:
        raise ValueError("sigma_m must be positive")
    sensors = {sensor.sensor_id: sensor for sensor in geometry.sensors}
    regions = sorted(geometry.anatomy_anchors_m, key=lambda item: item.value)
    matrix = np.zeros((len(regions), len(geometry.rf_links)), dtype=np.float32)
    for region_idx, region in enumerate(regions):
        point = _vec(geometry.anatomy_anchors_m[region])
        for link_idx, link in enumerate(geometry.rf_links):
            tx = sensors[link.tx_sensor_id]
            rx = sensors[link.rx_sensor_id]
            if tx.position_m is None or rx.position_m is None:
                continue
            distance = point_to_segment_distance(
                point, _vec(tx.position_m), _vec(rx.position_m)
            )
            matrix[region_idx, link_idx] = math.exp(
                -(distance * distance) / (2.0 * sigma_m * sigma_m)
            )
        total = float(matrix[region_idx].sum())
        if total > 0:
            matrix[region_idx] /= total
    return regions, matrix


def geometry_summary(geometry: SensorGeometry) -> np.ndarray:
    """Return a fixed-width geometry context vector for neural models.

    The summary contains mean and standard deviation of the 12 link features,
    followed by one anchor-presence bit per ``AnatomyRegion``. It is a compact
    baseline representation, not a replacement for future per-link attention.
    """
    _, features = link_feature_matrix(geometry)
    if features.size:
        aggregate = np.concatenate([features.mean(axis=0), features.std(axis=0)])
    else:
        aggregate = np.zeros(24, dtype=np.float32)
    anchors = np.asarray(
        [1.0 if region in geometry.anatomy_anchors_m else 0.0 for region in AnatomyRegion],
        dtype=np.float32,
    )
    return np.concatenate([aggregate.astype(np.float32), anchors])
