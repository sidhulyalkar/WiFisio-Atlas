from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np

from .calibration_reflector_schema import (
    ReflectorCalibrationHealth,
    ReflectorTelemetrySample,
    SafetyState,
)
from .reflector_rf_atlas_schema import (
    ReflectorDriftReport,
    ReflectorLinkSensitivity,
    ReflectorRfAtlas,
    ReflectorRfSample,
    ReflectorWaypointResponse,
)


def _nearest_telemetry(
    timestamps: np.ndarray,
    samples: list[ReflectorTelemetrySample],
    timestamp: float,
) -> tuple[ReflectorTelemetrySample, float]:
    index = int(np.searchsorted(timestamps, timestamp))
    candidates = np.clip([index - 1, index], 0, timestamps.size - 1)
    selected = min(candidates, key=lambda item: abs(timestamps[item] - timestamp))
    return samples[int(selected)], abs(float(timestamps[int(selected)]) - timestamp)


def build_reflector_rf_atlas(
    telemetry: Iterable[ReflectorTelemetrySample],
    rf_samples: Iterable[ReflectorRfSample],
    health: ReflectorCalibrationHealth,
    *,
    atlas_id: str,
    environment_id: str,
    maximum_time_gap_s: float = 0.05,
    maximum_velocity: float = 0.10,
    maximum_position_error: float = 0.25,
    waypoint_resolution: float = 0.10,
    minimum_samples_per_response: int = 3,
) -> ReflectorRfAtlas:
    motion = list(telemetry)
    radio = list(rf_samples)
    if not health.valid_for_rf_calibration:
        raise ValueError("reflector actuator health is invalid for RF calibration")
    if not motion or not radio:
        raise ValueError("reflector atlas requires telemetry and RF samples")
    if {item.plan_id for item in motion} != {health.plan_id}:
        raise ValueError("reflector health and telemetry plan IDs do not match")
    receipts = {item.alignment_receipt_id for item in radio}
    if len(receipts) != 1:
        raise ValueError("one immutable alignment receipt is required per atlas")
    widths: dict[tuple[str, str], int] = {}
    timestamps = np.asarray([item.monotonic_timestamp_s for item in motion], dtype=float)
    accepted = []
    rejected = 0
    for sample in radio:
        key = (sample.source, sample.link_id)
        width = len(sample.response_features)
        if key in widths and widths[key] != width:
            raise ValueError(f"RF feature width changed for {key}")
        widths[key] = width
        motor, gap = _nearest_telemetry(
            timestamps, motion, sample.timestamp_reflector_clock_s
        )
        stable = (
            gap <= maximum_time_gap_s
            and not motor.motor_power_enabled
            and abs(motor.actual_velocity_per_s) <= maximum_velocity
            and abs(motor.actual_position - motor.commanded_position)
            <= maximum_position_error
            and motor.safety_state not in {SafetyState.FAULTED, SafetyState.ESTOPPED}
            and motor.fault_code is None
        )
        if stable:
            position = round(motor.commanded_position / waypoint_resolution) * waypoint_resolution
            accepted.append((position, sample))
        else:
            rejected += 1
    grouped: dict[tuple[float, str, str, str], list[ReflectorRfSample]] = defaultdict(list)
    for position, sample in accepted:
        grouped[
            (position, sample.source, sample.link_id, sample.frequency_band)
        ].append(sample)
    responses = []
    for (position, source, link_id, band), samples in sorted(grouped.items()):
        if len(samples) < minimum_samples_per_response:
            rejected += len(samples)
            continue
        features = np.asarray([item.response_features for item in samples], dtype=float)
        energies = np.asarray([item.response_energy for item in samples], dtype=float)
        responses.append(
            ReflectorWaypointResponse(
                position=position,
                source=source,
                link_id=link_id,
                frequency_band=band,
                mean_energy=float(np.mean(energies)),
                energy_std=float(np.std(energies)),
                mean_features=np.mean(features, axis=0).tolist(),
                samples=len(samples),
                mean_quality=float(np.mean([item.quality for item in samples])),
            )
        )
    if not responses:
        raise ValueError("no stable motor-off RF dwell responses passed the atlas gates")
    sensitivity = _link_sensitivity(responses)
    return ReflectorRfAtlas(
        atlas_id=atlas_id,
        environment_id=environment_id,
        plan_id=health.plan_id,
        alignment_receipt_id=next(iter(receipts)),
        waypoint_responses=responses,
        link_sensitivity=sensitivity,
        cross_band_energy_correlation=_group_correlation(responses, "frequency_band"),
        csi_bfi_energy_correlation=_group_correlation(responses, "source"),
        accepted_rf_samples=sum(item.samples for item in responses),
        rejected_rf_samples=rejected,
        motor_off_dwell_verified=True,
        synthetic=health.synthetic,
    )


def _link_sensitivity(
    responses: list[ReflectorWaypointResponse],
) -> list[ReflectorLinkSensitivity]:
    grouped: dict[tuple[str, str, str], list[ReflectorWaypointResponse]] = defaultdict(list)
    for item in responses:
        grouped[(item.source, item.link_id, item.frequency_band)].append(item)
    result = []
    for (source, link_id, band), items in sorted(grouped.items()):
        energies = np.asarray([item.mean_energy for item in items])
        noise = float(np.mean([item.energy_std for item in items]))
        dynamic_range = float(np.ptp(energies))
        threshold = float(np.min(energies) + 0.1 * dynamic_range)
        result.append(
            ReflectorLinkSensitivity(
                source=source,
                link_id=link_id,
                frequency_band=band,
                dynamic_range=dynamic_range,
                repeatability_noise=noise,
                sensitivity_ratio=dynamic_range / max(noise, 1e-9),
                low_response_positions=[
                    item.position for item in items if item.mean_energy <= threshold
                ],
            )
        )
    return result


def _group_correlation(
    responses: list[ReflectorWaypointResponse],
    group_field: str,
) -> float | None:
    groups = sorted({getattr(item, group_field) for item in responses})
    if len(groups) != 2:
        return None
    by_group = {}
    for group in groups:
        values: dict[float, list[float]] = defaultdict(list)
        for item in responses:
            if getattr(item, group_field) == group:
                values[item.position].append(item.mean_energy)
        by_group[group] = {key: float(np.mean(value)) for key, value in values.items()}
    positions = sorted(set(by_group[groups[0]]) & set(by_group[groups[1]]))
    if len(positions) < 3:
        return None
    left = np.asarray([by_group[groups[0]][item] for item in positions])
    right = np.asarray([by_group[groups[1]][item] for item in positions])
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def compare_reflector_atlases(
    baseline: ReflectorRfAtlas,
    candidate: ReflectorRfAtlas,
    *,
    maximum_drift_score: float = 0.35,
) -> ReflectorDriftReport:
    if baseline.environment_id != candidate.environment_id:
        raise ValueError("reflector atlases must describe the same environment")
    def key(item: ReflectorWaypointResponse) -> tuple[float, str, str, str]:
        return (
            item.position,
            item.source,
            item.link_id,
            item.frequency_band,
        )
    left = {key(item): item for item in baseline.waypoint_responses}
    right = {key(item): item for item in candidate.waypoint_responses}
    shared = sorted(set(left) & set(right))
    reasons = []
    if len(shared) < 3:
        reasons.append("insufficient_matched_responses")
        return ReflectorDriftReport(
            baseline_atlas_id=baseline.atlas_id,
            candidate_atlas_id=candidate.atlas_id,
            matched_responses=len(shared),
            recalibration_required=True,
            reasons=reasons,
        )
    baseline_energy = np.asarray([left[item].mean_energy for item in shared])
    candidate_energy = np.asarray([right[item].mean_energy for item in shared])
    scale = max(float(np.ptp(baseline_energy)), float(np.mean(np.abs(baseline_energy))), 1e-9)
    energy_rmse = float(np.sqrt(np.mean((baseline_energy - candidate_energy) ** 2)) / scale)
    cosine_distances = []
    for item in shared:
        a = np.asarray(left[item].mean_features)
        b = np.asarray(right[item].mean_features)
        if a.size != b.size:
            reasons.append("feature_width_changed")
            continue
        cosine_distances.append(
            1.0 - float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
        )
    feature_distance = None if not cosine_distances else float(np.mean(cosine_distances))
    drift = energy_rmse + (feature_distance or 0.0)
    if drift > maximum_drift_score:
        reasons.append("reflector_transfer_drift_exceeded")
    return ReflectorDriftReport(
        baseline_atlas_id=baseline.atlas_id,
        candidate_atlas_id=candidate.atlas_id,
        matched_responses=len(shared),
        normalized_energy_rmse=energy_rmse,
        feature_cosine_distance=feature_distance,
        drift_score=drift,
        recalibration_required=bool(reasons),
        reasons=sorted(set(reasons)),
    )
