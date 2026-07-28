from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .household_bfi_schema import NormalizedBfiFeature
from .rf_fusion_schema import (
    CsiBfiFusionConfig,
    FusionDecision,
    ZoneEvidence,
    ZonePrototypeModel,
)
from .utils import make_json_safe


def _normalized(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=float).reshape(-1)
    if array.size < 2 or not np.all(np.isfinite(array)):
        raise ValueError("feature must contain at least two finite values")
    return array / max(float(np.linalg.norm(array)), 1e-12)


def fit_zone_prototype_model(
    samples: Iterable[dict[str, Any]],
    *,
    source: str,
    feature_version: str,
) -> ZonePrototypeModel:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for payload in samples:
        if payload.get("source") != source:
            raise ValueError("all prototype samples must use the declared source")
        if payload.get("feature_version") != feature_version:
            raise ValueError("prototype feature versions cannot be mixed")
        grouped[str(payload["zone_id"])].append(_normalized(payload["feature"]))
    if len(grouped) < 2:
        raise ValueError("prototype calibration requires at least two zones")
    zone_order, centroids, radii, counts = [], [], [], {}
    all_within = []
    for zone_id, features in sorted(grouped.items()):
        if len(features) < 2:
            raise ValueError(f"Zone {zone_id!r} needs at least two samples")
        centroid = _normalized(np.mean(np.stack(features), axis=0))
        distances = [1.0 - float(np.clip(np.dot(item, centroid), -1.0, 1.0)) for item in features]
        zone_order.append(zone_id)
        centroids.append(centroid.tolist())
        radii.append(float(np.quantile(distances, 0.95)))
        counts[zone_id] = len(features)
        all_within.extend(distances)
    temperature = max(float(np.quantile(all_within, 0.95)) * 2.0, 0.02)
    return ZonePrototypeModel(
        source=source,
        feature_version=feature_version,
        zone_order=zone_order,
        centroids=centroids,
        within_zone_distance_q95=radii,
        temperature=temperature,
        samples_per_zone=counts,
    )


def feature_to_zone_evidence(
    feature: Iterable[float],
    model: ZonePrototypeModel,
    *,
    evidence_id: str,
    timestamp_s: float,
    observability: float,
    signal_quality: float,
    synchronization_uncertainty_s: float,
    source_track_id: str | None = None,
    provenance: dict[str, object] | None = None,
) -> ZoneEvidence:
    vector = _normalized(np.asarray(list(feature), dtype=float))
    centroids = np.asarray(model.centroids, dtype=float)
    if vector.size != centroids.shape[1]:
        raise ValueError("feature width does not match the zone prototype model")
    distances = 1.0 - np.clip(centroids @ vector, -1.0, 1.0)
    logits = -distances / model.temperature
    logits -= float(np.max(logits))
    probabilities = np.exp(logits)
    probabilities /= max(float(np.sum(probabilities)), 1e-12)
    return ZoneEvidence(
        evidence_id=evidence_id,
        timestamp_s=timestamp_s,
        source=model.source,
        source_track_id=source_track_id,
        feature_version=model.feature_version,
        zone_probabilities={
            zone_id: float(probabilities[index])
            for index, zone_id in enumerate(model.zone_order)
        },
        observability=observability,
        signal_quality=signal_quality,
        synchronization_uncertainty_s=synchronization_uncertainty_s,
        provenance=provenance or {},
    )


def bfi_feature_vector(feature: NormalizedBfiFeature) -> np.ndarray:
    """Create the versioned BFI vector used for zone-prototype calibration."""

    return np.asarray(
        feature.normalized_features
        + feature.temporal_delta
        + [feature.perturbation_energy, feature.motion_energy],
        dtype=float,
    )


def bfi_feature_to_zone_evidence(
    feature: NormalizedBfiFeature,
    model: ZonePrototypeModel,
    *,
    evidence_id: str,
    synchronization_uncertainty_s: float,
) -> ZoneEvidence:
    if model.source != "wifi_bfi":
        raise ValueError("BFI features require a wifi_bfi zone prototype model")
    if model.feature_version != "physioatlas.household-bfi-zone-feature.v1":
        raise ValueError("BFI zone prototype feature version is incompatible")
    return feature_to_zone_evidence(
        bfi_feature_vector(feature),
        model,
        evidence_id=evidence_id,
        timestamp_s=feature.timestamp_s,
        observability=feature.observability,
        signal_quality=feature.quality,
        synchronization_uncertainty_s=synchronization_uncertainty_s,
        source_track_id=feature.stream_id,
        provenance={
            "bfi_schema": feature.schema_version,
            "stream_id": feature.stream_id,
            "identity_claim": False,
            "source_sha256": feature.provenance.source_sha256,
            "calibration_sha256": feature.provenance.calibration_sha256,
        },
    )


def load_normalized_bfi_jsonl(path: str | Path) -> list[NormalizedBfiFeature]:
    result = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                result.append(NormalizedBfiFeature.model_validate_json(line))
            except Exception as exc:
                raise ValueError(
                    f"Invalid normalized BFI feature at line {line_number}: {exc}"
                ) from exc
    if not result:
        raise ValueError(f"No normalized BFI features found in {path}")
    return result


def build_bfi_zone_prototype_model(
    zone_feature_files: dict[str, str | Path],
    output_path: str | Path,
) -> ZonePrototypeModel:
    samples = []
    for zone_id, path in sorted(zone_feature_files.items()):
        for feature in load_normalized_bfi_jsonl(path):
            if feature.status != "observable":
                continue
            samples.append(
                {
                    "source": "wifi_bfi",
                    "feature_version": "physioatlas.household-bfi-zone-feature.v1",
                    "zone_id": zone_id,
                    "feature": bfi_feature_vector(feature).tolist(),
                }
            )
    model = fit_zone_prototype_model(
        samples,
        source="wifi_bfi",
        feature_version="physioatlas.household-bfi-zone-feature.v1",
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return model


def load_zone_prototype_model(path: str | Path) -> ZonePrototypeModel:
    return ZonePrototypeModel.model_validate_json(Path(path).read_text(encoding="utf-8"))


def csi_observation_to_zone_evidence(
    payload: dict[str, Any],
    zone_ids: Iterable[str],
    *,
    evidence_id: str,
) -> ZoneEvidence:
    ids = list(zone_ids)
    zone_id = str(payload["zone_id"])
    if zone_id not in ids:
        raise ValueError(f"CSI observation zone {zone_id!r} is not configured for fusion")
    confidence = dict(payload.get("confidence", {}))
    peak = float(np.clip(confidence.get("localization_fit", 0.0), 0.0, 1.0))
    remainder = (1.0 - peak) / max(len(ids) - 1, 1)
    probabilities = dict.fromkeys(ids, remainder)
    probabilities[zone_id] = peak
    provenance = dict(payload.get("provenance", {}))
    sync_uncertainty = provenance.get("clock_uncertainty_s")
    if sync_uncertainty is None:
        sync_uncertainty = 0.0 if provenance.get("synchronization_verified") else 1.0
    return ZoneEvidence(
        evidence_id=evidence_id,
        timestamp_s=float(payload["timestamp_s"]),
        source="wifi_csi",
        source_track_id=payload.get("track_id", zone_id),
        feature_version="physioatlas.household-csi-zone-evidence.v1",
        zone_probabilities=probabilities,
        observability=float(payload.get("observability", 0.0)),
        signal_quality=float(payload.get("signal_quality", 0.0)),
        synchronization_uncertainty_s=float(sync_uncertainty),
        respiratory_rate_bpm=payload.get("respiratory_rate_bpm"),
        heart_rate_bpm=payload.get("heart_rate_bpm"),
        motion_index=payload.get("motion_index"),
        provenance=provenance,
    )


def _source_union(evidence: list[ZoneEvidence], zones: list[str]) -> np.ndarray:
    per_target = np.asarray(
        [[item.zone_probabilities.get(zone, 0.0) for zone in zones] for item in evidence],
        dtype=float,
    )
    return 1.0 - np.prod(1.0 - per_target, axis=0)


def _weighted_log_odds(
    probabilities: dict[str, np.ndarray],
    source_quality: dict[str, float],
    config: CsiBfiFusionConfig,
) -> np.ndarray:
    numerator = np.zeros(len(config.zones), dtype=float)
    denominator = 0.0
    for source, values in probabilities.items():
        clipped = np.clip(values, 1e-4, 1.0 - 1e-4)
        weight = config.source_weights[source] * source_quality[source]
        numerator += weight * np.log(clipped / (1.0 - clipped))
        denominator += weight
    if denominator <= 0.0:
        return np.zeros(len(config.zones), dtype=float)
    logits = numerator / denominator
    return 1.0 / (1.0 + np.exp(-logits))


def fuse_zone_evidence(
    evidence: Iterable[ZoneEvidence],
    config: CsiBfiFusionConfig,
) -> FusionDecision:
    items = list(evidence)
    if not items:
        raise ValueError("at least one source evidence record is required")
    zone_ids = [item.zone_id for item in config.zones]
    unknown = set().union(*(set(item.zone_probabilities) for item in items)) - set(zone_ids)
    if unknown:
        raise ValueError(f"evidence contains unconfigured zones: {sorted(unknown)}")
    timestamps = np.asarray([item.timestamp_s for item in items], dtype=float)
    timestamp = float(np.median(timestamps))
    sources = sorted({item.source for item in items})
    evidence_ids = [item.evidence_id for item in items]
    if float(np.ptp(timestamps)) > config.maximum_time_skew_s:
        return _abstention(timestamp, "time_skew", sources, evidence_ids)
    if max(item.synchronization_uncertainty_s for item in items) > config.maximum_sync_uncertainty_s:
        return _abstention(timestamp, "synchronization_unverified", sources, evidence_ids)
    if config.require_both_sources and set(sources) != {"wifi_csi", "wifi_bfi"}:
        return _abstention(timestamp, "required_source_missing", sources, evidence_ids)
    by_source = {source: [item for item in items if item.source == source] for source in sources}
    source_probabilities = {
        source: _source_union(records, zone_ids) for source, records in by_source.items()
    }
    source_quality = {
        source: float(
            np.mean([item.signal_quality * item.observability for item in records])
        )
        for source, records in by_source.items()
    }
    observability = float(np.mean(list(source_quality.values())))
    if observability < config.minimum_observability:
        return _abstention(
            timestamp, "low_observability", sources, evidence_ids, observability=observability
        )
    conflict = None
    if set(source_probabilities) == {"wifi_csi", "wifi_bfi"}:
        conflict = float(
            np.mean(
                np.abs(
                    source_probabilities["wifi_csi"] - source_probabilities["wifi_bfi"]
                )
            )
        )
        if conflict > config.maximum_cross_modal_conflict:
            return _abstention(
                timestamp,
                "cross_modal_conflict",
                sources,
                evidence_ids,
                conflict=conflict,
                observability=observability,
            )
    fused = _weighted_log_odds(source_probabilities, source_quality, config)
    active = [zone_ids[index] for index in np.flatnonzero(fused >= config.occupancy_threshold)]
    if not active:
        return _abstention(
            timestamp,
            "no_zone_above_threshold",
            sources,
            evidence_ids,
            conflict=conflict,
            observability=observability,
            probabilities=dict(zip(zone_ids, fused.tolist())),
        )
    observations = [
        _household_observation(zone_id, fused[zone_ids.index(zone_id)], timestamp, items, config)
        for zone_id in active
    ]
    return FusionDecision(
        timestamp_s=timestamp,
        status="fused",
        reason="calibrated_evidence_agreement",
        active_zones=active,
        fused_probabilities=dict(zip(zone_ids, fused.tolist())),
        cross_modal_conflict=conflict,
        observability=observability,
        sources_present=sources,
        evidence_ids=evidence_ids,
        household_observations=observations,
    )


def _abstention(
    timestamp: float,
    reason: str,
    sources: list[str],
    evidence_ids: list[str],
    *,
    conflict: float | None = None,
    observability: float = 0.0,
    probabilities: dict[str, float] | None = None,
) -> FusionDecision:
    return FusionDecision(
        timestamp_s=timestamp,
        status="abstained",
        reason=reason,
        active_zones=[],
        fused_probabilities=probabilities or {},
        cross_modal_conflict=conflict,
        observability=observability,
        sources_present=sources,
        evidence_ids=evidence_ids,
    )


def _household_observation(
    zone_id: str,
    probability: float,
    timestamp: float,
    evidence: list[ZoneEvidence],
    config: CsiBfiFusionConfig,
) -> dict[str, object]:
    zone = next(item for item in config.zones if item.zone_id == zone_id)
    csi = [
        item
        for item in evidence
        if item.source == "wifi_csi"
        and item.zone_probabilities.get(zone_id, 0.0) >= config.occupancy_threshold
    ]
    def median_value(name: str) -> float | None:
        values = [getattr(item, name) for item in csi if getattr(item, name) is not None]
        return None if not values else float(np.median(values))
    return make_json_safe(
        {
            "schema_version": "physioatlas.household-observation.v2",
            "timestamp_s": timestamp,
            "position_m": zone.center_m,
            "position_covariance_m2": [
                zone.position_uncertainty_m**2,
                0.0,
                0.0,
                zone.position_uncertainty_m**2,
            ],
            "zone_id": zone_id,
            "embeddings": {},
            "signal_quality": probability,
            "observability": probability,
            "respiratory_rate_bpm": median_value("respiratory_rate_bpm"),
            "heart_rate_bpm": median_value("heart_rate_bpm"),
            "motion_index": median_value("motion_index"),
            "measurement_status": {
                "position": "calibrated_csi_bfi_fused_zone_estimate",
                "respiration": "csi_attributed" if csi else "not_observable",
                "heart_rate": "csi_attributed" if csi else "not_observable",
                "identity": "not_attempted",
            },
            "confidence": {"fused_zone_occupancy": probability},
            "provenance": {
                "source_kind": "calibrated_csi_bfi_evidence_fusion",
                "claim_class": "predicted",
                "research_only": True,
            },
        }
    )


def load_zone_evidence_jsonl(path: str | Path) -> list[ZoneEvidence]:
    result = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                result.append(ZoneEvidence.model_validate(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"Invalid zone evidence at line {line_number}: {exc}") from exc
    return result


def process_fusion_windows_jsonl(
    input_path: str | Path,
    config_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    config = CsiBfiFusionConfig.model_validate_json(
        Path(config_path).read_text(encoding="utf-8")
    )
    decisions = []
    with Path(input_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                evidence_payload = payload["evidence"]
                if not isinstance(evidence_payload, list):
                    raise ValueError("evidence must be a list")
                evidence = [ZoneEvidence.model_validate(item) for item in evidence_payload]
                decisions.append(fuse_zone_evidence(evidence, config))
            except Exception as exc:
                raise ValueError(
                    f"Invalid fusion window at line {line_number}: {exc}"
                ) from exc
    if not decisions:
        raise ValueError("No CSI/BFI fusion windows were provided")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for decision in decisions:
            handle.write(decision.model_dump_json() + "\n")
    return {
        "schema_version": "physioatlas.csi-bfi-fusion-report.v1",
        "windows": len(decisions),
        "fused": sum(item.status == "fused" for item in decisions),
        "abstained": sum(item.status == "abstained" for item in decisions),
        "abstention_reasons": {
            reason: sum(item.reason == reason for item in decisions)
            for reason in sorted({item.reason for item in decisions if item.status == "abstained"})
        },
        "output_path": str(destination.resolve()),
        "research_only": True,
    }
