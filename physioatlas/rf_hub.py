from __future__ import annotations

from collections.abc import Iterable

from .calibration_reflector_schema import (
    ReflectorCalibrationHealth,
    ReflectorTelemetrySample,
)
from .reflector_rf_atlas_schema import ReflectorDriftReport, ReflectorRfAtlas
from .research_hub import ResearchHubStore
from .rf_fusion_schema import FusionDecision


def publish_reflector_state(
    store: ResearchHubStore,
    *,
    telemetry: Iterable[ReflectorTelemetrySample],
    health: ReflectorCalibrationHealth,
    atlas: ReflectorRfAtlas | None = None,
    drift: ReflectorDriftReport | None = None,
    maximum_telemetry_samples: int = 500,
) -> dict:
    samples = list(telemetry)
    if not samples:
        raise ValueError("reflector hub state requires telemetry")
    if {item.plan_id for item in samples} != {health.plan_id}:
        raise ValueError("reflector telemetry and health plan IDs must match")
    latest = samples[-1]
    status = (
        "faulted"
        if health.faults_observed
        else "recalibration_required"
        if drift is not None and drift.recalibration_required
        else "valid"
        if health.valid_for_rf_calibration
        else "invalid"
    )
    section = {
        "schema_version": "physioatlas.reflector-hub-state.v1",
        "status": status,
        "safety_state": latest.safety_state.value,
        "synthetic": health.synthetic,
        "plan_id": health.plan_id,
        "clock_id": health.clock_id,
        "latest_sequence": latest.sequence,
        "telemetry": [
            item.model_dump(mode="json") for item in samples[-maximum_telemetry_samples:]
        ],
        "health": health.model_dump(mode="json"),
        "rf_atlas": None if atlas is None else atlas.model_dump(mode="json"),
        "drift": None if drift is None else drift.model_dump(mode="json"),
        "hardware_control_available": False,
        "research_only": True,
    }
    store.update(reflector=section)
    store.append_event(
        "reflector_state_published",
        {
            "status": status,
            "plan_id": health.plan_id,
            "synthetic": health.synthetic,
            "valid_for_rf_calibration": health.valid_for_rf_calibration,
        },
    )
    return section


def publish_fusion_decision(
    store: ResearchHubStore,
    decision: FusionDecision,
) -> dict:
    state = store.read()
    current = dict(state.get("rf_fusion", {}))
    reasons = dict(current.get("abstention_summary", {}))
    if decision.status == "abstained":
        reasons[decision.reason] = int(reasons.get(decision.reason, 0)) + 1
    section = {
        "schema_version": "physioatlas.csi-bfi-fusion-hub-state.v1",
        "status": decision.status,
        "last_decision": decision.model_dump(mode="json"),
        "abstention_summary": reasons,
        "sources_present": decision.sources_present,
        "cross_modal_conflict": decision.cross_modal_conflict,
        "observability": decision.observability,
        "research_only": True,
    }
    store.update(rf_fusion=section)
    store.append_event(
        "csi_bfi_fusion_decision",
        {
            "status": decision.status,
            "reason": decision.reason,
            "active_zones": decision.active_zones,
        },
    )
    return section
