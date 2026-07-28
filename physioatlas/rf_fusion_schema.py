from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

RfEvidenceSource = Literal["wifi_csi", "wifi_bfi"]


class FusionZone(BaseModel):
    model_config = ConfigDict(extra="forbid")

    zone_id: str = Field(min_length=1)
    center_m: list[float] = Field(min_length=2, max_length=3)
    position_uncertainty_m: float = Field(default=0.75, gt=0.0)

    @model_validator(mode="after")
    def finite_center(self) -> FusionZone:
        if not np.all(np.isfinite(self.center_m)):
            raise ValueError("zone center must contain finite coordinates")
        return self


class ZonePrototypeModel(BaseModel):
    """Single-target feature-to-zone calibration for one evidence source."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "physioatlas.rf-zone-prototypes.v1"
    source: RfEvidenceSource
    feature_version: str = Field(min_length=1)
    zone_order: list[str] = Field(min_length=2)
    centroids: list[list[float]] = Field(min_length=2)
    within_zone_distance_q95: list[float] = Field(min_length=2)
    temperature: float = Field(gt=0.0)
    samples_per_zone: dict[str, int]
    research_only: bool = True

    @model_validator(mode="after")
    def compatible_dimensions(self) -> ZonePrototypeModel:
        if len(set(self.zone_order)) != len(self.zone_order):
            raise ValueError("zone_order must be unique")
        if len(self.centroids) != len(self.zone_order):
            raise ValueError("one centroid is required per zone")
        if len(self.within_zone_distance_q95) != len(self.zone_order):
            raise ValueError("one distance radius is required per zone")
        widths = {len(item) for item in self.centroids}
        if len(widths) != 1 or next(iter(widths), 0) < 2:
            raise ValueError("prototype centroids need one stable feature width")
        if not np.all(np.isfinite(self.centroids)):
            raise ValueError("prototype centroids must be finite")
        if set(self.samples_per_zone) != set(self.zone_order):
            raise ValueError("sample counts must match zone_order")
        if min(self.samples_per_zone.values(), default=0) < 2:
            raise ValueError("at least two samples are required per zone")
        return self


class ZoneEvidence(BaseModel):
    """Calibrated, source-specific evidence before cross-modal fusion."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "physioatlas.rf-zone-evidence.v1"
    evidence_id: str = Field(min_length=1)
    timestamp_s: float
    source: RfEvidenceSource
    source_track_id: str | None = None
    feature_version: str = Field(min_length=1)
    zone_probabilities: dict[str, float]
    observability: float = Field(ge=0.0, le=1.0)
    signal_quality: float = Field(ge=0.0, le=1.0)
    synchronization_uncertainty_s: float = Field(ge=0.0)
    respiratory_rate_bpm: float | None = Field(default=None, gt=0.0)
    heart_rate_bpm: float | None = Field(default=None, gt=0.0)
    motion_index: float | None = Field(default=None, ge=0.0)
    provenance: dict[str, object] = Field(default_factory=dict)
    research_only: bool = True

    @model_validator(mode="after")
    def probabilities_are_valid(self) -> ZoneEvidence:
        if not np.isfinite(self.timestamp_s):
            raise ValueError("timestamp must be finite")
        if not self.zone_probabilities:
            raise ValueError("zone probabilities cannot be empty")
        values = np.asarray(list(self.zone_probabilities.values()), dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("zone probabilities must be finite values in [0, 1]")
        return self


class CsiBfiFusionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "physioatlas.csi-bfi-fusion-config.v1"
    zones: list[FusionZone] = Field(min_length=1)
    source_weights: dict[RfEvidenceSource, float] = Field(
        default_factory=lambda: {"wifi_csi": 1.0, "wifi_bfi": 1.0}
    )
    maximum_time_skew_s: float = Field(default=0.10, gt=0.0)
    maximum_sync_uncertainty_s: float = Field(default=0.025, ge=0.0)
    occupancy_threshold: float = Field(default=0.65, gt=0.5, lt=1.0)
    maximum_cross_modal_conflict: float = Field(default=0.35, ge=0.0, le=1.0)
    minimum_observability: float = Field(default=0.55, ge=0.0, le=1.0)
    require_both_sources: bool = True

    @model_validator(mode="after")
    def unique_zones(self) -> CsiBfiFusionConfig:
        ids = [item.zone_id for item in self.zones]
        if len(ids) != len(set(ids)):
            raise ValueError("fusion zone IDs must be unique")
        if set(self.source_weights) != {"wifi_csi", "wifi_bfi"}:
            raise ValueError("source weights must declare wifi_csi and wifi_bfi")
        if min(self.source_weights.values()) <= 0.0:
            raise ValueError("source weights must be positive")
        return self


class FusionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "physioatlas.csi-bfi-fusion-decision.v1"
    timestamp_s: float
    status: Literal["fused", "abstained"]
    reason: str
    active_zones: list[str]
    fused_probabilities: dict[str, float]
    cross_modal_conflict: float | None = Field(default=None, ge=0.0, le=1.0)
    observability: float = Field(ge=0.0, le=1.0)
    sources_present: list[RfEvidenceSource]
    evidence_ids: list[str]
    household_observations: list[dict[str, object]] = Field(default_factory=list)
    research_only: bool = True


class FusionTruthWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_id: str = Field(min_length=1)
    timestamp_s: float
    active_zones: list[str]
    reference_kind: Literal["camera", "depth_camera", "uwb", "manual_annotation", "synthetic"]
    reference_id: str = Field(min_length=1)
    consent_verified: bool

    @model_validator(mode="after")
    def validate_truth(self) -> FusionTruthWindow:
        if not np.isfinite(self.timestamp_s):
            raise ValueError("truth timestamp must be finite")
        if len(self.active_zones) != len(set(self.active_zones)):
            raise ValueError("truth active zones must be unique")
        if self.reference_kind != "synthetic" and not self.consent_verified:
            raise ValueError("physical reference truth requires verified consent")
        return self


class FusionEvaluationMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    windows: int = Field(ge=0)
    coverage: float = Field(ge=0.0, le=1.0)
    abstention_rate: float = Field(ge=0.0, le=1.0)
    zone_precision: float = Field(ge=0.0, le=1.0)
    zone_recall: float = Field(ge=0.0, le=1.0)
    zone_f1: float = Field(ge=0.0, le=1.0)
    count_mae: float = Field(ge=0.0)
    exact_set_accuracy: float = Field(ge=0.0, le=1.0)
    selective_exact_accuracy: float = Field(ge=0.0, le=1.0)


class FusionAblationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "physioatlas.csi-bfi-ablation.v1"
    modes: dict[str, FusionEvaluationMetrics]
    reference_ids: list[str]
    research_only: bool = True
    claim_boundary: str = (
        "Synthetic or reference-linked zone evaluation; not a general household accuracy claim."
    )
