from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ReflectorRfSample(BaseModel):
    """RF response already transformed into the reflector telemetry clock."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    timestamp_reflector_clock_s: float = Field(ge=0.0)
    alignment_receipt_id: str = Field(min_length=1)
    source: Literal["wifi_csi", "wifi_bfi"]
    link_id: str = Field(min_length=1)
    frequency_band: Literal["2.4ghz", "5ghz", "6ghz"]
    response_energy: float = Field(ge=0.0)
    response_features: list[float] = Field(min_length=2)
    quality: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def finite_features(self) -> ReflectorRfSample:
        if not np.all(np.isfinite(self.response_features)):
            raise ValueError("reflector RF response features must be finite")
        return self


class ReflectorWaypointResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    position: float
    source: Literal["wifi_csi", "wifi_bfi"]
    link_id: str
    frequency_band: Literal["2.4ghz", "5ghz", "6ghz"]
    mean_energy: float = Field(ge=0.0)
    energy_std: float = Field(ge=0.0)
    mean_features: list[float]
    samples: int = Field(ge=3)
    mean_quality: float = Field(ge=0.0, le=1.0)


class ReflectorLinkSensitivity(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    source: Literal["wifi_csi", "wifi_bfi"]
    link_id: str
    frequency_band: Literal["2.4ghz", "5ghz", "6ghz"]
    dynamic_range: float = Field(ge=0.0)
    repeatability_noise: float = Field(ge=0.0)
    sensitivity_ratio: float = Field(ge=0.0)
    low_response_positions: list[float]


class ReflectorRfAtlas(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: str = "physioatlas.reflector-rf-atlas.v1"
    atlas_id: str = Field(min_length=1)
    created_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    environment_id: str = Field(min_length=1)
    plan_id: str = Field(min_length=1)
    alignment_receipt_id: str = Field(min_length=1)
    waypoint_responses: list[ReflectorWaypointResponse] = Field(min_length=1)
    link_sensitivity: list[ReflectorLinkSensitivity] = Field(min_length=1)
    cross_band_energy_correlation: float | None = Field(
        default=None, ge=-1.0, le=1.0
    )
    csi_bfi_energy_correlation: float | None = Field(
        default=None, ge=-1.0, le=1.0
    )
    accepted_rf_samples: int = Field(ge=1)
    rejected_rf_samples: int = Field(ge=0)
    motor_off_dwell_verified: bool
    synthetic: bool
    research_only: bool = True
    claim_boundary: str = (
        "Passive-reflector RF transfer response; not evidence of person sensing accuracy."
    )

    @model_validator(mode="after")
    def motor_off_required(self) -> ReflectorRfAtlas:
        if not self.motor_off_dwell_verified:
            raise ValueError("reflector RF atlas requires verified motor-off dwells")
        return self


class ReflectorDriftReport(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: str = "physioatlas.reflector-drift.v1"
    baseline_atlas_id: str
    candidate_atlas_id: str
    matched_responses: int = Field(ge=0)
    normalized_energy_rmse: float | None = Field(default=None, ge=0.0)
    feature_cosine_distance: float | None = Field(default=None, ge=0.0, le=2.0)
    drift_score: float | None = Field(default=None, ge=0.0)
    recalibration_required: bool
    reasons: list[str]
    research_only: bool = True
    claim_boundary: str = (
        "Reflector-transfer drift only; task degradation must be independently tested."
    )
