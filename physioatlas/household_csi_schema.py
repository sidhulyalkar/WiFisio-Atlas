from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class LinkCsiCalibration(BaseModel):
    """Empty-room statistics for one fixed transmitter/receiver link."""

    model_config = ConfigDict(extra="forbid")

    link_id: str = Field(min_length=1)
    n_subcarriers: int = Field(gt=3)
    frequency_band: str = "unknown"
    center_frequency_hz: Optional[float] = Field(default=None, gt=0.0)
    channel: Optional[int] = Field(default=None, ge=1)
    bandwidth_hz: Optional[float] = Field(default=None, gt=0.0)
    clock_offset_s: float = 0.0
    clock_offset_mad_s: Optional[float] = Field(default=None, ge=0.0)
    synchronization_pairs: int = Field(default=0, ge=0)
    amplitude_median: list[float]
    amplitude_mad: list[float]
    phase_mean: Optional[list[float]] = None
    phase_available: bool = False
    empty_energy_median: float = Field(ge=0.0)
    empty_energy_mad: float = Field(ge=0.0)
    frames: int = Field(gt=1)

    @model_validator(mode="after")
    def validate_vectors(self) -> "LinkCsiCalibration":
        if self.frequency_band not in {"2.4ghz", "5ghz", "6ghz", "unknown"}:
            raise ValueError("Unsupported CSI frequency band")
        if len(self.amplitude_median) != self.n_subcarriers:
            raise ValueError("amplitude_median length must equal n_subcarriers")
        if len(self.amplitude_mad) != self.n_subcarriers:
            raise ValueError("amplitude_mad length must equal n_subcarriers")
        if self.phase_mean is not None and len(self.phase_mean) != self.n_subcarriers:
            raise ValueError("phase_mean length must equal n_subcarriers")
        vectors = [self.amplitude_median, self.amplitude_mad]
        if self.phase_mean is not None:
            vectors.append(self.phase_mean)
        if not all(np.all(np.isfinite(item)) for item in vectors):
            raise ValueError("CSI calibration vectors must be finite")
        return self


class ZoneCsiCalibration(BaseModel):
    """Single-person spatial response measured at one room calibration zone."""

    model_config = ConfigDict(extra="forbid")

    zone_id: str = Field(min_length=1)
    center_m: list[float] = Field(min_length=2, max_length=3)
    link_signature: list[float]
    samples: int = Field(gt=1)
    source_capture: str

    @model_validator(mode="after")
    def validate_values(self) -> "ZoneCsiCalibration":
        if not np.all(np.isfinite(self.center_m)):
            raise ValueError("zone center must be finite")
        if not np.all(np.isfinite(self.link_signature)):
            raise ValueError("zone link signature must be finite")
        if max(self.link_signature, default=0.0) <= 0.0:
            raise ValueError("zone link signature must contain a positive response")
        return self


class HouseholdCsiCalibration(BaseModel):
    """Versioned room calibration for the auditable CSI separation baseline."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "physioatlas.household-csi-calibration.v1"
    created_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    environment_id: str = Field(min_length=1)
    sample_rate_hz: float = Field(gt=1.0)
    link_order: list[str] = Field(min_length=3)
    links: list[LinkCsiCalibration] = Field(min_length=3)
    zones: list[ZoneCsiCalibration] = Field(min_length=1)
    minimum_links: int = Field(default=3, ge=3)
    occupancy_activation_threshold: float = Field(default=0.30, gt=0.0)
    maximum_occupants: int = Field(default=5, ge=1)
    position_uncertainty_m: float = Field(default=0.75, gt=0.0)
    synchronization_verified: bool = False
    zone_signature_condition_number: Optional[float] = Field(default=None, ge=1.0)
    calibration_notes: list[str] = Field(default_factory=list)
    research_only: bool = True
    claim_boundary: str = (
        "Calibrated zone-level RF attribution; not direct imaging or clinical measurement."
    )

    @model_validator(mode="after")
    def validate_contract(self) -> "HouseholdCsiCalibration":
        link_ids = [item.link_id for item in self.links]
        if len(link_ids) != len(set(link_ids)):
            raise ValueError("CSI calibration link IDs must be unique")
        if self.link_order != link_ids:
            raise ValueError("link_order must match the ordered link calibration list")
        if self.minimum_links > len(link_ids):
            raise ValueError("minimum_links cannot exceed the calibrated link count")
        zone_ids = [item.zone_id for item in self.zones]
        if len(zone_ids) != len(set(zone_ids)):
            raise ValueError("CSI calibration zone IDs must be unique")
        for zone in self.zones:
            if len(zone.link_signature) != len(link_ids):
                raise ValueError(
                    f"Zone {zone.zone_id!r} signature must match calibrated links"
                )
        return self


class CsiProcessingConfig(BaseModel):
    """Conservative inference thresholds for mixed household CSI."""

    model_config = ConfigDict(extra="forbid")

    localization_window_s: float = Field(default=2.0, gt=0.25)
    vital_window_s: float = Field(default=30.0, ge=8.0)
    stride_s: float = Field(default=1.0, gt=0.0)
    maximum_gap_s: float = Field(default=0.25, gt=0.0)
    ridge: float = Field(default=0.05, gt=0.0)
    minimum_observability: float = Field(default=0.55, ge=0.0, le=1.0)
    maximum_separation_condition: float = Field(default=12.0, gt=1.0)
    enable_heart_rate: bool = False
    allow_multi_person_heart_rate: bool = False
    require_frequency_bands: list[str] = Field(default_factory=list)


class CsiProcessingReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "physioatlas.household-csi-report.v1"
    input_path: str
    calibration_path: str
    output_path: str
    accepted_frames: int = Field(ge=0)
    rejected_frames: int = Field(ge=0)
    links_observed: list[str]
    frequency_bands_observed: list[str]
    synchronization_verified: bool
    windows_evaluated: int = Field(ge=0)
    observations_emitted: int = Field(ge=0)
    not_observable_windows: int = Field(ge=0)
    research_only: bool = True
