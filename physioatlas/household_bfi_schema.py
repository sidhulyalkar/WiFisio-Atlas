from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

FrequencyBand = Literal["2.4ghz", "5ghz", "6ghz"]
BfiRepresentation = Literal["compressed_beamforming_matrix", "feature_vector"]


def _all_finite(values: list[float]) -> bool:
    return all(math.isfinite(value) for value in values)


class RawBfiFrame(BaseModel):
    """One decoded 802.11 beamforming-feedback observation."""

    model_config = ConfigDict(extra="forbid")

    timestamp_s: float = Field(ge=0.0)
    link_id: str = Field(min_length=1)
    client_id: str = Field(min_length=1)
    frequency_band: FrequencyBand
    channel: int = Field(ge=1)
    bandwidth_hz: float | None = Field(default=None, gt=0.0)
    packet_sequence: int | None = Field(default=None, ge=0)
    quality: float = Field(ge=0.0, le=1.0)
    compressed_beamforming_matrix: list[list[float]] | None = None
    feature_vector: list[float] | None = None

    @model_validator(mode="after")
    def validate_payload(self) -> RawBfiFrame:
        if not math.isfinite(self.timestamp_s):
            raise ValueError("timestamp_s must be finite")
        if self.bandwidth_hz is not None and not math.isfinite(self.bandwidth_hz):
            raise ValueError("bandwidth_hz must be finite")
        if not math.isfinite(self.quality):
            raise ValueError("quality must be finite")
        supplied = (
            self.compressed_beamforming_matrix is not None,
            self.feature_vector is not None,
        )
        if sum(supplied) != 1:
            raise ValueError(
                "exactly one of compressed_beamforming_matrix or feature_vector is required"
            )
        if self.feature_vector is not None:
            if len(self.feature_vector) < 4 or not _all_finite(self.feature_vector):
                raise ValueError("feature_vector must contain at least four finite values")
        if self.compressed_beamforming_matrix is not None:
            rows = self.compressed_beamforming_matrix
            if not rows or not rows[0]:
                raise ValueError("compressed_beamforming_matrix cannot be empty")
            width = len(rows[0])
            if width < 2 or len(rows) < 2 or any(len(row) != width for row in rows):
                raise ValueError(
                    "compressed_beamforming_matrix must be rectangular and at least 2x2"
                )
            if not all(_all_finite(row) for row in rows):
                raise ValueError("compressed_beamforming_matrix must be finite")
        return self

    @property
    def representation(self) -> BfiRepresentation:
        if self.feature_vector is not None:
            return "feature_vector"
        return "compressed_beamforming_matrix"

    def flattened_values(self) -> list[float]:
        if self.feature_vector is not None:
            return list(self.feature_vector)
        assert self.compressed_beamforming_matrix is not None
        return [value for row in self.compressed_beamforming_matrix for value in row]

    @property
    def matrix_shape(self) -> list[int] | None:
        if self.compressed_beamforming_matrix is None:
            return None
        return [
            len(self.compressed_beamforming_matrix),
            len(self.compressed_beamforming_matrix[0]),
        ]


class LinkBfiCalibration(BaseModel):
    """Robust empty-room statistics for one fixed link/client BFI stream."""

    model_config = ConfigDict(extra="forbid")

    stream_id: str = Field(min_length=1)
    link_id: str = Field(min_length=1)
    client_id: str = Field(min_length=1)
    frequency_band: FrequencyBand
    channel: int = Field(ge=1)
    bandwidth_hz: float | None = Field(default=None, gt=0.0)
    representation: BfiRepresentation
    matrix_shape: list[int] | None = None
    feature_width: int = Field(ge=4)
    feature_median: list[float]
    feature_mad: list[float]
    empty_energy_median: float = Field(ge=0.0)
    empty_energy_mad: float = Field(ge=0.0)
    frames: int = Field(ge=2)

    @model_validator(mode="after")
    def validate_vectors(self) -> LinkBfiCalibration:
        if len(self.feature_median) != self.feature_width:
            raise ValueError("feature_median length must equal feature_width")
        if len(self.feature_mad) != self.feature_width:
            raise ValueError("feature_mad length must equal feature_width")
        if not _all_finite(self.feature_median + self.feature_mad):
            raise ValueError("BFI calibration vectors must be finite")
        if any(value <= 0.0 for value in self.feature_mad):
            raise ValueError("feature_mad values must be positive")
        if self.representation == "compressed_beamforming_matrix":
            if (
                self.matrix_shape is None
                or len(self.matrix_shape) != 2
                or math.prod(self.matrix_shape) != self.feature_width
            ):
                raise ValueError("matrix_shape must match compressed matrix feature width")
        elif self.matrix_shape is not None:
            raise ValueError("matrix_shape is only valid for a compressed matrix")
        return self


class HouseholdBfiCalibration(BaseModel):
    """Versioned empty-room baseline; it carries no identity model."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["physioatlas.household-bfi-calibration.v1"] = (
        "physioatlas.household-bfi-calibration.v1"
    )
    created_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    environment_id: str = Field(min_length=1)
    stream_order: list[str] = Field(min_length=1)
    streams: list[LinkBfiCalibration] = Field(min_length=1)
    research_only: Literal[True] = True
    identity_claim: Literal[False] = False
    claim_boundary: str = (
        "Normalized beamforming-feedback perturbations only; no person identity, "
        "position, physiology, or clinical claim."
    )

    @model_validator(mode="after")
    def validate_contract(self) -> HouseholdBfiCalibration:
        stream_ids = [stream.stream_id for stream in self.streams]
        if len(stream_ids) != len(set(stream_ids)):
            raise ValueError("BFI calibration stream IDs must be unique")
        if stream_ids != self.stream_order:
            raise ValueError("stream_order must match the ordered stream list")
        return self


class BfiFeatureProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_capture: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    packet_sequence: int | None = Field(default=None, ge=0)
    feature_method: Literal["robust-empty-room-zscore-delta-v1"] = (
        "robust-empty-room-zscore-delta-v1"
    )


class NormalizedBfiFeature(BaseModel):
    """A fusion-ready BFI feature that remains explicitly non-identifying."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["physioatlas.household-bfi-feature.v1"] = (
        "physioatlas.household-bfi-feature.v1"
    )
    timestamp_s: float = Field(ge=0.0)
    stream_id: str
    link_id: str
    client_id: str
    frequency_band: FrequencyBand
    channel: int = Field(ge=1)
    normalized_features: list[float] = Field(min_length=4)
    temporal_delta: list[float] = Field(min_length=4)
    perturbation_energy: float = Field(ge=0.0)
    motion_energy: float = Field(ge=0.0)
    quality: float = Field(ge=0.0, le=1.0)
    observability: float = Field(ge=0.0, le=1.0)
    status: Literal["observable", "not_observable"]
    provenance: BfiFeatureProvenance
    research_only: Literal[True] = True
    identity_claim: Literal[False] = False

    @model_validator(mode="after")
    def validate_feature(self) -> NormalizedBfiFeature:
        if len(self.normalized_features) != len(self.temporal_delta):
            raise ValueError("normalized_features and temporal_delta widths must match")
        numeric = self.normalized_features + self.temporal_delta
        numeric += [
            self.timestamp_s,
            self.perturbation_energy,
            self.motion_energy,
            self.quality,
            self.observability,
        ]
        if not _all_finite(numeric):
            raise ValueError("normalized BFI feature values must be finite")
        expected = "observable" if self.observability >= 0.5 else "not_observable"
        if self.status != expected:
            raise ValueError("status must agree with the observability threshold")
        return self


class BfiProcessingReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["physioatlas.household-bfi-report.v1"] = (
        "physioatlas.household-bfi-report.v1"
    )
    input_path: str
    calibration_path: str
    output_path: str
    accepted_frames: int = Field(ge=0)
    observations_emitted: int = Field(ge=0)
    streams_observed: list[str]
    not_observable_frames: int = Field(ge=0)
    research_only: Literal[True] = True
    identity_claim: Literal[False] = False
