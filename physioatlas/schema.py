from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Modality(str, Enum):
    WIFI_CSI = "wifi_csi"
    MMWAVE = "mmwave"
    UWB = "uwb"
    ECG = "ecg"
    PPG = "ppg"
    RESP_BELT = "resp_belt"
    POSE = "pose"
    ULTRASOUND = "ultrasound"
    BLOOD_PRESSURE = "blood_pressure"
    SYNC = "sync"
    ENVIRONMENT = "environment"


class RFRepresentation(str, Enum):
    """How RF values are arranged in a stream.

    ``amplitude_phase`` uses ``[amplitude..., phase...]`` per row. ``iq`` uses
    ``[I..., Q...]``. ``features`` is a generic real-valued representation.
    The representation is declared because a neural model cannot infer it
    safely from an even feature count alone.
    """

    FEATURES = "features"
    AMPLITUDE_PHASE = "amplitude_phase"
    IQ = "iq"
    RANGE_DOPPLER = "range_doppler"


class AnatomyRegion(str, Enum):
    WHOLE_BODY = "whole_body"
    THORAX = "thorax"
    LEFT_THORAX = "left_thorax"
    RIGHT_THORAX = "right_thorax"
    CARDIAC_APEX = "cardiac_apex"
    CAROTID_LEFT = "carotid_left"
    CAROTID_RIGHT = "carotid_right"
    RADIAL_LEFT = "radial_left"
    RADIAL_RIGHT = "radial_right"
    ABDOMEN = "abdomen"
    DIAPHRAGM = "diaphragm"
    BLADDER = "bladder"
    CUSTOM = "custom"


class Vec3(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: float
    y: float
    z: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


class Sensor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sensor_id: str = Field(min_length=1)
    modality: Modality
    position_m: Optional[Vec3] = None
    orientation_xyz: Optional[Vec3] = None
    sample_rate_hz: float = Field(gt=0)
    clock_domain: str = "host"
    hardware: Optional[str] = None
    serial_number: Optional[str] = None
    latency_s: float = Field(default=0.0, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RFLink(BaseModel):
    model_config = ConfigDict(extra="forbid")

    link_id: str = Field(min_length=1)
    tx_sensor_id: str
    rx_sensor_id: str
    center_frequency_hz: float = Field(gt=0)
    bandwidth_hz: float = Field(gt=0)
    n_subcarriers: int = Field(gt=0)
    antenna_pair: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SensorGeometry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    coordinate_frame: str = "room_meters"
    sensors: list[Sensor]
    rf_links: list[RFLink] = Field(default_factory=list)
    anatomy_anchors_m: dict[AnatomyRegion, Vec3] = Field(default_factory=dict)
    calibration_uncertainty_m: Optional[float] = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_links(self) -> "SensorGeometry":
        ids = {sensor.sensor_id for sensor in self.sensors}
        if len(ids) != len(self.sensors):
            raise ValueError("sensor_id values must be unique")
        link_ids = {link.link_id for link in self.rf_links}
        if len(link_ids) != len(self.rf_links):
            raise ValueError("link_id values must be unique")
        for link in self.rf_links:
            if link.tx_sensor_id not in ids or link.rx_sensor_id not in ids:
                raise ValueError(f"RF link {link.link_id!r} references an unknown sensor")
            if link.tx_sensor_id == link.rx_sensor_id:
                raise ValueError(f"RF link {link.link_id!r} cannot connect a sensor to itself")
        return self


class TargetDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    source_modality: Modality
    anatomy_region: AnatomyRegion
    task: Literal["waveform", "scalar", "classification"] = "waveform"
    units: str = "arbitrary"
    supervision_required: bool = True
    clinical_claim_allowed: bool = False
    graph_node_id: Optional[str] = None
    description: str = ""


class StreamReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    modality: Modality
    sensor_id: str
    path: str
    timestamp_key: str = "timestamps_s"
    value_key: str = "values"
    quality_key: Optional[str] = "quality"
    representation: RFRepresentation = RFRepresentation.FEATURES
    clock_domain: Optional[str] = None
    units: str = "arbitrary"
    feature_names: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "physioatlas.session.v2"
    session_id: str
    subject_id: str
    started_at_utc: str
    geometry_path: str
    streams: list[StreamReference]
    targets: list[TargetDefinition]
    protocol: str
    consent_reference: Optional[str] = None
    consent_path: Optional[str] = None
    physiology_graph_path: Optional[str] = None
    calibration_path: Optional[str] = None
    privacy_policy_path: Optional[str] = None
    environment_id: str = "unknown"
    posture: Optional[str] = None
    condition: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_stream_sensor_ids(self) -> "SessionManifest":
        keys = [(stream.modality.value, stream.sensor_id) for stream in self.streams]
        if len(keys) != len(set(keys)):
            raise ValueError("stream modality/sensor_id pairs must be unique")
        return self

    def resolve(self, manifest_path: Union[str, Path]) -> "SessionManifest":
        """Return a copy whose referenced paths are absolute."""
        root = Path(manifest_path).resolve().parent
        streams = [
            stream.model_copy(update={"path": str((root / stream.path).resolve())})
            for stream in self.streams
        ]
        updates: dict[str, Any] = {
            "geometry_path": str((root / self.geometry_path).resolve()),
            "streams": streams,
        }
        for field in (
            "consent_path",
            "physiology_graph_path",
            "calibration_path",
            "privacy_policy_path",
        ):
            value = getattr(self, field)
            if value:
                updates[field] = str((root / value).resolve())
        return self.model_copy(update=updates)
