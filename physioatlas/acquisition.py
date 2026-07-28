from __future__ import annotations

import json
import socket
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .adapters import ruview_jsonl_to_npz
from .calibration import ClockCalibration, apply_clock_calibration
from .io import SignalStream, dump_json, load_stream, save_stream
from .schema import (
    Modality,
    RFRepresentation,
    SensorGeometry,
    SessionManifest,
    StreamReference,
    TargetDefinition,
)


@dataclass(frozen=True)
class AdapterCapabilities:
    adapter_type: str
    modalities: tuple[Modality, ...]
    live: bool
    replay: bool
    supplies_quality: bool
    supplies_hardware_timestamps: bool
    description: str


class AdapterSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    adapter_type: str
    sensor_id: str
    modality: Modality
    output_name: str
    sample_rate_hz: float = Field(gt=0)
    clock_domain: str = "host"
    representation: RFRepresentation = RFRepresentation.FEATURES
    parameters: dict[str, Any] = Field(default_factory=dict)


class AcquisitionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    subject_id: str
    protocol: str
    output_dir: str
    geometry_path: str
    duration_seconds: float = Field(gt=0)
    environment_id: str = "unknown"
    posture: Optional[str] = None
    condition: Optional[str] = None
    consent_path: Optional[str] = None
    adapters: list[AdapterSpec]
    targets: list[TargetDefinition] = Field(default_factory=list)


class SensorAdapter(ABC):
    capabilities: AdapterCapabilities

    def __init__(self, spec: AdapterSpec):
        self.spec = spec

    @abstractmethod
    def acquire(self, duration_seconds: float) -> SignalStream:
        raise NotImplementedError


class AdapterRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, Callable[[AdapterSpec], SensorAdapter]] = {}
        self._capabilities: dict[str, AdapterCapabilities] = {}

    def register(
        self,
        name: str,
        factory: Callable[[AdapterSpec], SensorAdapter],
        capabilities: AdapterCapabilities,
    ) -> None:
        if name in self._factories:
            raise ValueError(f"Adapter {name!r} is already registered")
        self._factories[name] = factory
        self._capabilities[name] = capabilities

    def create(self, spec: AdapterSpec) -> SensorAdapter:
        try:
            factory = self._factories[spec.adapter_type]
        except KeyError as exc:
            raise ValueError(f"Unknown adapter type {spec.adapter_type!r}") from exc
        capabilities = self._capabilities[spec.adapter_type]
        if spec.modality not in capabilities.modalities:
            raise ValueError(
                f"Adapter {spec.adapter_type!r} does not support modality {spec.modality.value!r}"
            )
        return factory(spec)

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "adapter_type": name,
                "modalities": [item.value for item in cap.modalities],
                "live": cap.live,
                "replay": cap.replay,
                "supplies_quality": cap.supplies_quality,
                "supplies_hardware_timestamps": cap.supplies_hardware_timestamps,
                "description": cap.description,
            }
            for name, cap in sorted(self._capabilities.items())
        ]


class NPZReplayAdapter(SensorAdapter):
    capabilities = AdapterCapabilities(
        adapter_type="npz_replay",
        modalities=tuple(Modality),
        live=False,
        replay=True,
        supplies_quality=True,
        supplies_hardware_timestamps=True,
        description="Replay a canonical PhysioAtlas NPZ stream.",
    )

    def acquire(self, duration_seconds: float) -> SignalStream:
        path = self.spec.parameters.get("path")
        if not path:
            raise ValueError("npz_replay requires parameters.path")
        reference = StreamReference(
            modality=self.spec.modality,
            sensor_id=self.spec.sensor_id,
            path=str(path),
            representation=self.spec.representation,
        )
        stream = load_stream(reference)
        limit = stream.timestamps_s[0] + duration_seconds
        keep = stream.timestamps_s <= limit
        if keep.sum() < 2:
            raise ValueError("Replay duration contains fewer than two samples")
        timestamps = stream.timestamps_s[keep] - stream.timestamps_s[keep][0]
        return SignalStream(
            timestamps_s=timestamps,
            values=stream.values[keep],
            quality=None if stream.quality is None else stream.quality[keep],
        )


class RuViewJSONLReplayAdapter(SensorAdapter):
    capabilities = AdapterCapabilities(
        adapter_type="ruview_jsonl",
        modalities=(Modality.WIFI_CSI,),
        live=False,
        replay=True,
        supplies_quality=True,
        supplies_hardware_timestamps=True,
        description="Replay RuView CSI JSONL/RVCSI through the canonical converter.",
    )

    def acquire(self, duration_seconds: float) -> SignalStream:
        source = self.spec.parameters.get("path")
        if not source:
            raise ValueError("ruview_jsonl requires parameters.path")
        temporary = Path(source).with_suffix(Path(source).suffix + ".physioatlas.tmp.npz")
        try:
            ruview_jsonl_to_npz(source, temporary)
            reference = StreamReference(
                modality=Modality.WIFI_CSI,
                sensor_id=self.spec.sensor_id,
                path=str(temporary),
                representation=RFRepresentation.AMPLITUDE_PHASE,
            )
            stream = load_stream(reference)
        finally:
            temporary.unlink(missing_ok=True)
        limit = stream.timestamps_s[0] + duration_seconds
        keep = stream.timestamps_s <= limit
        return SignalStream(
            timestamps_s=stream.timestamps_s[keep] - stream.timestamps_s[keep][0],
            values=stream.values[keep],
            quality=None if stream.quality is None else stream.quality[keep],
        )


class SyntheticWaveAdapter(SensorAdapter):
    capabilities = AdapterCapabilities(
        adapter_type="synthetic_wave",
        modalities=tuple(Modality),
        live=False,
        replay=False,
        supplies_quality=True,
        supplies_hardware_timestamps=True,
        description="Deterministic synthetic adapter for acquisition integration tests.",
    )

    def acquire(self, duration_seconds: float) -> SignalStream:
        seed = int(self.spec.parameters.get("seed", 0))
        frequency_hz = float(self.spec.parameters.get("frequency_hz", 1.0))
        features = int(self.spec.parameters.get("features", 1))
        noise = float(self.spec.parameters.get("noise", 0.01))
        offset_s = float(self.spec.parameters.get("clock_offset_s", 0.0))
        drift_ppm = float(self.spec.parameters.get("clock_drift_ppm", 0.0))
        rng = np.random.default_rng(seed)
        reference_t = np.arange(0.0, duration_seconds, 1.0 / self.spec.sample_rate_hz)
        sensor_t = (reference_t - offset_s) / (1.0 + drift_ppm * 1e-6)
        phases = np.linspace(0.0, np.pi / 3, features, endpoint=False)
        values = np.stack(
            [
                np.sin(2 * np.pi * frequency_hz * reference_t + phase)
                + rng.normal(0.0, noise, reference_t.size)
                for phase in phases
            ],
            axis=1,
        ).astype(np.float32)
        return SignalStream(
            timestamps_s=sensor_t,
            values=values,
            quality=np.ones(reference_t.size, dtype=np.float32),
        )


class UDPJSONAdapter(SensorAdapter):
    capabilities = AdapterCapabilities(
        adapter_type="udp_json",
        modalities=tuple(Modality),
        live=True,
        replay=False,
        supplies_quality=True,
        supplies_hardware_timestamps=True,
        description="Receive timestamped JSON sensor frames over UDP.",
    )

    def acquire(self, duration_seconds: float) -> SignalStream:
        host = str(self.spec.parameters.get("host", "0.0.0.0"))
        port = int(self.spec.parameters.get("port", 5005))
        receive_buffer = int(self.spec.parameters.get("receive_buffer", 65535))
        timestamps: list[float] = []
        rows: list[np.ndarray] = []
        quality: list[float] = []
        expected_width: Optional[int] = None
        started = time.monotonic()
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind((host, port))
            sock.settimeout(min(0.25, max(0.01, duration_seconds / 10.0)))
            while time.monotonic() - started < duration_seconds:
                try:
                    packet, _ = sock.recvfrom(receive_buffer)
                except socket.timeout:
                    continue
                payload = json.loads(packet.decode("utf-8"))
                timestamp = payload.get("timestamp_s")
                if timestamp is None and payload.get("timestamp_ns") is not None:
                    timestamp = float(payload["timestamp_ns"]) / 1e9
                if timestamp is None:
                    timestamp = time.monotonic() - started
                values = payload.get("values")
                if values is None:
                    amplitude = payload.get("amplitude", payload.get("amplitudes"))
                    phase = payload.get("phase")
                    if amplitude is None:
                        continue
                    amplitude_array = np.asarray(amplitude, dtype=np.float32).reshape(-1)
                    phase_array = (
                        np.asarray(phase, dtype=np.float32).reshape(-1)
                        if phase is not None
                        else np.zeros_like(amplitude_array)
                    )
                    row = np.concatenate([amplitude_array, phase_array])
                else:
                    row = np.asarray(values, dtype=np.float32).reshape(-1)
                if not np.all(np.isfinite(row)):
                    continue
                if expected_width is None:
                    expected_width = int(row.size)
                if row.size != expected_width:
                    raise ValueError("UDP sensor feature width changed during acquisition")
                timestamps.append(float(timestamp))
                rows.append(row)
                quality.append(float(payload.get("quality", payload.get("quality_score", 1.0))))
        if len(rows) < 2:
            raise ValueError("UDP acquisition received fewer than two valid frames")
        order = np.argsort(np.asarray(timestamps), kind="stable")
        timestamp_array = np.asarray(timestamps, dtype=np.float64)[order]
        timestamp_array -= timestamp_array[0]
        unique = np.concatenate([[True], np.diff(timestamp_array) > 0])
        return SignalStream(
            timestamps_s=timestamp_array[unique],
            values=np.stack(rows)[order][unique],
            quality=np.clip(np.asarray(quality, dtype=np.float32)[order][unique], 0.0, 1.0),
        )


DEFAULT_REGISTRY = AdapterRegistry()
for _adapter in (NPZReplayAdapter, RuViewJSONLReplayAdapter, SyntheticWaveAdapter, UDPJSONAdapter):
    DEFAULT_REGISTRY.register(
        _adapter.capabilities.adapter_type,
        _adapter,
        _adapter.capabilities,
    )


def run_acquisition(
    config: AcquisitionConfig,
    *,
    registry: AdapterRegistry = DEFAULT_REGISTRY,
    clock_calibrations: Optional[dict[str, ClockCalibration]] = None,
) -> dict[str, Any]:
    """Run adapters concurrently and create a canonical session directory.

    Each device keeps its declared clock domain. A supplied affine calibration
    maps that clock into the reference domain before storage. The recorded start
    skew is retained so a supposedly synchronized session cannot hide a serial
    launch path.
    """
    output = Path(config.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    geometry_source = Path(config.geometry_path).resolve()
    geometry = SensorGeometry.model_validate(json.loads(geometry_source.read_text(encoding="utf-8")))
    dump_json(output / "geometry.json", geometry.model_dump(mode="json"))
    clock_calibrations = clock_calibrations or {}
    adapter_pairs = [(index, spec, registry.create(spec)) for index, spec in enumerate(config.adapters)]

    def acquire_one(
        index: int, spec: AdapterSpec, adapter: SensorAdapter
    ) -> tuple[int, AdapterSpec, SignalStream, float, float]:
        launched = time.perf_counter()
        stream = adapter.acquire(config.duration_seconds)
        elapsed = time.perf_counter() - launched
        return index, spec, stream, launched, elapsed

    results: list[tuple[int, AdapterSpec, SignalStream, float, float]] = []
    with ThreadPoolExecutor(max_workers=max(1, len(adapter_pairs))) as executor:
        futures = [
            executor.submit(acquire_one, index, spec, adapter)
            for index, spec, adapter in adapter_pairs
        ]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: item[0])
    launch_times = [item[3] for item in results]
    launch_skew_s = max(launch_times) - min(launch_times) if launch_times else 0.0

    stream_references: list[StreamReference] = []
    acquisition_records: list[dict[str, Any]] = []
    for _, spec, stream, _, elapsed in results:
        if spec.clock_domain in clock_calibrations:
            stream = apply_clock_calibration(stream, clock_calibrations[spec.clock_domain])
        stream.validate()
        destination = output / spec.output_name
        if destination.suffix.lower() != ".npz":
            destination = destination.with_suffix(".npz")
        save_stream(destination, stream)
        stream_references.append(
            StreamReference(
                modality=spec.modality,
                sensor_id=spec.sensor_id,
                path=destination.name,
                representation=spec.representation,
                clock_domain=spec.clock_domain,
                metadata={"adapter_type": spec.adapter_type},
            )
        )
        acquisition_records.append(
            {
                "sensor_id": spec.sensor_id,
                "adapter_type": spec.adapter_type,
                "samples": int(stream.timestamps_s.size),
                "features": int(np.asarray(stream.values).reshape(stream.timestamps_s.size, -1).shape[1]),
                "duration_s": float(stream.timestamps_s[-1] - stream.timestamps_s[0]),
                "elapsed_s": elapsed,
                "clock_domain": spec.clock_domain,
                "clock_calibrated": spec.clock_domain in clock_calibrations,
            }
        )

    manifest = SessionManifest(
        session_id=config.session_id,
        subject_id=config.subject_id,
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        geometry_path="geometry.json",
        streams=stream_references,
        targets=config.targets,
        protocol=config.protocol,
        consent_path=Path(config.consent_path).name if config.consent_path else None,
        environment_id=config.environment_id,
        posture=config.posture,
        condition=config.condition,
        metadata={
            "acquisition_orchestrator": "physioatlas.v0.4",
            "adapter_launch_skew_s": launch_skew_s,
        },
    )
    if config.consent_path:
        consent_source = Path(config.consent_path)
        (output / consent_source.name).write_bytes(consent_source.read_bytes())
    dump_json(output / "manifest.json", manifest.model_dump(mode="json"))
    report = {
        "schema_version": "physioatlas.acquisition-report.v2",
        "session_dir": str(output),
        "manifest": str(output / "manifest.json"),
        "adapter_launch_skew_s": launch_skew_s,
        "streams": acquisition_records,
        "adapters": registry.describe(),
    }
    dump_json(output / "acquisition-report.json", report)
    return report
