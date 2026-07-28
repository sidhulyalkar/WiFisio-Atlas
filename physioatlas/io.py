from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np

from .schema import SensorGeometry, SessionManifest, StreamReference


@dataclass(frozen=True)
class SignalStream:
    timestamps_s: np.ndarray
    values: np.ndarray
    quality: Optional[np.ndarray] = None

    def validate(self) -> None:
        if self.timestamps_s.ndim != 1:
            raise ValueError("timestamps_s must be one-dimensional")
        if self.values.shape[0] != self.timestamps_s.shape[0]:
            raise ValueError("values and timestamps_s must have equal first dimension")
        if self.quality is not None and self.quality.shape[0] != self.timestamps_s.shape[0]:
            raise ValueError("quality and timestamps_s must have equal first dimension")
        if self.timestamps_s.size and np.any(np.diff(self.timestamps_s) <= 0):
            raise ValueError("timestamps_s must be strictly increasing")
        if not np.all(np.isfinite(self.timestamps_s)):
            raise ValueError("timestamps_s contains non-finite values")


@dataclass(frozen=True)
class LoadedSession:
    manifest: SessionManifest
    geometry: SensorGeometry
    streams: dict[str, SignalStream]


def load_json(path: Union[str, Path]) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Union[str, Path], payload: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_manifest(path: Union[str, Path], *, resolve_paths: bool = True) -> SessionManifest:
    manifest = SessionManifest.model_validate(load_json(path))
    return manifest.resolve(path) if resolve_paths else manifest


def load_geometry(path: Union[str, Path]) -> SensorGeometry:
    return SensorGeometry.model_validate(load_json(path))


def save_stream(path: Union[str, Path], stream: SignalStream) -> None:
    stream.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, np.ndarray] = {
        "timestamps_s": np.asarray(stream.timestamps_s, dtype=np.float64),
        "values": np.asarray(stream.values, dtype=np.float32),
    }
    if stream.quality is not None:
        kwargs["quality"] = np.asarray(stream.quality, dtype=np.float32)
    np.savez_compressed(destination, **kwargs)


def load_stream(reference: StreamReference) -> SignalStream:
    path = Path(reference.path)
    if path.suffix.lower() != ".npz":
        raise ValueError(f"Only .npz streams are currently supported, got {path}")
    with np.load(path, allow_pickle=False) as archive:
        if reference.timestamp_key not in archive or reference.value_key not in archive:
            raise ValueError(f"{path} is missing configured timestamp/value arrays")
        quality = None
        if reference.quality_key and reference.quality_key in archive:
            quality = np.asarray(archive[reference.quality_key])
        stream = SignalStream(
            timestamps_s=np.asarray(archive[reference.timestamp_key], dtype=np.float64),
            values=np.asarray(archive[reference.value_key]),
            quality=quality,
        )
    stream.validate()
    return stream


def load_session(path: Union[str, Path]) -> LoadedSession:
    manifest = load_manifest(path, resolve_paths=True)
    geometry = load_geometry(manifest.geometry_path)
    streams: dict[str, SignalStream] = {}
    for reference in manifest.streams:
        key = f"{reference.modality.value}:{reference.sensor_id}"
        if key in streams:
            raise ValueError(f"Duplicate stream key {key}")
        streams[key] = load_stream(reference)
    return LoadedSession(manifest=manifest, geometry=geometry, streams=streams)


def iter_manifests(root: Union[str, Path]) -> Iterable[Path]:
    yield from sorted(Path(root).rglob("manifest.json"))
