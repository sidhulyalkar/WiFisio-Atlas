from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Union

import numpy as np

from .household_bfi_schema import (
    BfiFeatureProvenance,
    BfiProcessingReport,
    HouseholdBfiCalibration,
    LinkBfiCalibration,
    NormalizedBfiFeature,
    RawBfiFrame,
)

PathLike = Union[str, Path]


def _stream_id(frame: RawBfiFrame) -> str:
    return f"{frame.link_id}::{frame.client_id}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(frame: RawBfiFrame) -> tuple[Any, ...]:
    return (
        frame.link_id,
        frame.client_id,
        frame.frequency_band,
        frame.channel,
        frame.bandwidth_hz,
        frame.representation,
        tuple(frame.matrix_shape or []),
        len(frame.flattened_values()),
    )


def load_raw_bfi_jsonl(path: PathLike) -> dict[str, list[RawBfiFrame]]:
    """Load a strict BFI JSONL capture and reject metadata drift."""

    source = Path(path)
    grouped: dict[str, list[RawBfiFrame]] = {}
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError("frame must be a JSON object")
                frame = RawBfiFrame.model_validate(payload)
            except Exception as exc:
                raise ValueError(
                    f"Invalid BFI frame at {source}:{line_number}: {exc}"
                ) from exc
            grouped.setdefault(_stream_id(frame), []).append(frame)
    if not grouped:
        raise ValueError(f"No BFI frames found in {source}")
    for stream_id, frames in grouped.items():
        baseline = _metadata(frames[0])
        if any(_metadata(frame) != baseline for frame in frames[1:]):
            raise ValueError(
                f"BFI metadata changed on stream {stream_id!r}; "
                "use one fixed channel and representation per stream"
            )
        frames.sort(key=lambda frame: frame.timestamp_s)
        timestamps = np.asarray([frame.timestamp_s for frame in frames], dtype=float)
        if timestamps.size < 2 or np.any(np.diff(timestamps) <= 0.0):
            raise ValueError(
                f"BFI stream {stream_id!r} needs strictly increasing timestamps"
            )
    return dict(sorted(grouped.items()))


def _values(frames: list[RawBfiFrame]) -> np.ndarray:
    values = np.asarray([frame.flattened_values() for frame in frames], dtype=float)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("BFI feature matrix must be finite and rectangular")
    return values


def _calibrate_stream(
    stream_id: str, frames: list[RawBfiFrame]
) -> LinkBfiCalibration:
    if len(frames) < 2:
        raise ValueError(f"BFI stream {stream_id!r} needs at least two empty frames")
    values = _values(frames)
    median = np.median(values, axis=0)
    raw_mad = np.median(np.abs(values - median), axis=0) * 1.4826
    scale_floor = np.maximum(np.abs(median) * 0.01, 1e-6)
    mad = np.maximum(raw_mad, scale_floor)
    normalized = np.clip((values - median) / mad, -8.0, 8.0)
    energy = np.sqrt(np.mean(normalized * normalized, axis=1))
    energy_median = float(np.median(energy))
    energy_mad = float(
        np.median(np.abs(energy - energy_median)) * 1.4826
    )
    first = frames[0]
    return LinkBfiCalibration(
        stream_id=stream_id,
        link_id=first.link_id,
        client_id=first.client_id,
        frequency_band=first.frequency_band,
        channel=first.channel,
        bandwidth_hz=first.bandwidth_hz,
        representation=first.representation,
        matrix_shape=first.matrix_shape,
        feature_width=int(values.shape[1]),
        feature_median=median.tolist(),
        feature_mad=mad.tolist(),
        empty_energy_median=energy_median,
        empty_energy_mad=energy_mad,
        frames=len(frames),
    )


def save_household_bfi_calibration(
    calibration: HouseholdBfiCalibration, output_path: PathLike
) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        calibration.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )


def build_household_bfi_calibration(
    empty_capture_path: PathLike,
    output_path: PathLike,
    *,
    environment_id: str,
) -> HouseholdBfiCalibration:
    """Build and save per-stream robust empty-room BFI calibration."""

    grouped = load_raw_bfi_jsonl(empty_capture_path)
    streams = [
        _calibrate_stream(stream_id, frames)
        for stream_id, frames in grouped.items()
    ]
    calibration = HouseholdBfiCalibration(
        environment_id=environment_id,
        stream_order=[stream.stream_id for stream in streams],
        streams=streams,
    )
    save_household_bfi_calibration(calibration, output_path)
    return calibration


def load_household_bfi_calibration(path: PathLike) -> HouseholdBfiCalibration:
    source = Path(path)
    try:
        return HouseholdBfiCalibration.model_validate_json(
            source.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise ValueError(f"Invalid BFI calibration at {source}: {exc}") from exc


def _validate_against_calibration(
    frame: RawBfiFrame, calibration: LinkBfiCalibration
) -> None:
    expected = (
        calibration.link_id,
        calibration.client_id,
        calibration.frequency_band,
        calibration.channel,
        calibration.bandwidth_hz,
        calibration.representation,
        tuple(calibration.matrix_shape or []),
        calibration.feature_width,
    )
    if _metadata(frame) != expected:
        raise ValueError(
            f"BFI metadata does not match calibration for stream "
            f"{calibration.stream_id!r}"
        )


def process_household_bfi(
    input_path: PathLike,
    calibration_path: PathLike,
    output_path: PathLike,
) -> BfiProcessingReport:
    """Normalize BFI against empty-room baselines for later CSI/BFI fusion."""

    source = Path(input_path)
    calibration_source = Path(calibration_path)
    destination = Path(output_path)
    grouped = load_raw_bfi_jsonl(source)
    calibration = load_household_bfi_calibration(calibration_source)
    by_stream = {stream.stream_id: stream for stream in calibration.streams}
    unknown = sorted(set(grouped) - set(by_stream))
    if unknown:
        raise ValueError(f"Uncalibrated BFI streams: {unknown}")
    missing = sorted(set(by_stream) - set(grouped))
    if missing:
        raise ValueError(f"Calibrated BFI streams missing from capture: {missing}")
    source_hash = _sha256(source)
    calibration_hash = _sha256(calibration_source)
    emitted: list[NormalizedBfiFeature] = []
    not_observable = 0
    for stream_id in calibration.stream_order:
        stream = by_stream[stream_id]
        frames = grouped[stream_id]
        median = np.asarray(stream.feature_median, dtype=float)
        mad = np.asarray(stream.feature_mad, dtype=float)
        previous: np.ndarray | None = None
        for frame in frames:
            _validate_against_calibration(frame, stream)
            normalized = np.clip(
                (np.asarray(frame.flattened_values()) - median) / mad, -8.0, 8.0
            )
            delta = np.zeros_like(normalized) if previous is None else normalized - previous
            perturbation = float(np.sqrt(np.mean(normalized * normalized)))
            motion = float(np.sqrt(np.mean(delta * delta)))
            # A large calibrated perturbation is signal, not poor observability.
            # Until packet-loss/coherence telemetry is added, retain the receiver's
            # explicit quality score rather than manufacturing confidence.
            observability = frame.quality
            status = "observable" if observability >= 0.5 else "not_observable"
            not_observable += int(status == "not_observable")
            emitted.append(
                NormalizedBfiFeature(
                    timestamp_s=frame.timestamp_s,
                    stream_id=stream_id,
                    link_id=frame.link_id,
                    client_id=frame.client_id,
                    frequency_band=frame.frequency_band,
                    channel=frame.channel,
                    normalized_features=normalized.tolist(),
                    temporal_delta=delta.tolist(),
                    perturbation_energy=perturbation,
                    motion_energy=motion,
                    quality=frame.quality,
                    observability=observability,
                    status=status,
                    provenance=BfiFeatureProvenance(
                        source_capture=str(source),
                        source_sha256=source_hash,
                        calibration_sha256=calibration_hash,
                        packet_sequence=frame.packet_sequence,
                    ),
                )
            )
            previous = normalized
    emitted.sort(key=lambda item: (item.timestamp_s, item.stream_id))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for feature in emitted:
            handle.write(feature.model_dump_json() + "\n")
    return BfiProcessingReport(
        input_path=str(source),
        calibration_path=str(calibration_source),
        output_path=str(destination),
        accepted_frames=sum(len(frames) for frames in grouped.values()),
        observations_emitted=len(emitted),
        streams_observed=calibration.stream_order,
        not_observable_frames=not_observable,
    )
