from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from .household_csi_schema import LinkCsiCalibration


@dataclass(frozen=True)
class RawLinkCsiFrame:
    timestamp_s: float
    link_id: str
    amplitude: np.ndarray
    phase: Optional[np.ndarray]
    quality: Optional[float]
    sequence: Optional[int]
    frequency_band: str
    center_frequency_hz: Optional[float]
    channel: Optional[int]
    bandwidth_hz: Optional[float]


@dataclass(frozen=True)
class LinkTrace:
    timestamps_s: np.ndarray
    signed_signal: np.ndarray
    energy: np.ndarray
    quality: np.ndarray


def _finite_vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size < 4 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain at least four finite values")
    return array


def _frequency_metadata(payload: dict[str, Any]) -> tuple[str, Optional[float], Optional[int], Optional[float]]:
    band = str(payload.get("frequency_band", payload.get("band", "unknown"))).lower()
    aliases = {
        "2g": "2.4ghz",
        "2.4g": "2.4ghz",
        "2.4": "2.4ghz",
        "5g": "5ghz",
        "5": "5ghz",
        "6g": "6ghz",
        "6": "6ghz",
    }
    band = aliases.get(band, band)
    frequency_value = payload.get(
        "center_frequency_hz", payload.get("frequency_hz", payload.get("center_freq_hz"))
    )
    frequency = None if frequency_value is None else float(frequency_value)
    channel_value = payload.get("channel", payload.get("wifi_channel"))
    channel = None if channel_value is None else int(channel_value)
    bandwidth_value = payload.get("bandwidth_hz")
    if bandwidth_value is None and payload.get("bandwidth_mhz") is not None:
        bandwidth_value = float(payload["bandwidth_mhz"]) * 1e6
    bandwidth = None if bandwidth_value is None else float(bandwidth_value)
    if frequency is not None:
        if not math.isfinite(frequency) or frequency <= 0.0:
            raise ValueError("center frequency must be positive and finite")
        if band == "unknown":
            band = "2.4ghz" if frequency < 3e9 else "5ghz" if frequency < 5.925e9 else "6ghz"
    if band not in {"2.4ghz", "5ghz", "6ghz", "unknown"}:
        raise ValueError("frequency_band must be 2.4ghz, 5ghz, 6ghz, or unknown")
    if bandwidth is not None and (not math.isfinite(bandwidth) or bandwidth <= 0.0):
        raise ValueError("bandwidth must be positive and finite")
    return band, frequency, channel, bandwidth


def _normalize_raw_frame(payload: dict[str, Any]) -> RawLinkCsiFrame:
    timestamp = payload.get("timestamp_s")
    if timestamp is None:
        timestamp_ns = payload.get("timestamp_ns", payload.get("ts_ns"))
        if timestamp_ns is None:
            raise ValueError("CSI frame is missing timestamp_s/timestamp_ns")
        timestamp = float(timestamp_ns) / 1e9
    link_id = payload.get("link_id")
    if link_id is None and payload.get("node_id") is not None:
        link_id = f"router-to-node-{payload['node_id']}"
    if not isinstance(link_id, str) or not link_id:
        raise ValueError("CSI frame is missing a stable link_id")
    amplitude = payload.get("amplitude", payload.get("amplitudes"))
    phase = payload.get("phase")
    if amplitude is None:
        iq_hex = payload.get("iq_hex")
        if not iq_hex:
            raise ValueError("CSI frame is missing amplitude or iq_hex")
        raw = np.frombuffer(bytes.fromhex(str(iq_hex)), dtype=np.int8).astype(float)
        if raw.size < 8 or raw.size % 2:
            raise ValueError("iq_hex must contain adjacent I/Q byte pairs")
        iq = raw.reshape(-1, 2)
        amplitude_array = np.sqrt(np.sum(iq * iq, axis=1))
        phase_array: Optional[np.ndarray] = np.arctan2(iq[:, 1], iq[:, 0])
    else:
        amplitude_array = _finite_vector(amplitude, "amplitude")
        phase_array = None if phase is None else _finite_vector(phase, "phase")
    if phase_array is not None and phase_array.size != amplitude_array.size:
        raise ValueError("phase width must match amplitude width")
    quality_value = payload.get("quality", payload.get("quality_score"))
    quality: Optional[float] = None
    if quality_value is not None:
        quality = float(quality_value)
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("quality must be within [0, 1]")
    elif payload.get("rssi_dbm") is not None and payload.get("noise_floor_dbm") is not None:
        snr = float(payload["rssi_dbm"]) - float(payload["noise_floor_dbm"])
        quality = float(np.clip((snr - 3.0) / 27.0, 0.0, 1.0))
    band, frequency, channel, bandwidth = _frequency_metadata(payload)
    sequence = payload.get("sequence")
    return RawLinkCsiFrame(
        timestamp_s=float(timestamp),
        link_id=link_id,
        amplitude=amplitude_array,
        phase=phase_array,
        quality=quality,
        sequence=None if sequence is None else int(sequence),
        frequency_band=band,
        center_frequency_hz=frequency,
        channel=channel,
        bandwidth_hz=bandwidth,
    )


def load_raw_csi_jsonl(
    path: Union[str, Path],
) -> tuple[dict[str, list[RawLinkCsiFrame]], dict[str, Any]]:
    grouped: dict[str, list[RawLinkCsiFrame]] = {}
    rejected: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError("frame must be a JSON object")
                if "rvcsi_capture_version" in payload:
                    continue
                frame = _normalize_raw_frame(payload)
                grouped.setdefault(frame.link_id, []).append(frame)
            except Exception as exc:
                rejected.append({"line": line_number, "reason": str(exc)})
    for link_id, frames in grouped.items():
        frames.sort(key=lambda item: item.timestamp_s)
        widths = {item.amplitude.size for item in frames}
        metadata = {
            (item.frequency_band, item.center_frequency_hz, item.channel, item.bandwidth_hz)
            for item in frames
        }
        if len(widths) != 1:
            raise ValueError(f"CSI feature width changed on link {link_id!r}")
        if len(metadata) != 1:
            raise ValueError(
                f"Frequency metadata changed on link {link_id!r}; use one fixed channel per link"
            )
        timestamps = np.asarray([item.timestamp_s for item in frames])
        if timestamps.size < 2 or np.any(np.diff(timestamps) <= 0):
            raise ValueError(f"Link {link_id!r} timestamps must be strictly increasing")
    if not grouped:
        raise ValueError(f"No valid link CSI frames found in {path}")
    return grouped, {
        "accepted_frames": sum(len(items) for items in grouped.values()),
        "rejected_frames": len(rejected),
        "rejections": rejected[:50],
    }


def circular_mean(phases: np.ndarray) -> np.ndarray:
    return np.arctan2(np.mean(np.sin(phases), axis=0), np.mean(np.cos(phases), axis=0))


def phase_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * (left - right)))


def link_empty_calibration(
    link_id: str, frames: list[RawLinkCsiFrame]
) -> LinkCsiCalibration:
    amplitude = np.stack([item.amplitude for item in frames])
    median = np.median(amplitude, axis=0)
    mad = np.median(np.abs(amplitude - median), axis=0) * 1.4826
    floor = np.maximum(np.abs(median) * 0.01, 1e-3)
    mad = np.maximum(mad, floor)
    residual = (np.log1p(amplitude) - np.log1p(median)) / np.maximum(
        mad / np.maximum(1.0 + median, 1e-6), 1e-3
    )
    energy = np.sqrt(np.mean(np.clip(residual, -8.0, 8.0) ** 2, axis=1))
    phases = [item.phase for item in frames if item.phase is not None]
    phase_mean = circular_mean(np.stack(phases)).tolist() if len(phases) == len(frames) else None
    metadata = frames[0]
    return LinkCsiCalibration(
        link_id=link_id,
        n_subcarriers=int(amplitude.shape[1]),
        frequency_band=metadata.frequency_band,
        center_frequency_hz=metadata.center_frequency_hz,
        channel=metadata.channel,
        bandwidth_hz=metadata.bandwidth_hz,
        amplitude_median=median.tolist(),
        amplitude_mad=mad.tolist(),
        phase_mean=phase_mean,
        phase_available=phase_mean is not None,
        empty_energy_median=float(np.median(energy)),
        empty_energy_mad=float(np.median(np.abs(energy - np.median(energy))) * 1.4826),
        frames=len(frames),
    )


def trace_for_link(
    frames: list[RawLinkCsiFrame], calibration: LinkCsiCalibration
) -> LinkTrace:
    median = np.asarray(calibration.amplitude_median)
    mad = np.asarray(calibration.amplitude_mad)
    signed_values, energy_values, quality_values = [], [], []
    for frame in frames:
        if frame.amplitude.size != calibration.n_subcarriers:
            raise ValueError(f"Subcarrier width mismatch for link {frame.link_id!r}")
        if frame.frequency_band != calibration.frequency_band:
            raise ValueError(f"Frequency band mismatch for link {frame.link_id!r}")
        log_residual = np.log1p(frame.amplitude) - np.log1p(median)
        robust_scale = np.maximum(mad / np.maximum(1.0 + median, 1e-6), 1e-3)
        standardized = np.clip(log_residual / robust_scale, -8.0, 8.0)
        signed = float(np.median(standardized))
        amplitude_energy = float(np.sqrt(np.mean(standardized**2)))
        phase_energy = 0.0
        if frame.phase is not None and calibration.phase_mean is not None:
            phase_residual = phase_distance(frame.phase, np.asarray(calibration.phase_mean))
            phase_residual -= float(np.median(phase_residual))
            phase_energy = float(np.sqrt(np.mean(phase_residual**2))) / np.pi
        signed_values.append(signed)
        energy_values.append(amplitude_energy + 0.25 * phase_energy)
        quality_values.append(np.nan if frame.quality is None else frame.quality)
    quality = np.asarray(quality_values, dtype=float)
    if np.all(np.isnan(quality)):
        variability = np.asarray(energy_values)
        quality[:] = np.clip(1.0 / (1.0 + np.maximum(variability - 8.0, 0.0)), 0.25, 0.75)
    else:
        quality[np.isnan(quality)] = float(np.nanmedian(quality))
    return LinkTrace(
        timestamps_s=np.asarray([item.timestamp_s for item in frames], dtype=float),
        signed_signal=np.asarray(signed_values, dtype=float),
        energy=np.asarray(energy_values, dtype=float),
        quality=np.clip(quality, 0.0, 1.0),
    )
