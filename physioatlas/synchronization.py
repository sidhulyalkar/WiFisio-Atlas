from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io import SignalStream


@dataclass(frozen=True)
class AlignedStreams:
    timestamps_s: np.ndarray
    values: dict[str, np.ndarray]
    observed: dict[str, np.ndarray]
    quality: dict[str, np.ndarray]


def _flatten_features(values: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        return values[:, None]
    return values.reshape(values.shape[0], -1)


def resample_stream(
    stream: SignalStream,
    target_timestamps_s: np.ndarray,
    *,
    max_gap_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly resample a stream and return values plus an observation mask.

    Values outside the stream range or farther than ``max_gap_s`` from a source
    sample are zero-filled and marked unobserved. This prevents silent
    extrapolation across sensor dropouts.
    """
    stream.validate()
    source_t = stream.timestamps_s
    source_v = _flatten_features(np.asarray(stream.values, dtype=np.float64))
    target_t = np.asarray(target_timestamps_s, dtype=np.float64)
    output = np.zeros((target_t.size, source_v.shape[1]), dtype=np.float32)
    observed = np.zeros(target_t.size, dtype=bool)
    if source_t.size == 0 or target_t.size == 0:
        return output, observed

    insertion = np.searchsorted(source_t, target_t, side="left")
    left_idx = np.clip(insertion - 1, 0, source_t.size - 1)
    right_idx = np.clip(insertion, 0, source_t.size - 1)
    nearest_gap = np.minimum(
        np.abs(target_t - source_t[left_idx]),
        np.abs(source_t[right_idx] - target_t),
    )
    observed = (
        (target_t >= source_t[0])
        & (target_t <= source_t[-1])
        & (nearest_gap <= max_gap_s)
    )
    for column in range(source_v.shape[1]):
        output[:, column] = np.interp(
            target_t,
            source_t,
            source_v[:, column],
            left=0.0,
            right=0.0,
        ).astype(np.float32)
    output[~observed] = 0.0
    return output, observed


def align_streams(
    streams: dict[str, SignalStream],
    *,
    sample_rate_hz: float,
    max_gap_s: float = 0.25,
    overlap_only: bool = True,
) -> AlignedStreams:
    if not streams:
        raise ValueError("At least one stream is required")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")

    starts = [stream.timestamps_s[0] for stream in streams.values() if stream.timestamps_s.size]
    stops = [stream.timestamps_s[-1] for stream in streams.values() if stream.timestamps_s.size]
    if len(starts) != len(streams):
        raise ValueError("Empty streams cannot be aligned")
    start = max(starts) if overlap_only else min(starts)
    stop = min(stops) if overlap_only else max(stops)
    if stop <= start:
        raise ValueError("Streams do not share a valid time interval")

    step = 1.0 / sample_rate_hz
    timestamps = np.arange(start, stop + step * 0.25, step, dtype=np.float64)
    values: dict[str, np.ndarray] = {}
    observed: dict[str, np.ndarray] = {}
    quality: dict[str, np.ndarray] = {}
    for key, stream in streams.items():
        values[key], observed[key] = resample_stream(
            stream, timestamps, max_gap_s=max_gap_s
        )
        if stream.quality is None:
            quality[key] = observed[key].astype(np.float32)
        else:
            quality_stream = SignalStream(
                timestamps_s=stream.timestamps_s,
                values=np.asarray(stream.quality, dtype=np.float32).reshape(-1, 1),
            )
            quality_values, quality_observed = resample_stream(
                quality_stream, timestamps, max_gap_s=max_gap_s
            )
            quality[key] = np.clip(quality_values[:, 0], 0.0, 1.0)
            quality[key][~quality_observed] = 0.0
    return AlignedStreams(
        timestamps_s=timestamps,
        values=values,
        observed=observed,
        quality=quality,
    )
