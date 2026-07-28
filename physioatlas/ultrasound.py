from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

from .io import SignalStream, save_stream


def phase_correlation_shift(reference: np.ndarray, moving: np.ndarray) -> tuple[float, float]:
    """Return the integer pixel shift ``(dy, dx)`` aligning moving to reference."""
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(moving, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("Ultrasound phase correlation expects equal 2-D frames")
    a = np.nan_to_num(a - a.mean())
    b = np.nan_to_num(b - b.mean())
    cross = np.fft.fft2(a) * np.conj(np.fft.fft2(b))
    cross /= np.maximum(np.abs(cross), 1e-12)
    correlation = np.fft.ifft2(cross).real
    peak = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    shifts = np.asarray(peak, dtype=np.float64)
    for axis, size in enumerate(a.shape):
        if shifts[axis] > size // 2:
            shifts[axis] -= size
    # Shift needed to align moving with reference; displacement of moving is opposite.
    return float(-shifts[0]), float(-shifts[1])


def extract_displacement_trace(
    frames: np.ndarray,
    *,
    pixel_spacing_m: tuple[float, float] = (1.0, 1.0),
    reference_mode: str = "previous",
) -> np.ndarray:
    images = np.asarray(frames)
    if images.ndim != 3 or images.shape[0] < 2:
        raise ValueError("frames must have shape [time, height, width] with at least two frames")
    if reference_mode not in {"previous", "first"}:
        raise ValueError("reference_mode must be 'previous' or 'first'")
    displacement = np.zeros((images.shape[0], 2), dtype=np.float32)
    cumulative = np.zeros(2, dtype=np.float64)
    for index in range(1, images.shape[0]):
        reference_index = index - 1 if reference_mode == "previous" else 0
        dy, dx = phase_correlation_shift(images[reference_index], images[index])
        step = np.asarray([dy * pixel_spacing_m[0], dx * pixel_spacing_m[1]])
        if reference_mode == "previous":
            cumulative += step
            displacement[index] = cumulative
        else:
            displacement[index] = step
    return displacement


def ultrasound_npz_to_displacement(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    frames_key: str = "frames",
    timestamps_key: str = "timestamps_s",
    pixel_spacing_m: tuple[float, float] = (1.0, 1.0),
) -> dict:
    with np.load(input_path, allow_pickle=False) as archive:
        if frames_key not in archive or timestamps_key not in archive:
            raise ValueError("Ultrasound NPZ is missing frames or timestamps")
        frames = np.asarray(archive[frames_key])
        timestamps = np.asarray(archive[timestamps_key], dtype=np.float64)
    if timestamps.shape[0] != frames.shape[0]:
        raise ValueError("Ultrasound frame and timestamp counts differ")
    values = extract_displacement_trace(frames, pixel_spacing_m=pixel_spacing_m)
    stream = SignalStream(timestamps_s=timestamps, values=values)
    save_stream(output_path, stream)
    return {
        "schema_version": "physioatlas.ultrasound-displacement.v1",
        "input": str(input_path),
        "output": str(output_path),
        "frames": int(frames.shape[0]),
        "features": 2,
        "units": "meters",
        "method": "2d_phase_correlation",
    }
