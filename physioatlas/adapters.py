from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .io import SignalStream, save_stream


def ruview_jsonl_to_npz(input_path: Union[str, Path], output_path: Union[str, Path]) -> dict:
    """Convert RuView ``*.csi.jsonl`` or ``*.rvcsi`` frames to an NPZ stream.

    Both amplitude and phase are retained when available. Each output row is
    ``[amplitude..., phase...]``. Frames with a different feature width are
    rejected rather than padded silently, because changing subcarrier layouts
    is a hardware/protocol event that should be modeled explicitly.
    """
    timestamps_ns: list[int] = []
    rows: list[np.ndarray] = []
    quality: list[float] = []
    expected_width: Optional[int] = None
    rejected = 0
    with Path(input_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if "rvcsi_capture_version" in payload:
                continue
            timestamp_ns = payload.get("timestamp_ns", payload.get("ts_ns"))
            if timestamp_ns is None:
                rejected += 1
                continue
            amplitude = payload.get("amplitude", payload.get("amplitudes"))
            phase = payload.get("phase")
            if amplitude is None:
                iq_hex = payload.get("iq_hex")
                if not iq_hex:
                    rejected += 1
                    continue
                raw = np.frombuffer(bytes.fromhex(iq_hex), dtype=np.int8).astype(np.float32)
                if raw.size % 2:
                    rejected += 1
                    continue
                # RuView recorder stores adjacent I/Q bytes. Amplitude remains
                # valid even when chipset-specific real/imag ordering differs.
                iq = raw.reshape(-1, 2)
                amplitude_array = np.sqrt(np.sum(iq * iq, axis=1))
                phase_array = np.arctan2(iq[:, 1], iq[:, 0])
            else:
                amplitude_array = np.asarray(amplitude, dtype=np.float32)
                phase_array = (
                    np.asarray(phase, dtype=np.float32)
                    if phase is not None
                    else np.zeros_like(amplitude_array)
                )
            row = np.concatenate([amplitude_array, phase_array]).astype(np.float32)
            if expected_width is None:
                expected_width = row.size
            if row.size != expected_width or not np.all(np.isfinite(row)):
                rejected += 1
                continue
            timestamps_ns.append(int(timestamp_ns))
            rows.append(row)
            quality.append(float(payload.get("quality_score", 1.0)))
    if not rows:
        raise ValueError(f"No valid CSI frames found in {input_path}")
    order = np.argsort(np.asarray(timestamps_ns, dtype=np.int64), kind="stable")
    timestamp_array = np.asarray(timestamps_ns, dtype=np.float64)[order]
    timestamp_array = (timestamp_array - timestamp_array[0]) / 1e9
    # Duplicate timestamps can occur after relay/replay. Keep the first frame.
    unique = np.concatenate([[True], np.diff(timestamp_array) > 0])
    stream = SignalStream(
        timestamps_s=timestamp_array[unique],
        values=np.stack(rows)[order][unique],
        quality=np.asarray(quality, dtype=np.float32)[order][unique],
    )
    save_stream(output_path, stream)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "accepted_frames": int(stream.timestamps_s.size),
        "rejected_frames": rejected,
        "features": int(stream.values.shape[1]),
    }


def csv_to_npz(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    timestamp_column: str,
    value_columns: list[str],
    timestamp_scale: float = 1.0,
) -> dict:
    """Convert synchronized physiological CSV channels into a stream NPZ."""
    timestamps: list[float] = []
    rows: list[list[float]] = []
    with Path(input_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {timestamp_column, *value_columns} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing columns: {sorted(missing)}")
        for row in reader:
            timestamps.append(float(row[timestamp_column]) * timestamp_scale)
            rows.append([float(row[column]) for column in value_columns])
    if not rows:
        raise ValueError(f"CSV {input_path} contains no data rows")
    timestamp_array = np.asarray(timestamps, dtype=np.float64)
    timestamp_array -= timestamp_array[0]
    stream = SignalStream(timestamp_array, np.asarray(rows, dtype=np.float32))
    save_stream(output_path, stream)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "samples": len(rows),
        "features": len(value_columns),
    }
