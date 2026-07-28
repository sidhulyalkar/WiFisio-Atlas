from pathlib import Path

import numpy as np
import pytest

from physioatlas.io import SignalStream, load_stream, save_stream
from physioatlas.schema import Modality, StreamReference
from physioatlas.synchronization import align_streams, resample_stream


def test_stream_round_trip(tmp_path: Path):
    path = tmp_path / "stream.npz"
    original = SignalStream(
        timestamps_s=np.array([0.0, 0.5, 1.0]),
        values=np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
        quality=np.array([1.0, 0.8, 1.0], dtype=np.float32),
    )
    save_stream(path, original)
    loaded = load_stream(
        StreamReference(modality=Modality.ECG, sensor_id="ecg", path=str(path))
    )
    np.testing.assert_allclose(loaded.values, original.values)
    np.testing.assert_allclose(loaded.quality, original.quality)


def test_resampling_marks_large_gaps_unobserved():
    stream = SignalStream(
        timestamps_s=np.array([0.0, 0.1, 1.0, 1.1]),
        values=np.array([0.0, 1.0, 2.0, 3.0]),
    )
    target = np.array([0.05, 0.5, 1.05])
    values, observed = resample_stream(stream, target, max_gap_s=0.15)
    assert observed.tolist() == [True, False, True]
    assert values[1, 0] == 0.0


def test_alignment_requires_overlap():
    streams = {
        "a": SignalStream(np.array([0.0, 0.1]), np.ones((2, 1))),
        "b": SignalStream(np.array([1.0, 1.1]), np.ones((2, 1))),
    }
    with pytest.raises(ValueError, match="share a valid time interval"):
        align_streams(streams, sample_rate_hz=10)
