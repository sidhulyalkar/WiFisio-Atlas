import json
from pathlib import Path

import numpy as np

from physioatlas.adapters import csv_to_npz, ruview_jsonl_to_npz


def test_ruview_jsonl_conversion(tmp_path: Path):
    source = tmp_path / "capture.csi.jsonl"
    rows = [
        {
            "type": "raw_csi",
            "ts_ns": 1_000_000_000 + index * 10_000_000,
            "amplitudes": [1.0, 2.0],
            "quality_score": 0.9,
        }
        for index in range(3)
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    output = tmp_path / "wifi.npz"
    result = ruview_jsonl_to_npz(source, output)
    assert result["accepted_frames"] == 3
    with np.load(output) as archive:
        assert archive["values"].shape == (3, 4)
        np.testing.assert_allclose(archive["timestamps_s"], [0.0, 0.01, 0.02])


def test_csv_conversion(tmp_path: Path):
    source = tmp_path / "ecg.csv"
    source.write_text("time,lead\n10.0,0.1\n10.5,0.2\n")
    output = tmp_path / "ecg.npz"
    result = csv_to_npz(
        source,
        output,
        timestamp_column="time",
        value_columns=["lead"],
    )
    assert result["samples"] == 2
    with np.load(output) as archive:
        np.testing.assert_allclose(archive["timestamps_s"], [0.0, 0.5])
