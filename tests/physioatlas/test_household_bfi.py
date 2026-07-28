import json
from pathlib import Path

import pytest

from physioatlas.household_bfi import (
    build_household_bfi_calibration,
    load_raw_bfi_jsonl,
    process_household_bfi,
)


def _frame(
    timestamp: float,
    values: list[float],
    *,
    channel: int = 36,
    quality: float = 0.9,
) -> dict:
    return {
        "timestamp_s": timestamp,
        "link_id": "router-ap",
        "client_id": "monitor-01",
        "frequency_band": "5ghz",
        "channel": channel,
        "bandwidth_hz": 80e6,
        "packet_sequence": int(timestamp * 10),
        "quality": quality,
        "feature_vector": values,
    }


def _write(path: Path, frames: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(frame) + "\n" for frame in frames),
        encoding="utf-8",
    )


def test_valid_capture_calibration_and_deterministic_features(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    capture = tmp_path / "capture.jsonl"
    calibration = tmp_path / "calibration.json"
    output_a = tmp_path / "features-a.jsonl"
    output_b = tmp_path / "features-b.jsonl"
    _write(
        empty,
        [
            _frame(0.0, [10.0, 20.0, 30.0, 40.0]),
            _frame(0.1, [10.2, 19.8, 30.3, 39.6]),
            _frame(0.2, [9.8, 20.2, 29.7, 40.4]),
        ],
    )
    _write(
        capture,
        [
            _frame(1.0, [10.1, 20.1, 30.1, 40.1]),
            _frame(1.1, [11.0, 19.0, 31.0, 39.0]),
        ],
    )
    result = build_household_bfi_calibration(
        empty, calibration, environment_id="living-room-v1"
    )
    assert result.research_only is True
    assert result.identity_claim is False
    assert result.stream_order == ["router-ap::monitor-01"]
    report_a = process_household_bfi(capture, calibration, output_a)
    report_b = process_household_bfi(capture, calibration, output_b)
    assert report_a.observations_emitted == 2
    assert report_b.observations_emitted == 2
    assert output_a.read_text(encoding="utf-8") == output_b.read_text(encoding="utf-8")
    payloads = [
        json.loads(line) for line in output_a.read_text(encoding="utf-8").splitlines()
    ]
    assert payloads[0]["temporal_delta"] == [0.0, 0.0, 0.0, 0.0]
    assert payloads[1]["motion_energy"] > 0.0
    assert payloads[0]["identity_claim"] is False
    assert payloads[0]["provenance"]["source_sha256"]
    assert payloads[0]["provenance"]["calibration_sha256"]


@pytest.mark.parametrize(
    "frame, match",
    [
        (_frame(0.0, [1.0, 2.0, float("nan"), 4.0]), "finite"),
        (_frame(0.0, [1.0, 2.0, 3.0]), "at least four"),
        ({**_frame(0.0, [1.0] * 4), "unexpected": True}, "extra"),
        (
            {
                **_frame(0.0, [1.0] * 4),
                "compressed_beamforming_matrix": [[1.0, 2.0], [3.0, 4.0]],
            },
            "exactly one",
        ),
    ],
)
def test_malformed_or_nonfinite_input_fails_closed(
    tmp_path: Path, frame: dict, match: str
) -> None:
    capture = tmp_path / "invalid.jsonl"
    _write(capture, [frame, {**frame, "timestamp_s": 0.1}])
    with pytest.raises(ValueError, match=match):
        load_raw_bfi_jsonl(capture)


def test_metadata_or_channel_drift_fails_closed(tmp_path: Path) -> None:
    capture = tmp_path / "drift.jsonl"
    _write(
        capture,
        [
            _frame(0.0, [1.0, 2.0, 3.0, 4.0], channel=36),
            _frame(0.1, [1.0, 2.0, 3.0, 4.0], channel=40),
        ],
    )
    with pytest.raises(ValueError, match="metadata changed"):
        load_raw_bfi_jsonl(capture)


def test_compressed_matrix_contract_is_supported(tmp_path: Path) -> None:
    capture = tmp_path / "matrix.jsonl"
    frames = []
    for timestamp in (0.0, 0.1):
        payload = _frame(timestamp, [1.0] * 4)
        payload.pop("feature_vector")
        payload["compressed_beamforming_matrix"] = [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
        frames.append(payload)
    _write(capture, frames)
    grouped = load_raw_bfi_jsonl(capture)
    frame = grouped["router-ap::monitor-01"][0]
    assert frame.representation == "compressed_beamforming_matrix"
    assert frame.flattened_values() == [1.0, 2.0, 3.0, 4.0]
