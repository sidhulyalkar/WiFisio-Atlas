import json
from pathlib import Path

import numpy as np
import pytest

from physioatlas.household_csi import (
    build_household_csi_calibration,
    process_mixed_household_csi,
)
from physioatlas.household_csi_input import load_raw_csi_jsonl
from physioatlas.household_csi_schema import CsiProcessingConfig
from physioatlas.household_live import observation_from_payload
from physioatlas.schema import Modality


LINKS = {
    "link-24-a": ("2.4ghz", 2.437e9, 6),
    "link-24-b": ("2.4ghz", 2.462e9, 11),
    "link-5-a": ("5ghz", 5.18e9, 36),
    "link-5-b": ("5ghz", 5.745e9, 149),
}
ZONE_A = np.asarray([0.85, 0.22, 0.10, 0.46])
ZONE_B = np.asarray([0.12, 0.54, 0.81, 0.18])


def _amplitude(standardized: float, base: float = 20.0) -> list[float]:
    robust_log_scale = (base * 0.01) / (1.0 + base)
    value = np.exp(np.log1p(base) + robust_log_scale * standardized) - 1.0
    return np.full(16, value).tolist()


def _write_capture(
    path: Path,
    duration_s: float,
    response,
    *,
    sample_rate_hz: float = 20.0,
    timestamp_offsets: dict[str, float] | None = None,
) -> None:
    rng = np.random.default_rng(9)
    with path.open("w", encoding="utf-8") as handle:
        for sample in range(int(duration_s * sample_rate_hz)):
            timestamp = sample / sample_rate_hz
            for index, (link_id, (band, frequency, channel)) in enumerate(LINKS.items()):
                standardized = float(response(timestamp, index))
                payload = {
                    "timestamp_s": timestamp + (timestamp_offsets or {}).get(link_id, 0.0),
                    "sequence": sample,
                    "link_id": link_id,
                    "amplitude": _amplitude(standardized)
                    if standardized
                    else (20.0 + rng.normal(0.0, 0.01, 16)).tolist(),
                    "quality": 0.93,
                    "frequency_band": band,
                    "center_frequency_hz": frequency,
                    "channel": channel,
                    "bandwidth_mhz": 20,
                }
                handle.write(json.dumps(payload) + "\n")


def test_multiband_calibration_and_two_person_observation_bridge(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    zone_a = tmp_path / "zone-a.jsonl"
    zone_b = tmp_path / "zone-b.jsonl"
    mixed = tmp_path / "mixed.jsonl"
    calibration = tmp_path / "calibration.json"
    observations = tmp_path / "observations.jsonl"
    _write_capture(empty, 12.0, lambda _t, _index: 0.0)
    _write_capture(
        zone_a, 12.0, lambda t, index: 2.0 * ZONE_A[index] + 0.08 * np.sin(2 * np.pi * 0.2 * t)
    )
    _write_capture(
        zone_b, 12.0, lambda t, index: 2.0 * ZONE_B[index] + 0.08 * np.sin(2 * np.pi * 0.3 * t)
    )
    _write_capture(
        mixed,
        40.0,
        lambda t, index: (
            1.2 * ZONE_A[index]
            + 1.0 * ZONE_B[index]
            + 0.22 * ZONE_A[index] * np.sin(2 * np.pi * 0.2 * t)
            + 0.18 * ZONE_B[index] * np.sin(2 * np.pi * 0.3 * t)
        ),
    )
    result = build_household_csi_calibration(
        empty,
        {
            "desk": ([1.0, 1.0], zone_a),
            "sofa": ([4.0, 2.0], zone_b),
        },
        calibration,
        environment_id="test-room",
    )
    assert result["condition_number"] < 5.0
    report = process_mixed_household_csi(
        mixed,
        calibration,
        observations,
        config=CsiProcessingConfig(
            vital_window_s=30.0,
            stride_s=5.0,
            minimum_observability=0.4,
            require_frequency_bands=["2.4ghz", "5ghz"],
        ),
    )
    assert report["frequency_bands_observed"] == ["2.4ghz", "5ghz"]
    assert report["observations_emitted"] >= 4
    payloads = [
        json.loads(line) for line in observations.read_text(encoding="utf-8").splitlines()
    ]
    assert {item["zone_id"] for item in payloads} == {"desk", "sofa"}
    assert all(item["heart_rate_bpm"] is None for item in payloads)
    assert all(item["provenance"]["multi_band_fusion"] for item in payloads)
    rates_by_zone = {
        zone: np.median(
            [
                item["respiratory_rate_bpm"]
                for item in payloads
                if item["zone_id"] == zone and item["respiratory_rate_bpm"] is not None
            ]
        )
        for zone in ("desk", "sofa")
    }
    assert rates_by_zone["desk"] == pytest.approx(12.0, abs=2.1)
    assert rates_by_zone["sofa"] == pytest.approx(18.0, abs=2.1)
    observation = observation_from_payload(payloads[0])
    assert observation.embedding_versions[Modality.WIFI_CSI] == (
        "physioatlas.household-csi-identity.v1"
    )


def test_channel_hopping_and_missing_required_band_fail_closed(tmp_path: Path) -> None:
    capture = tmp_path / "hopping.jsonl"
    first = {
        "timestamp_s": 0.0,
        "link_id": "link",
        "amplitude": [1.0] * 8,
        "frequency_band": "5ghz",
        "channel": 36,
    }
    second = {**first, "timestamp_s": 0.1, "channel": 40}
    capture.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fixed channel"):
        load_raw_csi_jsonl(capture)


def test_shared_packet_sequences_calibrate_receiver_clocks(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    zone = tmp_path / "zone.jsonl"
    sync = tmp_path / "sync.jsonl"
    calibration = tmp_path / "calibration.json"
    offsets = {
        "link-24-a": 0.0,
        "link-24-b": 0.018,
        "link-5-a": -0.011,
        "link-5-b": 0.027,
    }
    _write_capture(empty, 5.0, lambda _t, _index: 0.0)
    _write_capture(zone, 5.0, lambda _t, index: 2.0 * ZONE_A[index])
    _write_capture(
        sync,
        5.0,
        lambda t, index: ZONE_A[index] * np.sin(2 * np.pi * 0.5 * t),
        timestamp_offsets=offsets,
    )
    result = build_household_csi_calibration(
        empty,
        {"desk": ([1.0, 1.0], zone)},
        calibration,
        environment_id="clock-test",
        synchronization_capture=sync,
    )
    assert result["synchronization_verified"]
    payload = json.loads(calibration.read_text(encoding="utf-8"))
    by_link = {item["link_id"]: item for item in payload["links"]}
    for link_id, expected in offsets.items():
        assert by_link[link_id]["clock_offset_s"] == pytest.approx(expected, abs=1e-9)
        assert by_link[link_id]["synchronization_pairs"] == 100
