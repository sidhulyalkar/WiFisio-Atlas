import json
from pathlib import Path

import numpy as np
import pytest

from physioatlas.acquisition import (
    AcquisitionConfig,
    AdapterSpec,
    DEFAULT_REGISTRY,
    run_acquisition,
)
from physioatlas.calibration import (
    apply_clock_calibration,
    estimate_clock_calibration,
    estimate_rigid_transform,
)
from physioatlas.io import SignalStream, load_session
from physioatlas.schema import Modality, RFRepresentation, Sensor, SensorGeometry, Vec3


def test_affine_clock_calibration_rejects_outlier():
    source = np.arange(8.0)
    reference = 1.00012 * source + 0.04
    reference[5] += 0.5
    calibration = estimate_clock_calibration(source, reference)
    assert calibration.drift_ppm == pytest.approx(120.0, abs=1e-5)
    assert calibration.offset_s == pytest.approx(0.04, abs=1e-7)
    assert calibration.inlier_fraction < 1.0
    stream = SignalStream(source, np.ones((source.size, 1)))
    transformed = apply_clock_calibration(stream, calibration)
    np.testing.assert_allclose(transformed.timestamps_s[[0, -1]], [0.04, 7.04084], atol=1e-6)


def test_rigid_geometry_calibration_recovers_transform():
    source = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    rotation = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    translation = np.asarray([0.2, -0.1, 0.5])
    target = source @ rotation.T + translation
    transform = estimate_rigid_transform(source, target)
    np.testing.assert_allclose(transform.apply(source), target, atol=1e-8)
    assert transform.rmse_m < 1e-8


def test_acquisition_registry_and_orchestrator(tmp_path: Path):
    geometry = SensorGeometry(
        sensors=[
            Sensor(
                sensor_id="rf",
                modality=Modality.WIFI_CSI,
                position_m=Vec3(x=0, y=0, z=1),
                sample_rate_hz=20,
                hardware="synthetic",
            )
        ]
    )
    geometry_path = tmp_path / "geometry.json"
    geometry_path.write_text(json.dumps(geometry.model_dump(mode="json")))
    report = run_acquisition(
        AcquisitionConfig(
            session_id="session",
            subject_id="subject",
            protocol="test",
            output_dir=str(tmp_path / "recording"),
            geometry_path=str(geometry_path),
            duration_seconds=1.0,
            adapters=[
                AdapterSpec(
                    adapter_type="synthetic_wave",
                    sensor_id="rf",
                    modality=Modality.WIFI_CSI,
                    output_name="rf.npz",
                    sample_rate_hz=20,
                    representation=RFRepresentation.FEATURES,
                    parameters={"features": 3, "seed": 4},
                )
            ],
        )
    )
    assert report["streams"][0]["samples"] == 20
    session = load_session(tmp_path / "recording" / "manifest.json")
    assert "wifi_csi:rf" in session.streams
    names = {item["adapter_type"] for item in DEFAULT_REGISTRY.describe()}
    assert {"npz_replay", "ruview_jsonl", "synthetic_wave"} <= names


def test_udp_json_adapter_receives_frames() -> None:
    import json
    import socket
    import threading
    import time

    from physioatlas.acquisition import AdapterSpec, UDPJSONAdapter
    from physioatlas.schema import Modality, RFRepresentation

    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = int(probe.getsockname()[1])
    probe.close()

    spec = AdapterSpec(
        adapter_type="udp_json",
        sensor_id="udp-rf",
        modality=Modality.WIFI_CSI,
        output_name="udp.npz",
        sample_rate_hz=50.0,
        representation=RFRepresentation.AMPLITUDE_PHASE,
        parameters={"host": "127.0.0.1", "port": port},
    )
    holder: dict[str, object] = {}

    def receive() -> None:
        holder["stream"] = UDPJSONAdapter(spec).acquire(0.35)

    thread = threading.Thread(target=receive, daemon=True)
    thread.start()
    time.sleep(0.05)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
        for index in range(10):
            payload = {
                "timestamp_s": index * 0.02,
                "amplitude": [1.0 + index, 2.0 + index],
                "phase": [0.1 * index, 0.2 * index],
                "quality": 0.9,
            }
            sender.sendto(json.dumps(payload).encode("utf-8"), ("127.0.0.1", port))
            time.sleep(0.01)
    thread.join(timeout=1.0)
    assert not thread.is_alive()
    stream = holder["stream"]
    assert stream.values.shape == (10, 4)
    assert stream.quality is not None
    assert np.allclose(stream.quality, 0.9)
    assert np.all(np.diff(stream.timestamps_s) > 0)
