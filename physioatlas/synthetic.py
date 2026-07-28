from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .io import SignalStream, dump_json, save_stream
from .physiology import PhysiologyEdge, PhysiologyGraph, PhysiologyNode
from .privacy import ConsentPolicy
from .schema import (
    AnatomyRegion,
    Modality,
    RFLink,
    RFRepresentation,
    Sensor,
    SensorGeometry,
    StreamReference,
    TargetDefinition,
    Vec3,
)


def _sensor(
    sensor_id: str,
    modality: Modality,
    position: Optional[tuple[float, float, float]],
    rate: float,
    *,
    hardware: str = "synthetic",
    clock_domain: str = "reference",
) -> Sensor:
    return Sensor(
        sensor_id=sensor_id,
        modality=modality,
        position_m=Vec3(x=position[0], y=position[1], z=position[2]) if position else None,
        sample_rate_hz=rate,
        hardware=hardware,
        clock_domain=clock_domain,
    )


def _delayed(signal: np.ndarray, samples: int) -> np.ndarray:
    if samples <= 0:
        return signal.copy()
    output = np.empty_like(signal)
    output[:samples] = signal[0]
    output[samples:] = signal[:-samples]
    return output


def create_synthetic_cohort(
    output_dir: Union[str, Path],
    *,
    subjects: int = 4,
    sessions_per_subject: int = 1,
    duration_seconds: float = 30.0,
    sample_rate_hz: float = 20.0,
    seed: int = 7,
) -> list[Path]:
    """Create deterministic multi-modal data for software verification.

    The signals intentionally contain known cardiac, respiratory, clock,
    geometry, observability, and propagation structure. They are not a model of
    real RF tissue interaction and must never be used as evidence of sensing
    accuracy.
    """
    if subjects < 2:
        raise ValueError("Synthetic cohort requires at least two subjects")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    manifests: list[Path] = []

    for subject_index in range(subjects):
        subject_id = f"subject-{subject_index + 1:03d}"
        subject_hr_hz = 0.95 + 0.08 * subject_index
        subject_resp_hz = 0.20 + 0.01 * subject_index
        # Stable, synthetic-only channel texture used to verify household
        # enrollment infrastructure. It is not evidence that real WiFi provides
        # an equally separable biometric signature.
        identity_rng = np.random.default_rng(seed + 10_000 + subject_index)
        subject_channel_signature = identity_rng.uniform(0.05, 0.16, size=(4, 6))
        subject_identity_frequency_hz = 3.1 + 0.42 * subject_index
        for session_index in range(sessions_per_subject):
            session_id = f"{subject_id}-session-{session_index + 1:02d}"
            session_dir = root / subject_id / session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            t = np.arange(0.0, duration_seconds, 1.0 / sample_rate_hz)
            phase = rng.uniform(0, 2 * np.pi)
            respiratory = np.sin(2 * np.pi * subject_resp_hz * t + phase)
            cardiac = np.sin(2 * np.pi * subject_hr_hz * t + phase * 0.25)
            ecg = cardiac + 0.15 * np.sin(4 * np.pi * subject_hr_hz * t)
            ecg += rng.normal(0, 0.035, size=t.size)
            resp = respiratory + rng.normal(0, 0.025, size=t.size)
            ppg_delay_samples = max(1, int(round(0.18 * sample_rate_hz)))
            ppg = 0.85 * _delayed(cardiac, ppg_delay_samples)
            ppg += 0.08 * _delayed(np.sin(4 * np.pi * subject_hr_hz * t), ppg_delay_samples)
            ppg += rng.normal(0, 0.03, size=t.size)
            ultrasound_delay_samples = max(1, int(round(0.10 * sample_rate_hz)))
            diaphragm = 0.012 * _delayed(respiratory, ultrasound_delay_samples)
            diaphragm += rng.normal(0, 0.0004, size=t.size)

            # Four WiFi links, six subcarriers per link, arranged as
            # [all amplitudes by link, all phases by link]. Each link is stored
            # contiguously as amplitude[6], phase[6] so declared link IDs can be
            # split without guessing.
            links = 4
            subcarriers = 6
            per_link: list[np.ndarray] = []
            for link_index in range(links):
                amplitudes = []
                phases = []
                for subcarrier in range(subcarriers):
                    cardiac_weight = rng.normal(0.18 + 0.03 * link_index, 0.04)
                    resp_weight = rng.normal(0.55 + 0.05 * link_index, 0.08)
                    drift = 0.04 * np.sin(
                        2 * np.pi * (0.012 + subcarrier * 0.001) * t
                    )
                    identity_texture = subject_channel_signature[link_index, subcarrier] * np.sin(
                        2 * np.pi * subject_identity_frequency_hz * t + 0.31 * subcarrier + 0.17 * link_index
                    )
                    amplitude = (
                        1.0
                        + cardiac_weight * cardiac
                        + resp_weight * respiratory
                        + identity_texture
                        + drift
                        + rng.normal(0, 0.08, size=t.size)
                    )
                    phase_signal = (
                        0.08 * cardiac
                        + (0.18 + 0.02 * link_index) * respiratory
                        + 0.35 * identity_texture
                        + 0.02 * np.sin(2 * np.pi * 0.03 * t + subcarrier)
                        + rng.normal(0, 0.025, size=t.size)
                    )
                    amplitudes.append(amplitude)
                    phases.append(phase_signal)
                per_link.append(
                    np.concatenate(
                        [np.stack(amplitudes, axis=1), np.stack(phases, axis=1)],
                        axis=1,
                    )
                )
            wifi = np.concatenate(per_link, axis=1).astype(np.float32)
            mmwave = np.stack(
                [
                    respiratory,
                    cardiac,
                    _delayed(cardiac, 1),
                    np.gradient(respiratory),
                ],
                axis=1,
            ).astype(np.float32)
            mmwave += rng.normal(0, 0.05, size=mmwave.shape).astype(np.float32)
            pose = np.stack(
                [
                    0.02 * respiratory,
                    1.25 + 0.03 * respiratory,
                    np.zeros_like(respiratory),
                ],
                axis=1,
            ).astype(np.float32)
            sync = ((np.mod(t, 2.0) < (1.0 / sample_rate_hz))).astype(np.float32)[:, None]

            quality = np.ones(t.size, dtype=np.float32)
            if t.size > 12:
                quality[t.size // 3 : t.size // 3 + 2] = 0.0
            save_stream(session_dir / "wifi_csi.npz", SignalStream(t, wifi, quality))
            save_stream(session_dir / "mmwave.npz", SignalStream(t, mmwave, quality))
            save_stream(session_dir / "ecg.npz", SignalStream(t, ecg[:, None]))
            save_stream(session_dir / "ppg.npz", SignalStream(t, ppg[:, None]))
            save_stream(session_dir / "resp_belt.npz", SignalStream(t, resp[:, None]))
            save_stream(
                session_dir / "ultrasound.npz",
                SignalStream(t, np.stack([diaphragm, np.zeros_like(diaphragm)], axis=1)),
            )
            save_stream(session_dir / "pose.npz", SignalStream(t, pose))
            save_stream(session_dir / "sync.npz", SignalStream(t, sync))

            sensors = [
                _sensor("tx-1", Modality.WIFI_CSI, (-1.5, 0.0, 1.1), sample_rate_hz, hardware="esp32-synthetic"),
                _sensor("tx-2", Modality.WIFI_CSI, (1.5, 0.0, 1.1), sample_rate_hz, hardware="esp32-synthetic"),
                _sensor("rx-1", Modality.WIFI_CSI, (0.0, -1.5, 1.1), sample_rate_hz, hardware="esp32-synthetic"),
                _sensor("rx-2", Modality.WIFI_CSI, (0.0, 1.5, 1.1), sample_rate_hz, hardware="esp32-synthetic"),
                _sensor("mmwave-1", Modality.MMWAVE, (0.0, -2.0, 1.2), sample_rate_hz, hardware="fmcw-synthetic"),
                _sensor("ecg-1", Modality.ECG, None, sample_rate_hz),
                _sensor("ppg-1", Modality.PPG, None, sample_rate_hz),
                _sensor("belt-1", Modality.RESP_BELT, None, sample_rate_hz),
                _sensor("ultrasound-1", Modality.ULTRASOUND, None, sample_rate_hz),
                _sensor("pose-1", Modality.POSE, None, sample_rate_hz),
                _sensor("sync-1", Modality.SYNC, None, sample_rate_hz),
            ]
            rf_links = []
            link_index = 0
            for tx in ("tx-1", "tx-2"):
                for rx in ("rx-1", "rx-2"):
                    link_index += 1
                    rf_links.append(
                        RFLink(
                            link_id=f"link-{link_index}",
                            tx_sensor_id=tx,
                            rx_sensor_id=rx,
                            center_frequency_hz=5.18e9,
                            bandwidth_hz=40e6,
                            n_subcarriers=subcarriers,
                        )
                    )
            geometry = SensorGeometry(
                sensors=sensors,
                rf_links=rf_links,
                anatomy_anchors_m={
                    AnatomyRegion.THORAX: Vec3(x=0.0, y=0.0, z=1.25),
                    AnatomyRegion.CARDIAC_APEX: Vec3(x=-0.12, y=-0.05, z=1.20),
                    AnatomyRegion.CAROTID_LEFT: Vec3(x=-0.08, y=-0.02, z=1.52),
                    AnatomyRegion.RADIAL_LEFT: Vec3(x=-0.45, y=-0.05, z=1.05),
                    AnatomyRegion.ABDOMEN: Vec3(x=0.0, y=0.0, z=0.95),
                    AnatomyRegion.DIAPHRAGM: Vec3(x=0.0, y=0.0, z=1.05),
                },
                calibration_uncertainty_m=0.005,
            )
            dump_json(session_dir / "geometry.json", geometry.model_dump(mode="json"))

            graph = PhysiologyGraph(
                graph_id="synthetic-cardiorespiratory",
                nodes=[
                    PhysiologyNode(node_id="ecg", modality=Modality.ECG, sensor_id="ecg-1", anatomy_region=AnatomyRegion.CARDIAC_APEX),
                    PhysiologyNode(node_id="ppg", modality=Modality.PPG, sensor_id="ppg-1", anatomy_region=AnatomyRegion.RADIAL_LEFT),
                    PhysiologyNode(node_id="resp", modality=Modality.RESP_BELT, sensor_id="belt-1", anatomy_region=AnatomyRegion.THORAX),
                    PhysiologyNode(node_id="diaphragm", modality=Modality.ULTRASOUND, sensor_id="ultrasound-1", anatomy_region=AnatomyRegion.DIAPHRAGM, units="meters"),
                ],
                edges=[
                    PhysiologyEdge(source="ecg", target="ppg", min_delay_s=0.05, max_delay_s=0.4, mechanism="synthetic_pulse_arrival"),
                    PhysiologyEdge(source="resp", target="diaphragm", min_delay_s=0.0, max_delay_s=0.3, mechanism="synthetic_respiratory_motion"),
                ],
            )
            dump_json(session_dir / "physiology_graph.json", graph.model_dump(mode="json"))
            dump_json(
                session_dir / "calibration.json",
                {
                    "schema_version": "physioatlas.calibration-bundle.v1",
                    "reference_clock": "reference",
                    "clocks": {
                        "reference": {
                            "scale": 1.0,
                            "offset_s": 0.0,
                            "drift_ppm": 0.0,
                            "rmse_s": 0.0,
                        }
                    },
                    "geometry_rmse_m": 0.005,
                },
            )
            targets = [
                TargetDefinition(
                    name="ecg_waveform",
                    source_modality=Modality.ECG,
                    anatomy_region=AnatomyRegion.CARDIAC_APEX,
                    units="normalized_voltage",
                    graph_node_id="ecg",
                    description="Synthetic ECG-like waveform for pipeline verification only.",
                ),
                TargetDefinition(
                    name="respiratory_waveform",
                    source_modality=Modality.RESP_BELT,
                    anatomy_region=AnatomyRegion.THORAX,
                    units="normalized_displacement",
                    graph_node_id="resp",
                    description="Synthetic respiratory belt waveform.",
                ),
                TargetDefinition(
                    name="ultrasound_diaphragm_displacement",
                    source_modality=Modality.ULTRASOUND,
                    anatomy_region=AnatomyRegion.DIAPHRAGM,
                    units="meters",
                    graph_node_id="diaphragm",
                    description="Synthetic regional displacement supervision.",
                ),
            ]
            consent = ConsentPolicy(
                consent_id=f"consent-{subject_id}",
                subject_id=subject_id,
                granted_at_utc="2026-01-01T00:00:00+00:00",
                expires_at_utc="2035-01-01T00:00:00+00:00",
                allowed_modalities=[
                    Modality.WIFI_CSI,
                    Modality.MMWAVE,
                    Modality.ECG,
                    Modality.PPG,
                    Modality.RESP_BELT,
                    Modality.ULTRASOUND,
                    Modality.POSE,
                    Modality.SYNC,
                ],
                allowed_targets=[target.name for target in targets],
                allowed_uses=["research", "model_training", "identity_enrollment", "live_tracking", "household_dashboard"],
                allow_model_training=True,
                allow_identity_enrollment=True,
                allow_live_tracking=True,
                allow_household_dashboard=True,
                participant_acknowledged=True,
            )
            dump_json(session_dir / "consent.json", consent.model_dump(mode="json"))

            manifest = {
                "schema_version": "physioatlas.session.v2",
                "session_id": session_id,
                "subject_id": subject_id,
                "started_at_utc": (
                    datetime(2026, 1, 1, tzinfo=timezone.utc)
                    + timedelta(days=subject_index, hours=session_index)
                ).isoformat(),
                "geometry_path": "geometry.json",
                "consent_path": "consent.json",
                "physiology_graph_path": "physiology_graph.json",
                "calibration_path": "calibration.json",
                "streams": [
                    StreamReference(
                        modality=Modality.WIFI_CSI,
                        sensor_id="mesh",
                        path="wifi_csi.npz",
                        representation=RFRepresentation.AMPLITUDE_PHASE,
                        clock_domain="reference",
                        metadata={"link_ids": [f"link-{index}" for index in range(1, 5)]},
                    ).model_dump(mode="json"),
                    StreamReference(
                        modality=Modality.MMWAVE,
                        sensor_id="mmwave-1",
                        path="mmwave.npz",
                        representation=RFRepresentation.FEATURES,
                        clock_domain="reference",
                        metadata={"link_id": "mmwave-beam-1"},
                    ).model_dump(mode="json"),
                    StreamReference(modality=Modality.ECG, sensor_id="ecg-1", path="ecg.npz", clock_domain="reference").model_dump(mode="json"),
                    StreamReference(modality=Modality.PPG, sensor_id="ppg-1", path="ppg.npz", clock_domain="reference").model_dump(mode="json"),
                    StreamReference(modality=Modality.RESP_BELT, sensor_id="belt-1", path="resp_belt.npz", clock_domain="reference").model_dump(mode="json"),
                    StreamReference(modality=Modality.ULTRASOUND, sensor_id="ultrasound-1", path="ultrasound.npz", clock_domain="reference", units="meters").model_dump(mode="json"),
                    StreamReference(modality=Modality.POSE, sensor_id="pose-1", path="pose.npz", clock_domain="reference").model_dump(mode="json"),
                    StreamReference(modality=Modality.SYNC, sensor_id="sync-1", path="sync.npz", clock_domain="reference").model_dump(mode="json"),
                ],
                "targets": [target.model_dump(mode="json") for target in targets],
                "protocol": "synthetic_pipeline_verification",
                "environment_id": f"synthetic-room-{(subject_index + session_index) % 2 + 1}",
                "posture": "seated" if subject_index % 2 == 0 else "supine",
                "condition": "rest" if session_index % 2 == 0 else "paced_breathing",
                "metadata": {
                    "synthetic": True,
                    "generator_seed": seed,
                    "known_ppg_delay_s": ppg_delay_samples / sample_rate_hz,
                    "known_ultrasound_delay_s": ultrasound_delay_samples / sample_rate_hz,
                },
            }
            manifest_path = session_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            manifests.append(manifest_path)
    return manifests
