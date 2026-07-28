from pathlib import Path

import numpy as np
import torch

from physioatlas.physiology import (
    PhysiologyEdge,
    PhysiologyGraph,
    PhysiologyNode,
    analyze_dataset_propagation,
    propagation_consistency_loss,
)
from physioatlas.schema import AnatomyRegion, Modality
from physioatlas.synthetic import create_synthetic_cohort
from physioatlas.ultrasound import extract_displacement_trace, phase_correlation_shift


def test_ultrasound_phase_correlation_tracks_motion():
    image = np.zeros((32, 32), dtype=np.float32)
    image[8:15, 10:17] = 1.0
    moved = np.roll(image, shift=(3, -2), axis=(0, 1))
    dy, dx = phase_correlation_shift(image, moved)
    assert (dy, dx) == (3.0, -2.0)
    frames = np.stack([np.roll(image, index, axis=0) for index in range(4)])
    trace = extract_displacement_trace(frames, pixel_spacing_m=(0.001, 0.001))
    np.testing.assert_allclose(trace[-1], [0.003, 0.0], atol=1e-7)


def test_synthetic_propagation_graph_recovers_declared_edges(tmp_path: Path):
    create_synthetic_cohort(
        tmp_path / "data", subjects=3, duration_seconds=20.0, sample_rate_hz=20.0, seed=55
    )
    report = analyze_dataset_propagation(tmp_path / "data", sample_rate_hz=20.0)
    assert report["valid"] is True
    delays = [
        edge["delay_s"]
        for session in report["sessions"]
        for edge in session["edges"]
        if edge["source"] == "ecg" and edge["target"] == "ppg"
    ]
    assert all(0.05 <= delay <= 0.4 for delay in delays)


def test_propagation_regularizer_is_finite():
    graph = PhysiologyGraph(
        graph_id="test",
        nodes=[
            PhysiologyNode(node_id="a", modality=Modality.ECG, sensor_id="a", anatomy_region=AnatomyRegion.CARDIAC_APEX),
            PhysiologyNode(node_id="b", modality=Modality.PPG, sensor_id="b", anatomy_region=AnatomyRegion.RADIAL_LEFT),
        ],
        edges=[PhysiologyEdge(source="a", target="b", min_delay_s=0, max_delay_s=1)],
    )
    source = torch.zeros(2, 20, 1)
    target = torch.zeros(2, 20, 1)
    source[:, 5] = 1
    target[:, 8] = 1
    loss = propagation_consistency_loss({"a": source, "b": target}, graph, sample_rate_hz=10)
    assert torch.isfinite(loss)
    assert float(loss) == 0.0
