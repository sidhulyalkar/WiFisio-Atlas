from pathlib import Path

import torch

from physioatlas.dataset import load_windows_from_root
from physioatlas.model import PhysioAtlasModel, masked_waveform_loss
from physioatlas.pretraining import RFPretrainConfig, load_rf_foundation, train_rf_foundation
from physioatlas.synthetic import create_synthetic_cohort


def test_complex_link_model_outputs_uncertainty_and_observability():
    model = PhysioAtlasModel(
        rf_features=30,
        auxiliary_features=2,
        target_features=1,
        geometry_features=37,
        hidden_dim=16,
        layers=1,
        architecture="complex_link",
        linked_rf_channels=10,
    )
    output = model(
        torch.randn(2, 12, 30),
        torch.randn(2, 12, 2),
        torch.randn(2, 37),
        torch.randn(2, 12, 3, 10),
        torch.randn(2, 3, 12),
        torch.ones(2, 3),
        torch.ones(2, 12, 3, 1),
    )
    assert output.waveform.shape == (2, 12, 1)
    assert output.log_variance.shape == (2, 12, 1)
    assert torch.all((output.observability >= 0) & (output.observability <= 1))
    loss = masked_waveform_loss(
        output,
        torch.randn(2, 12, 1),
        torch.ones(2, 12, 1),
        torch.ones(2, 12, 1),
    )
    assert torch.isfinite(loss)


def test_label_free_rf_pretraining_writes_reloadable_checkpoint(tmp_path: Path):
    data = tmp_path / "data"
    create_synthetic_cohort(data, subjects=3, duration_seconds=8, sample_rate_hz=10, seed=13)
    examples = load_windows_from_root(
        data,
        sample_rate_hz=10,
        window_seconds=4,
        stride_seconds=2,
        target_key="ecg",
    )
    result = train_rf_foundation(
        examples,
        tmp_path / "pretrain",
        RFPretrainConfig(epochs=1, batch_size=4, hidden_dim=16, seed=4, device="cpu"),
    )
    assert result["status"] == "completed"
    assert result["label_free"] is True
    checkpoint = tmp_path / "pretrain" / result["checkpoint"]
    model = load_rf_foundation(checkpoint)
    first = examples[0]
    with torch.no_grad():
        reconstruction, embedding = model(
            torch.from_numpy(first.rf_linked[None]),
            torch.from_numpy(first.link_geometry[None]),
            torch.from_numpy(first.link_mask[None]),
            torch.from_numpy(first.rf_observed[None]),
            torch.zeros_like(torch.from_numpy(first.rf_linked[None]), dtype=torch.bool),
        )
    assert reconstruction.shape == torch.from_numpy(first.rf_linked[None]).shape
    assert embedding.shape == (1, 128)
