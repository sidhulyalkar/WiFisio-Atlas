import torch

from physioatlas.model import PhysioAtlasModel, masked_waveform_loss


def test_model_forward_and_loss():
    model = PhysioAtlasModel(
        rf_features=16,
        auxiliary_features=3,
        target_features=1,
        geometry_features=37,
        hidden_dim=32,
        layers=1,
    )
    output = model(
        torch.randn(2, 20, 16),
        torch.randn(2, 20, 3),
        torch.randn(2, 37),
    )
    assert output.waveform.shape == (2, 20, 1)
    assert output.confidence.shape == (2, 20, 1)
    loss = masked_waveform_loss(
        output,
        torch.randn(2, 20, 1),
        torch.ones(2, 20, 1),
    )
    assert torch.isfinite(loss)
    loss.backward()
