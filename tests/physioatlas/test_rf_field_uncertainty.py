from __future__ import annotations

import numpy as np

from physioatlas.rf_field import (
    ActiveSensingCandidate,
    canonical_to_complex,
    delay_doppler_map,
    delay_spread_features,
    reconstruct_cir,
    select_active_measurement,
)
from physioatlas.schema import RFRepresentation
from physioatlas.uncertainty import (
    conformal_interval,
    fit_split_conformal,
    interval_coverage,
    should_abstain as abstain_decision,
)


def test_complex_rf_field_and_active_selection() -> None:
    time = np.arange(16, dtype=float)[:, None]
    phase = 2 * np.pi * time * 0.1 + np.linspace(0.0, 0.4, 8)[None, :]
    amplitude = np.ones_like(phase)
    canonical = np.concatenate([amplitude, phase], axis=1)
    complex_csi = canonical_to_complex(canonical, RFRepresentation.AMPLITUDE_PHASE)
    assert complex_csi.shape == (16, 8)
    cir = reconstruct_cir(complex_csi, n_delay_bins=16)
    features = delay_spread_features(cir, delay_resolution_s=1e-9)
    assert features["dominant_tap_ratio"] > 0
    delay_doppler = delay_doppler_map(complex_csi, n_delay_bins=16, n_doppler_bins=32)
    assert delay_doppler.shape == (32, 16)
    assert np.isfinite(delay_doppler).all()

    decision = select_active_measurement(
        [
            ActiveSensingCandidate("link-a", 0.4, 0.2, novelty=0.1, energy_cost=1.0),
            ActiveSensingCandidate("link-b", 0.6, 0.3, novelty=0.2, energy_cost=0.5),
        ]
    )
    assert decision.candidate_id == "link-b"
    assert len(decision.ranked) == 2


def test_split_conformal_and_abstention() -> None:
    truth = np.linspace(-1.0, 1.0, 40)
    prediction = truth + 0.1 * np.sin(np.arange(40))
    calibration = fit_split_conformal(truth[:20], prediction[:20], alpha=0.1)
    lower, upper = conformal_interval(prediction[20:], calibration)
    coverage = interval_coverage(truth[20:], lower, upper)
    assert 0.0 <= coverage <= 1.0
    assert calibration.calibration_samples == 20
    assert not abstain_decision(
        observability=0.9,
        interval_width=0.2,
        max_interval_width=0.5,
        out_of_distribution_score=0.1,
    )
    assert abstain_decision(
        observability=0.2,
        interval_width=0.2,
        max_interval_width=0.5,
        out_of_distribution_score=0.1,
    )
