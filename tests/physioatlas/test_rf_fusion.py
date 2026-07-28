import numpy as np
import pytest

from physioatlas.household_bfi_schema import (
    BfiFeatureProvenance,
    NormalizedBfiFeature,
)
from physioatlas.rf_fusion import (
    bfi_feature_to_zone_evidence,
    bfi_feature_vector,
    feature_to_zone_evidence,
    fit_zone_prototype_model,
    fuse_zone_evidence,
)
from physioatlas.rf_fusion_schema import (
    CsiBfiFusionConfig,
    FusionZone,
    ZoneEvidence,
)


def _config() -> CsiBfiFusionConfig:
    return CsiBfiFusionConfig(
        zones=[
            FusionZone(zone_id="desk", center_m=[1.0, 1.0]),
            FusionZone(zone_id="sofa", center_m=[4.0, 2.0]),
        ],
        minimum_observability=0.4,
    )


def _evidence(
    source: str,
    probabilities: dict[str, float],
    *,
    timestamp: float = 10.0,
    sync: float = 0.005,
) -> ZoneEvidence:
    return ZoneEvidence(
        evidence_id=f"{source}-{timestamp}",
        timestamp_s=timestamp,
        source=source,
        source_track_id="target-1",
        feature_version=f"{source}.v1",
        zone_probabilities=probabilities,
        observability=0.9,
        signal_quality=0.9,
        synchronization_uncertainty_s=sync,
        respiratory_rate_bpm=12.0 if source == "wifi_csi" else None,
    )


def test_agreeing_csi_bfi_evidence_fuses_and_preserves_csi_respiration():
    result = fuse_zone_evidence(
        [
            _evidence("wifi_csi", {"desk": 0.92, "sofa": 0.08}),
            _evidence("wifi_bfi", {"desk": 0.88, "sofa": 0.12}, timestamp=10.02),
        ],
        _config(),
    )
    assert result.status == "fused"
    assert result.active_zones == ["desk"]
    assert result.cross_modal_conflict < 0.1
    assert result.household_observations[0]["respiratory_rate_bpm"] == 12.0


def test_cross_modal_conflict_and_clock_uncertainty_abstain():
    conflict = fuse_zone_evidence(
        [
            _evidence("wifi_csi", {"desk": 0.95, "sofa": 0.05}),
            _evidence("wifi_bfi", {"desk": 0.05, "sofa": 0.95}),
        ],
        _config(),
    )
    assert conflict.status == "abstained"
    assert conflict.reason == "cross_modal_conflict"
    unsynchronized = fuse_zone_evidence(
        [
            _evidence("wifi_csi", {"desk": 0.9, "sofa": 0.1}),
            _evidence("wifi_bfi", {"desk": 0.9, "sofa": 0.1}, sync=0.2),
        ],
        _config(),
    )
    assert unsynchronized.reason == "synchronization_unverified"


def test_feature_prototypes_produce_deterministic_zone_evidence():
    samples = []
    for zone, center in (("desk", np.array([1.0, 0.0, 0.0])), ("sofa", np.array([0.0, 1.0, 0.0]))):
        for offset in (0.0, 0.01, -0.01):
            samples.append(
                {
                    "source": "wifi_bfi",
                    "feature_version": "bfi.test.v1",
                    "zone_id": zone,
                    "feature": (center + np.array([offset, -offset, 0.0])).tolist(),
                }
            )
    model = fit_zone_prototype_model(
        samples, source="wifi_bfi", feature_version="bfi.test.v1"
    )
    evidence = feature_to_zone_evidence(
        [1.0, 0.01, 0.0],
        model,
        evidence_id="bfi-1",
        timestamp_s=1.0,
        observability=0.9,
        signal_quality=0.8,
        synchronization_uncertainty_s=0.001,
    )
    assert evidence.zone_probabilities["desk"] > 0.95


def test_unconfigured_zone_fails_closed():
    with pytest.raises(ValueError, match="unconfigured zones"):
        fuse_zone_evidence(
            [
                _evidence("wifi_csi", {"desk": 0.9, "garage": 0.1}),
                _evidence("wifi_bfi", {"desk": 0.9, "sofa": 0.1}),
            ],
            _config(),
        )


def test_normalized_bfi_feature_adapts_to_versioned_zone_model():
    feature = NormalizedBfiFeature(
        timestamp_s=2.0,
        stream_id="ap::client",
        link_id="ap",
        client_id="client",
        frequency_band="5ghz",
        channel=36,
        normalized_features=[1.0, 0.0, 0.0, 0.0],
        temporal_delta=[0.1, 0.0, 0.0, 0.0],
        perturbation_energy=1.0,
        motion_energy=0.1,
        quality=0.9,
        observability=0.9,
        status="observable",
        provenance=BfiFeatureProvenance(
            source_capture="capture.jsonl",
            source_sha256="a" * 64,
            calibration_sha256="b" * 64,
        ),
    )
    vector = bfi_feature_vector(feature)
    samples = []
    for zone, center in (
        ("desk", vector),
        ("sofa", np.roll(vector, 1)),
    ):
        for scale in (0.99, 1.0, 1.01):
            samples.append(
                {
                    "source": "wifi_bfi",
                    "feature_version": "physioatlas.household-bfi-zone-feature.v1",
                    "zone_id": zone,
                    "feature": (center * scale).tolist(),
                }
            )
    model = fit_zone_prototype_model(
        samples,
        source="wifi_bfi",
        feature_version="physioatlas.household-bfi-zone-feature.v1",
    )
    evidence = bfi_feature_to_zone_evidence(
        feature,
        model,
        evidence_id="bfi-feature-1",
        synchronization_uncertainty_s=0.002,
    )
    assert evidence.zone_probabilities["desk"] > evidence.zone_probabilities["sofa"]
    assert evidence.provenance["identity_claim"] is False
