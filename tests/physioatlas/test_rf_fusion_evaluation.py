from physioatlas.rf_fusion_evaluation import run_fusion_ablation
from physioatlas.rf_fusion_schema import (
    CsiBfiFusionConfig,
    FusionTruthWindow,
    FusionZone,
    ZoneEvidence,
)


def _evidence(source: str, timestamp: float, desk: float) -> ZoneEvidence:
    return ZoneEvidence(
        evidence_id=f"{source}-{timestamp}",
        timestamp_s=timestamp,
        source=source,
        feature_version=f"{source}.v1",
        zone_probabilities={"desk": desk, "sofa": 1.0 - desk},
        observability=0.9,
        signal_quality=0.9,
        synchronization_uncertainty_s=0.005,
    )


def test_ablation_reports_coverage_and_selective_accuracy():
    config = CsiBfiFusionConfig(
        zones=[
            FusionZone(zone_id="desk", center_m=[1.0, 1.0]),
            FusionZone(zone_id="sofa", center_m=[4.0, 2.0]),
        ],
        minimum_observability=0.4,
    )
    windows = []
    scenarios = [
        (1.0, 0.92, 0.88, ["desk"]),
        (2.0, 0.08, 0.12, ["sofa"]),
        (3.0, 0.90, 0.10, ["desk"]),
    ]
    for timestamp, csi_desk, bfi_desk, zones in scenarios:
        windows.append(
            (
                [
                    _evidence("wifi_csi", timestamp, csi_desk),
                    _evidence("wifi_bfi", timestamp, bfi_desk),
                ],
                FusionTruthWindow(
                    window_id=f"window-{timestamp}",
                    timestamp_s=timestamp,
                    active_zones=zones,
                    reference_kind="synthetic",
                    reference_id="synthetic-test",
                    consent_verified=False,
                ),
            )
        )
    report = run_fusion_ablation(windows, config)
    assert set(report.modes) == {"csi_only", "bfi_only", "csi_bfi_fusion"}
    assert report.modes["csi_only"].coverage == 1.0
    assert report.modes["csi_bfi_fusion"].coverage == 2 / 3
    assert report.modes["csi_bfi_fusion"].selective_exact_accuracy == 1.0


def test_physical_truth_without_consent_is_rejected():
    try:
        FusionTruthWindow(
            window_id="physical",
            timestamp_s=1.0,
            active_zones=["desk"],
            reference_kind="camera",
            reference_id="camera-1",
            consent_verified=False,
        )
    except ValueError as exc:
        assert "consent" in str(exc)
    else:
        raise AssertionError("physical truth without consent should fail")
