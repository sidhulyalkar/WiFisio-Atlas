from pathlib import Path

import pytest
from pydantic import ValidationError

from physioatlas.persona import (
    PersonaSignal,
    SleepStageEstimate,
    build_persona_snapshot,
    demo_persona_snapshot,
    write_persona_snapshot,
)


def _signal(confidence: float = 0.9, observability: float = 0.9) -> PersonaSignal:
    return PersonaSignal(
        key="respiration_rate",
        label="respiration",
        value=13.5,
        unit="breaths/min",
        confidence=confidence,
        observability=observability,
        origin="rf",
        evidence="predicted",
        method="test-estimator",
        claim_boundary="Research estimate only.",
    )


def test_persona_abstains_instead_of_exposing_weak_value():
    snapshot = build_persona_snapshot(
        session_id="session-a",
        signals=[_signal(confidence=0.2)],
        mode="live",
        overall_observability=0.8,
    )

    assert snapshot.signals[0].available is False
    assert snapshot.signals[0].value is None
    assert snapshot.signals[0].confidence == pytest.approx(0.2)


def test_persona_identity_requires_explicit_presentation_scope():
    with pytest.raises(ValueError, match="presentation_identity"):
        build_persona_snapshot(
            session_id="session-a",
            signals=[_signal()],
            mode="live",
            overall_observability=0.8,
            subject_alias="member-a",
        )

    snapshot = build_persona_snapshot(
        session_id="session-a",
        signals=[_signal()],
        mode="live",
        overall_observability=0.8,
        subject_alias="member-a",
        consent_scopes=["presentation_identity"],
    )
    assert snapshot.subject_alias == "member-a"
    assert snapshot.privacy.identity_included is True


def test_sleep_stage_probabilities_must_sum_to_one():
    with pytest.raises(ValidationError):
        SleepStageEstimate(
            wake=0.2,
            light=0.2,
            deep=0.2,
            rem=0.2,
            confidence=0.7,
            observability=0.8,
            model_id="bad-test-model",
        )


def test_demo_snapshot_is_explicitly_synthetic_and_raw_free(tmp_path: Path):
    snapshot = demo_persona_snapshot()
    destination = write_persona_snapshot(tmp_path / "persona.json", snapshot)
    payload = destination.read_text(encoding="utf-8")

    assert snapshot.mode == "demo"
    assert snapshot.research_only is True
    assert snapshot.clinical_claim_allowed is False
    assert snapshot.privacy.raw_rf_included is False
    assert snapshot.privacy.raw_camera_included is False
    assert '"schema_version": "physioatlas.persona.v1"' in payload
    assert "raw_csi" not in payload
    assert "camera_frame" not in payload
