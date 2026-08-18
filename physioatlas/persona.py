from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


SignalOrigin = Literal[
    "rf",
    "reference",
    "camera",
    "self_report",
    "derived",
    "demo",
]
EvidenceLevel = Literal["measured", "derived", "predicted", "self_report", "demo"]
PersonaMode = Literal["demo", "replay", "live"]


class PersonaSignal(BaseModel):
    """A presentation-safe scalar or categorical signal with explicit evidence."""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1)
    label: str = Field(min_length=1)
    value: Optional[Union[float, str, bool]] = None
    unit: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    observability: float = Field(ge=0.0, le=1.0)
    origin: SignalOrigin
    evidence: EvidenceLevel
    method: str = Field(min_length=1)
    reference_verified: bool = False
    age_ms: int = Field(default=0, ge=0)
    available: bool = True
    claim_boundary: str = Field(min_length=1)


class SleepStageEstimate(BaseModel):
    """Four-class research sleep estimate suitable for visualisation, not diagnosis."""

    model_config = ConfigDict(extra="forbid")

    wake: float = Field(ge=0.0, le=1.0)
    light: float = Field(ge=0.0, le=1.0)
    deep: float = Field(ge=0.0, le=1.0)
    rem: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    observability: float = Field(ge=0.0, le=1.0)
    model_id: str = Field(min_length=1)
    reference_status: Literal[
        "unvalidated",
        "development-reference",
        "held-out-psg",
        "demo",
    ] = "unvalidated"
    claim_boundary: str = (
        "Research sleep-stage estimate only; not a clinical sleep assessment."
    )

    @model_validator(mode="after")
    def probabilities_sum_to_one(self) -> "SleepStageEstimate":
        total = self.wake + self.light + self.deep + self.rem
        if abs(total - 1.0) > 1e-4:
            raise ValueError("sleep-stage probabilities must sum to 1")
        return self


class PersonaPrivacy(BaseModel):
    """Privacy flags are deliberately part of the transport contract."""

    model_config = ConfigDict(extra="forbid")

    raw_rf_included: Literal[False] = False
    raw_camera_included: Literal[False] = False
    biometric_template_included: Literal[False] = False
    identity_included: bool = False
    local_processing_preferred: bool = True
    consent_scopes: list[str] = Field(default_factory=list)


class PersonaSnapshot(BaseModel):
    """Versioned, presentation-safe state shared with external visual clients."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["physioatlas.persona.v1"] = "physioatlas.persona.v1"
    generated_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    research_only: Literal[True] = True
    clinical_claim_allowed: Literal[False] = False
    mode: PersonaMode
    session_id: str = Field(min_length=1)
    subject_alias: Optional[str] = None
    signals: list[PersonaSignal] = Field(default_factory=list)
    sleep: Optional[SleepStageEstimate] = None
    overall_observability: float = Field(ge=0.0, le=1.0)
    privacy: PersonaPrivacy
    notes: list[str] = Field(default_factory=list)


def build_persona_snapshot(
    *,
    session_id: str,
    signals: Sequence[PersonaSignal],
    mode: PersonaMode,
    overall_observability: float,
    consent_scopes: Sequence[str] = (),
    subject_alias: Optional[str] = None,
    sleep: Optional[SleepStageEstimate] = None,
    minimum_confidence: float = 0.35,
    minimum_observability: float = 0.35,
    notes: Sequence[str] = (),
) -> PersonaSnapshot:
    """Build a fail-closed snapshot for a visual client.

    Low-confidence or poorly observable values keep their provenance but have their
    value removed. Identity is only transported when the explicit
    ``presentation_identity`` scope is present. Raw RF, camera frames, and biometric
    templates are never represented by this contract.
    """

    if not 0.0 <= overall_observability <= 1.0:
        raise ValueError("overall_observability must be in [0, 1]")
    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("minimum_confidence must be in [0, 1]")
    if not 0.0 <= minimum_observability <= 1.0:
        raise ValueError("minimum_observability must be in [0, 1]")

    scopes = sorted(set(str(scope) for scope in consent_scopes))
    identity_allowed = "presentation_identity" in scopes
    if subject_alias is not None and not identity_allowed:
        raise ValueError(
            "subject_alias requires the explicit presentation_identity consent scope"
        )

    filtered_signals: list[PersonaSignal] = []
    for signal in signals:
        visible = (
            signal.available
            and signal.confidence >= minimum_confidence
            and signal.observability >= minimum_observability
        )
        if visible:
            filtered_signals.append(signal)
        else:
            filtered_signals.append(
                signal.model_copy(update={"available": False, "value": None})
            )

    filtered_sleep = sleep
    if sleep is not None and (
        sleep.confidence < minimum_confidence
        or sleep.observability < minimum_observability
    ):
        filtered_sleep = None

    return PersonaSnapshot(
        mode=mode,
        session_id=session_id,
        subject_alias=subject_alias,
        signals=filtered_signals,
        sleep=filtered_sleep,
        overall_observability=overall_observability,
        privacy=PersonaPrivacy(
            identity_included=identity_allowed and subject_alias is not None,
            consent_scopes=scopes,
        ),
        notes=list(notes),
    )


def write_persona_snapshot(path: Union[str, Path], snapshot: PersonaSnapshot) -> Path:
    """Write deterministic JSON for replay, website demos, or a secure local bridge."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            snapshot.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def demo_persona_snapshot() -> PersonaSnapshot:
    """Return synthetic data that exercises the public visualisation contract."""

    signals = [
        PersonaSignal(
            key="respiration_rate",
            label="respiration",
            value=14.2,
            unit="breaths/min",
            confidence=0.88,
            observability=0.92,
            origin="demo",
            evidence="demo",
            method="synthetic-periodic-respiration",
            reference_verified=False,
            claim_boundary="Synthetic visualisation value; no physiological claim.",
        ),
        PersonaSignal(
            key="movement_intensity",
            label="movement",
            value=0.24,
            unit="normalized",
            confidence=0.91,
            observability=0.95,
            origin="demo",
            evidence="demo",
            method="synthetic-motion-envelope",
            reference_verified=False,
            claim_boundary="Synthetic visualisation value; no activity claim.",
        ),
        PersonaSignal(
            key="cardiac_rate",
            label="cardiac rate",
            value=63.0,
            unit="beats/min",
            confidence=0.72,
            observability=0.68,
            origin="demo",
            evidence="demo",
            method="synthetic-cardiac-oscillator",
            reference_verified=False,
            claim_boundary="Synthetic visualisation value; no cardiac claim.",
        ),
    ]
    return build_persona_snapshot(
        session_id="demo-session",
        signals=signals,
        mode="demo",
        overall_observability=0.86,
        sleep=SleepStageEstimate(
            wake=0.08,
            light=0.57,
            deep=0.22,
            rem=0.13,
            confidence=0.71,
            observability=0.83,
            model_id="synthetic-demo",
            reference_status="demo",
        ),
        notes=[
            "All values in this snapshot are synthetic.",
            "The presentation contract never includes raw RF or camera frames.",
        ],
    )
