from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExperimentDataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root: str = Field(min_length=1)
    target: str = Field(default="ecg", min_length=1)
    sample_rate_hz: float = Field(default=20.0, gt=0)
    window_seconds: float = Field(default=8.0, gt=0)
    stride_seconds: float = Field(default=4.0, gt=0)
    max_gap_s: float = Field(default=0.25, gt=0)
    require_consent: bool = False

    @model_validator(mode="after")
    def validate_windowing(self) -> "ExperimentDataConfig":
        if self.stride_seconds > self.window_seconds:
            raise ValueError("stride_seconds cannot exceed window_seconds")
        if self.window_seconds * self.sample_rate_hz < 2:
            raise ValueError("window configuration must contain at least two samples")
        return self


class ExperimentSplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal[
        "subject",
        "session",
        "environment",
        "posture",
        "condition",
        "hardware",
    ] = "subject"
    validation_fraction: float = Field(default=0.25, gt=0, lt=1)
    held_out_groups: list[str] = Field(default_factory=list)


class ExperimentTrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epochs: int = Field(default=5, ge=1)
    batch_size: int = Field(default=8, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0)
    hidden_dim: int = Field(default=64, ge=8)
    layers: int = Field(default=2, ge=1)
    seed: int = 42
    validation_fraction: float = Field(default=0.25, gt=0, lt=1)
    device: str = Field(default="auto", min_length=1)
    use_auxiliary: bool = True
    use_geometry: bool = True
    architecture: Literal["compact", "complex_link"] = "complex_link"
    causal: bool = False
    uncertainty_weight: float = Field(default=0.05, ge=0.0)
    observability_weight: float = Field(default=0.1, ge=0.0)


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "physioatlas.config.v2"
    name: str = Field(min_length=1)
    hypothesis: str = Field(min_length=10)
    claim_boundary: str = Field(
        default="Association/prediction only; no anatomical or clinical claim without external validation.",
        min_length=10,
    )
    research_only: Literal[True] = True
    output: Optional[str] = None
    data: ExperimentDataConfig
    split: ExperimentSplitConfig = Field(default_factory=ExperimentSplitConfig)
    training: ExperimentTrainingConfig = Field(default_factory=ExperimentTrainingConfig)

    @model_validator(mode="after")
    def synchronize_legacy_validation_fraction(self) -> "ExperimentConfig":
        # v0.2 stored this setting under training. Preserve that behavior when
        # no explicit split section was supplied.
        if self.split.validation_fraction == 0.25 and self.training.validation_fraction != 0.25:
            self.split.validation_fraction = self.training.validation_fraction
        return self
