from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

STRICT_MODEL = ConfigDict(
    extra="forbid",
    validate_assignment=True,
    allow_inf_nan=False,
)


class MotionUnit(str, Enum):
    DEGREES = "degrees"
    MILLIMETERS = "millimeters"


class TrajectoryKind(str, Enum):
    HOME = "home"
    SWEEP = "sweep"
    DWELL = "dwell"
    SINUSOID = "sinusoid"
    STEP = "step"


class SafetyState(str, Enum):
    SAFE_HOLD = "safe_hold"
    ARMED = "armed"
    RUNNING = "running"
    ESTOPPED = "estopped"
    FAULTED = "faulted"


class ReflectorDeviceIdentity(BaseModel):
    """Hardware-neutral identity; it does not attest physical integration."""

    model_config = STRICT_MODEL

    device_id: str = Field(min_length=1)
    hardware_revision: str = Field(min_length=1)
    firmware_version: str = Field(min_length=1)
    driver_name: str = Field(min_length=1)
    telemetry_clock_id: str = Field(min_length=1)
    serial_number: str | None = None


class ReflectorInstallation(BaseModel):
    """Fixed room pose and the reflector's single commanded motion axis."""

    model_config = STRICT_MODEL

    environment_id: str = Field(min_length=1)
    origin_m: tuple[float, float, float]
    axis_unit_vector: tuple[float, float, float]
    motion_unit: MotionUnit
    home_position: float
    reflector_diameter_m: float = Field(gt=0.0)
    reflector_material: str = Field(min_length=1)

    @model_validator(mode="after")
    def axis_is_unit_length(self) -> ReflectorInstallation:
        norm_sq = sum(value * value for value in self.axis_unit_vector)
        if not 0.999 <= norm_sq <= 1.001:
            raise ValueError("axis_unit_vector must have unit length")
        return self


class ReflectorSafetyLimits(BaseModel):
    """Hard bounds enforced before and during every trajectory."""

    model_config = STRICT_MODEL

    minimum_position: float
    maximum_position: float
    maximum_velocity_per_s: float = Field(gt=0.0)
    maximum_acceleration_per_s2: float = Field(gt=0.0)
    maximum_motor_current_a: float = Field(gt=0.0)
    fail_closed: bool = True
    require_estop_input: bool = True

    @model_validator(mode="after")
    def ordered_positions(self) -> ReflectorSafetyLimits:
        if self.minimum_position >= self.maximum_position:
            raise ValueError("minimum_position must be below maximum_position")
        if not self.fail_closed:
            raise ValueError("calibration reflector must be configured fail-closed")
        return self


class TrajectorySegment(BaseModel):
    """One typed command in a calibration trajectory."""

    model_config = STRICT_MODEL

    kind: TrajectoryKind
    duration_s: float = Field(gt=0.0)
    repeats: int = Field(default=1, ge=1, le=100)
    target_position: float | None = None
    start_position: float | None = None
    end_position: float | None = None
    center_position: float | None = None
    amplitude: float | None = Field(default=None, gt=0.0)
    cycles: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def required_parameters(self) -> TrajectorySegment:
        if self.kind in {TrajectoryKind.DWELL, TrajectoryKind.STEP}:
            if self.target_position is None:
                raise ValueError(f"{self.kind.value} requires target_position")
        elif self.kind == TrajectoryKind.SWEEP:
            if self.start_position is None or self.end_position is None:
                raise ValueError("sweep requires start_position and end_position")
        elif self.kind == TrajectoryKind.SINUSOID:
            if (
                self.center_position is None
                or self.amplitude is None
                or self.cycles is None
            ):
                raise ValueError(
                    "sinusoid requires center_position, amplitude, and cycles"
                )
        return self

    def extrema(self, home_position: float) -> tuple[float, float]:
        if self.kind == TrajectoryKind.HOME:
            return home_position, home_position
        if self.kind in {TrajectoryKind.DWELL, TrajectoryKind.STEP}:
            assert self.target_position is not None
            return self.target_position, self.target_position
        if self.kind == TrajectoryKind.SWEEP:
            assert self.start_position is not None and self.end_position is not None
            return (
                min(self.start_position, self.end_position),
                max(self.start_position, self.end_position),
            )
        assert self.center_position is not None and self.amplitude is not None
        return (
            self.center_position - self.amplitude,
            self.center_position + self.amplitude,
        )


class ReflectorTrajectoryPlan(BaseModel):
    model_config = STRICT_MODEL

    schema_version: str = "physioatlas.reflector-trajectory.v1"
    plan_id: str = Field(min_length=1)
    sample_rate_hz: float = Field(gt=1.0, le=1000.0)
    segments: list[TrajectorySegment] = Field(min_length=1)
    research_only: bool = True
    claim_boundary: str = (
        "Command plan for RF calibration; not evidence of physical motor motion."
    )

    @model_validator(mode="after")
    def preserve_claim_boundary(self) -> ReflectorTrajectoryPlan:
        if not self.research_only:
            raise ValueError("reflector trajectories must remain research-only")
        return self


class ReflectorTelemetrySample(BaseModel):
    """Measured or simulated actuator telemetry in one named clock domain."""

    model_config = STRICT_MODEL

    schema_version: str = "physioatlas.reflector-telemetry.v1"
    sequence: int = Field(ge=0)
    monotonic_timestamp_s: float = Field(ge=0.0)
    clock_id: str = Field(min_length=1)
    plan_id: str = Field(min_length=1)
    segment_index: int = Field(ge=0)
    repeat_index: int = Field(ge=0)
    phase: float = Field(ge=0.0, le=1.0)
    commanded_position: float
    actual_position: float
    actual_velocity_per_s: float
    motor_current_a: float = Field(ge=0.0)
    motor_power_enabled: bool
    minimum_limit_active: bool = False
    maximum_limit_active: bool = False
    estop_active: bool = False
    fault_code: str | None = None
    safety_state: SafetyState
    plan_complete: bool = False
    synthetic: bool
    research_only: bool = True

    @model_validator(mode="after")
    def preserve_provenance(self) -> ReflectorTelemetrySample:
        if not self.research_only:
            raise ValueError("reflector telemetry must remain research-only")
        return self


class ReflectorHealthThresholds(BaseModel):
    model_config = STRICT_MODEL

    maximum_lag_s: float = Field(default=0.5, gt=0.0)
    maximum_rmse: float = Field(default=2.0, gt=0.0)
    maximum_absolute_error: float = Field(default=5.0, gt=0.0)
    maximum_repeatability_rmse: float = Field(default=1.0, gt=0.0)
    minimum_command_range: float = Field(default=1.0, gt=0.0)
    minimum_samples: int = Field(default=20, ge=3)


class ReflectorSyncAnchor(BaseModel):
    model_config = STRICT_MODEL

    sequence: int = Field(ge=0)
    monotonic_timestamp_s: float = Field(ge=0.0)
    event: str = Field(min_length=1)
    commanded_position: float
    actual_position: float


class ReflectorCalibrationHealth(BaseModel):
    """Trajectory health used to accept or reject synchronized RF captures."""

    model_config = STRICT_MODEL

    schema_version: str = "physioatlas.reflector-health.v1"
    plan_id: str = Field(min_length=1)
    clock_id: str = Field(min_length=1)
    sample_count: int = Field(ge=0)
    estimated_lag_s: float | None = Field(default=None, ge=0.0)
    trajectory_rmse: float | None = Field(default=None, ge=0.0)
    maximum_absolute_error: float | None = Field(default=None, ge=0.0)
    repeatability_rmse: float | None = Field(default=None, ge=0.0)
    synchronization_anchors: list[ReflectorSyncAnchor]
    valid_for_rf_calibration: bool
    invalidation_reasons: list[str]
    faults_observed: list[str]
    synthetic: bool
    research_only: bool = True
    clinical_claim_allowed: bool = False
    claim_boundary: str = (
        "Actuator trajectory synchronization health only; not RF sensing accuracy."
    )

    @model_validator(mode="after")
    def preserve_claim_boundary(self) -> ReflectorCalibrationHealth:
        if not self.research_only or self.clinical_claim_allowed:
            raise ValueError("reflector health must remain non-clinical research output")
        if self.valid_for_rf_calibration and self.invalidation_reasons:
            raise ValueError("a valid result cannot contain invalidation reasons")
        return self
