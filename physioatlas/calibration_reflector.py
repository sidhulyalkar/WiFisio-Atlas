from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .calibration_reflector_schema import (
    ReflectorCalibrationHealth,
    ReflectorDeviceIdentity,
    ReflectorHealthThresholds,
    ReflectorInstallation,
    ReflectorSafetyLimits,
    ReflectorSyncAnchor,
    ReflectorTelemetrySample,
    ReflectorTrajectoryPlan,
    SafetyState,
    TrajectoryKind,
    TrajectorySegment,
)


class ReflectorSafetyError(RuntimeError):
    """Raised before motion when a plan or driver state is unsafe."""


class ReflectorDriver(Protocol):
    @property
    def safety_state(self) -> SafetyState: ...

    def arm(self) -> None: ...

    def safe_stop(self) -> None: ...

    def execute(
        self, plan: ReflectorTrajectoryPlan
    ) -> list[ReflectorTelemetrySample]: ...


@dataclass(frozen=True)
class SimulatedReflectorDynamics:
    """Explicitly synthetic response model used only for software verification."""

    response_lag_s: float = 0.12
    tracking_bias: float = 0.0
    noise_standard_deviation: float = 0.0
    idle_current_a: float = 0.08
    error_current_gain: float = 0.015
    fault_at_s: float | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.response_lag_s < 0.0:
            raise ValueError("response_lag_s cannot be negative")
        if self.noise_standard_deviation < 0.0:
            raise ValueError("noise_standard_deviation cannot be negative")
        if self.idle_current_a < 0.0 or self.error_current_gain < 0.0:
            raise ValueError("current model values cannot be negative")


def validate_trajectory_safety(
    plan: ReflectorTrajectoryPlan,
    installation: ReflectorInstallation,
    limits: ReflectorSafetyLimits,
) -> None:
    """Reject an entire plan before execution if any command is unsafe."""

    if not limits.minimum_position <= installation.home_position <= limits.maximum_position:
        raise ReflectorSafetyError("home position is outside safety limits")
    previous = installation.home_position
    for index, segment in enumerate(plan.segments):
        low, high = segment.extrema(installation.home_position)
        if low < limits.minimum_position or high > limits.maximum_position:
            raise ReflectorSafetyError(
                f"segment {index} exceeds configured position limits"
            )
        start, end = _segment_endpoints(segment, installation.home_position, previous)
        maximum_speed = _required_speed(segment, start, end)
        if maximum_speed > limits.maximum_velocity_per_s:
            raise ReflectorSafetyError(
                f"segment {index} exceeds configured velocity limit"
            )
        maximum_acceleration = _required_acceleration(segment, start, end)
        if maximum_acceleration > limits.maximum_acceleration_per_s2:
            raise ReflectorSafetyError(
                f"segment {index} exceeds configured acceleration limit"
            )
        previous = end


def _segment_endpoints(
    segment: TrajectorySegment, home: float, previous: float
) -> tuple[float, float]:
    if segment.kind == TrajectoryKind.HOME:
        return previous, home
    if segment.kind in {TrajectoryKind.DWELL, TrajectoryKind.STEP}:
        assert segment.target_position is not None
        return previous, segment.target_position
    if segment.kind == TrajectoryKind.SWEEP:
        assert segment.start_position is not None and segment.end_position is not None
        return segment.start_position, segment.end_position
    assert segment.center_position is not None
    return segment.center_position, segment.center_position


def _required_speed(segment: TrajectorySegment, start: float, end: float) -> float:
    if segment.kind == TrajectoryKind.SINUSOID:
        assert segment.amplitude is not None and segment.cycles is not None
        return (
            2.0
            * np.pi
            * segment.amplitude
            * segment.cycles
            / segment.duration_s
        )
    if segment.kind == TrajectoryKind.DWELL:
        return 0.0
    if segment.kind in {TrajectoryKind.HOME, TrajectoryKind.STEP}:
        return 1.5 * abs(end - start) / segment.duration_s
    return abs(end - start) / segment.duration_s


def _required_acceleration(
    segment: TrajectorySegment, start: float, end: float
) -> float:
    if segment.kind == TrajectoryKind.SINUSOID:
        assert segment.amplitude is not None and segment.cycles is not None
        angular_rate = 2.0 * np.pi * segment.cycles / segment.duration_s
        return angular_rate * angular_rate * segment.amplitude
    if segment.kind in {TrajectoryKind.HOME, TrajectoryKind.STEP}:
        return 6.0 * abs(end - start) / (segment.duration_s * segment.duration_s)
    return 0.0


def _commanded_position(
    segment: TrajectorySegment, phase: np.ndarray, home: float, previous: float
) -> np.ndarray:
    if segment.kind == TrajectoryKind.HOME:
        smooth_phase = phase * phase * (3.0 - 2.0 * phase)
        return previous + smooth_phase * (home - previous)
    if segment.kind == TrajectoryKind.DWELL:
        assert segment.target_position is not None
        return np.full_like(phase, segment.target_position)
    if segment.kind == TrajectoryKind.STEP:
        assert segment.target_position is not None
        smooth_phase = phase * phase * (3.0 - 2.0 * phase)
        return previous + smooth_phase * (segment.target_position - previous)
    if segment.kind == TrajectoryKind.SWEEP:
        assert segment.start_position is not None and segment.end_position is not None
        return segment.start_position + phase * (
            segment.end_position - segment.start_position
        )
    assert (
        segment.center_position is not None
        and segment.amplitude is not None
        and segment.cycles is not None
    )
    return segment.center_position + segment.amplitude * np.sin(
        2.0 * np.pi * segment.cycles * phase
    )


class DeterministicSimulatedReflector:
    """Deterministic fake driver; never claims to have actuated hardware."""

    def __init__(
        self,
        identity: ReflectorDeviceIdentity,
        installation: ReflectorInstallation,
        limits: ReflectorSafetyLimits,
        dynamics: SimulatedReflectorDynamics = SimulatedReflectorDynamics(),
    ) -> None:
        self.identity = identity
        self.installation = installation
        self.limits = limits
        self.dynamics = dynamics
        self._safety_state = SafetyState.SAFE_HOLD
        self._estop_active = False
        self._fault_code: str | None = None

    @property
    def safety_state(self) -> SafetyState:
        return self._safety_state

    def arm(self) -> None:
        if self._estop_active or self._fault_code is not None:
            raise ReflectorSafetyError("cannot arm while e-stop or fault is active")
        self._safety_state = SafetyState.ARMED

    def safe_stop(self) -> None:
        if self._safety_state not in {SafetyState.ESTOPPED, SafetyState.FAULTED}:
            self._safety_state = SafetyState.SAFE_HOLD

    def engage_estop(self) -> None:
        self._estop_active = True
        self._safety_state = SafetyState.ESTOPPED

    def clear_estop(self) -> None:
        self._estop_active = False
        self._fault_code = None
        self._safety_state = SafetyState.SAFE_HOLD

    def execute(
        self, plan: ReflectorTrajectoryPlan
    ) -> list[ReflectorTelemetrySample]:
        if self._safety_state != SafetyState.ARMED:
            raise ReflectorSafetyError("driver must be armed before execution")
        validate_trajectory_safety(plan, self.installation, self.limits)
        command, metadata = self._compile_plan(plan)
        dt = 1.0 / plan.sample_rate_hz
        timestamps = np.arange(command.size, dtype=np.float64) * dt
        delayed = np.interp(
            timestamps - self.dynamics.response_lag_s,
            timestamps,
            command,
            left=self.installation.home_position,
        )
        rng = np.random.default_rng(self.dynamics.seed)
        noise = rng.normal(
            0.0, self.dynamics.noise_standard_deviation, command.size
        )
        actual = delayed + self.dynamics.tracking_bias + noise
        velocity = np.gradient(actual, dt) if command.size > 1 else np.zeros(1)
        current = self.dynamics.idle_current_a + self.dynamics.error_current_gain * np.abs(
            command - actual
        )
        self._safety_state = SafetyState.RUNNING
        samples: list[ReflectorTelemetrySample] = []
        for sequence, timestamp in enumerate(timestamps):
            fault = self._sample_fault(float(timestamp), float(current[sequence]))
            state = SafetyState.FAULTED if fault else SafetyState.RUNNING
            segment_index, repeat_index, phase = metadata[sequence]
            motor_power_enabled = (
                plan.segments[segment_index].kind != TrajectoryKind.DWELL
            )
            samples.append(
                ReflectorTelemetrySample(
                    sequence=sequence,
                    monotonic_timestamp_s=float(timestamp),
                    clock_id=self.identity.telemetry_clock_id,
                    plan_id=plan.plan_id,
                    segment_index=segment_index,
                    repeat_index=repeat_index,
                    phase=phase,
                    commanded_position=float(command[sequence]),
                    actual_position=float(actual[sequence]),
                    actual_velocity_per_s=float(velocity[sequence]),
                    motor_current_a=float(current[sequence]),
                    motor_power_enabled=motor_power_enabled,
                    minimum_limit_active=actual[sequence] <= self.limits.minimum_position,
                    maximum_limit_active=actual[sequence] >= self.limits.maximum_position,
                    estop_active=self._estop_active,
                    fault_code=fault,
                    safety_state=state,
                    plan_complete=sequence == command.size - 1 and fault is None,
                    synthetic=True,
                )
            )
            if fault:
                self._fault_code = fault
                self._safety_state = SafetyState.FAULTED
                break
        if self._safety_state == SafetyState.RUNNING:
            self._safety_state = SafetyState.SAFE_HOLD
        return samples

    def _sample_fault(self, timestamp: float, current: float) -> str | None:
        if self._estop_active:
            return "emergency_stop"
        if current > self.limits.maximum_motor_current_a:
            return "over_current"
        if self.dynamics.fault_at_s is not None and timestamp >= self.dynamics.fault_at_s:
            return "simulated_fault"
        return None

    def _compile_plan(
        self, plan: ReflectorTrajectoryPlan
    ) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
        chunks: list[np.ndarray] = []
        metadata: list[tuple[int, int, float]] = []
        previous = self.installation.home_position
        for segment_index, segment in enumerate(plan.segments):
            count = max(2, int(round(segment.duration_s * plan.sample_rate_hz)))
            phase = np.linspace(0.0, 1.0, count, endpoint=False)
            for repeat_index in range(segment.repeats):
                values = _commanded_position(
                    segment, phase, self.installation.home_position, previous
                )
                chunks.append(values)
                metadata.extend(
                    (segment_index, repeat_index, float(value)) for value in phase
                )
                previous = float(values[-1])
        return np.concatenate(chunks), metadata


def analyze_reflector_telemetry(
    samples: Sequence[ReflectorTelemetrySample],
    thresholds: ReflectorHealthThresholds = ReflectorHealthThresholds(),
) -> ReflectorCalibrationHealth:
    """Analyze actuator behavior before an RF capture is accepted."""

    if not samples:
        raise ValueError("reflector telemetry cannot be empty")
    plan_ids = {sample.plan_id for sample in samples}
    clock_ids = {sample.clock_id for sample in samples}
    if len(plan_ids) != 1 or len(clock_ids) != 1:
        raise ValueError("telemetry must contain one plan and one clock domain")
    timestamps = np.asarray([item.monotonic_timestamp_s for item in samples])
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("telemetry timestamps must be strictly monotonic")
    command = np.asarray([item.commanded_position for item in samples])
    actual = np.asarray([item.actual_position for item in samples])
    lag = _estimate_lag(timestamps, command, actual)
    aligned = np.interp(timestamps - lag, timestamps, command, left=command[0])
    error = actual - aligned
    rmse = float(np.sqrt(np.mean(error * error)))
    maximum_error = float(np.max(np.abs(error)))
    repeatability = _repeatability_rmse(samples, lag)
    faults = sorted(
        {
            item.fault_code
            for item in samples
            if item.fault_code is not None
        }
    )
    reasons: list[str] = []
    if len(samples) < thresholds.minimum_samples:
        reasons.append("insufficient_samples")
    if float(np.ptp(command)) < thresholds.minimum_command_range:
        reasons.append("insufficient_trajectory_excitation")
    if lag > thresholds.maximum_lag_s:
        reasons.append("excessive_lag")
    if rmse > thresholds.maximum_rmse:
        reasons.append("trajectory_rmse_exceeded")
    if maximum_error > thresholds.maximum_absolute_error:
        reasons.append("maximum_error_exceeded")
    if repeatability is not None and repeatability > thresholds.maximum_repeatability_rmse:
        reasons.append("repeatability_exceeded")
    if faults:
        reasons.append("actuator_fault")
    if samples[-1].safety_state in {SafetyState.FAULTED, SafetyState.ESTOPPED}:
        if "actuator_fault" not in reasons:
            reasons.append("unsafe_terminal_state")
    if not samples[-1].plan_complete:
        reasons.append("incomplete_trajectory")
    return ReflectorCalibrationHealth(
        plan_id=next(iter(plan_ids)),
        clock_id=next(iter(clock_ids)),
        sample_count=len(samples),
        estimated_lag_s=lag,
        trajectory_rmse=rmse,
        maximum_absolute_error=maximum_error,
        repeatability_rmse=repeatability,
        synchronization_anchors=_sync_anchors(samples),
        valid_for_rf_calibration=not reasons,
        invalidation_reasons=reasons,
        faults_observed=faults,
        synthetic=any(item.synthetic for item in samples),
    )


def _estimate_lag(
    timestamps: np.ndarray, command: np.ndarray, actual: np.ndarray
) -> float:
    dt = float(np.median(np.diff(timestamps)))
    maximum = min(2.0, float(timestamps[-1] - timestamps[0]) / 3.0)
    candidates = np.arange(0.0, maximum + dt / 2.0, dt)
    scores = []
    for candidate in candidates:
        aligned = np.interp(
            timestamps - candidate, timestamps, command, left=command[0]
        )
        scores.append(float(np.mean((actual - aligned) ** 2)))
    return float(candidates[int(np.argmin(scores))])


def _repeatability_rmse(
    samples: Sequence[ReflectorTelemetrySample],
    lag_s: float,
) -> float | None:
    groups: dict[tuple[int, int], list[float]] = {}
    repeat_counts: dict[int, set[int]] = {}
    timestamps = np.asarray([sample.monotonic_timestamp_s for sample in samples])
    actual = np.asarray([sample.actual_position for sample in samples])
    aligned_actual = np.interp(
        timestamps + lag_s,
        timestamps,
        actual,
        left=np.nan,
        right=np.nan,
    )
    for sample, aligned_value in zip(samples, aligned_actual):
        if not np.isfinite(aligned_value):
            continue
        phase_bin = int(round(sample.phase * 1_000_000.0))
        groups.setdefault((sample.segment_index, phase_bin), []).append(
            float(aligned_value)
        )
        repeat_counts.setdefault(sample.segment_index, set()).add(sample.repeat_index)
    deviations: list[float] = []
    for (segment_index, _), values in groups.items():
        if len(repeat_counts[segment_index]) > 1 and len(values) > 1:
            center = float(np.mean(values))
            deviations.extend((value - center) ** 2 for value in values)
    if not deviations:
        return None
    return float(np.sqrt(np.mean(deviations)))


def _sync_anchors(
    samples: Sequence[ReflectorTelemetrySample],
) -> list[ReflectorSyncAnchor]:
    selected = {0, len(samples) - 1}
    for index in range(1, len(samples)):
        previous = samples[index - 1]
        current = samples[index]
        if (
            current.segment_index != previous.segment_index
            or current.repeat_index != previous.repeat_index
        ):
            selected.add(index)
        if current.fault_code is not None:
            selected.add(index)
    anchors = []
    for index in sorted(selected):
        sample = samples[index]
        event = "plan_start" if index == 0 else "plan_end"
        if sample.fault_code is not None:
            event = "fault"
        elif index not in {0, len(samples) - 1}:
            event = "segment_or_repeat_start"
        anchors.append(
            ReflectorSyncAnchor(
                sequence=sample.sequence,
                monotonic_timestamp_s=sample.monotonic_timestamp_s,
                event=event,
                commanded_position=sample.commanded_position,
                actual_position=sample.actual_position,
            )
        )
    return anchors
