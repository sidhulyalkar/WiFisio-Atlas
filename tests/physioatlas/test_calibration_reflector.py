from __future__ import annotations

import pytest

from physioatlas.calibration_reflector import (
    DeterministicSimulatedReflector,
    ReflectorSafetyError,
    SimulatedReflectorDynamics,
    analyze_reflector_telemetry,
)
from physioatlas.calibration_reflector_schema import (
    MotionUnit,
    ReflectorDeviceIdentity,
    ReflectorHealthThresholds,
    ReflectorInstallation,
    ReflectorSafetyLimits,
    ReflectorTrajectoryPlan,
    SafetyState,
    TrajectoryKind,
    TrajectorySegment,
)


def _driver(
    dynamics: SimulatedReflectorDynamics = SimulatedReflectorDynamics(),
) -> DeterministicSimulatedReflector:
    return DeterministicSimulatedReflector(
        identity=ReflectorDeviceIdentity(
            device_id="sim-reflector-01",
            hardware_revision="simulation-v1",
            firmware_version="simulation-v1",
            driver_name="deterministic-simulator",
            telemetry_clock_id="sim-monotonic-01",
        ),
        installation=ReflectorInstallation(
            environment_id="test-room",
            origin_m=(1.0, 2.0, 0.8),
            axis_unit_vector=(1.0, 0.0, 0.0),
            motion_unit=MotionUnit.DEGREES,
            home_position=0.0,
            reflector_diameter_m=0.2,
            reflector_material="synthetic-reference",
        ),
        limits=ReflectorSafetyLimits(
            minimum_position=-30.0,
            maximum_position=30.0,
            maximum_velocity_per_s=80.0,
            maximum_acceleration_per_s2=300.0,
            maximum_motor_current_a=1.5,
        ),
        dynamics=dynamics,
    )


def _plan() -> ReflectorTrajectoryPlan:
    return ReflectorTrajectoryPlan(
        plan_id="repeatable-sweep",
        sample_rate_hz=50.0,
        segments=[
            TrajectorySegment(
                kind=TrajectoryKind.SINUSOID,
                duration_s=4.0,
                repeats=3,
                center_position=0.0,
                amplitude=10.0,
                cycles=1.0,
            )
        ],
    )


def test_rejects_unsafe_plan_before_simulation() -> None:
    driver = _driver()
    driver.arm()
    unsafe = ReflectorTrajectoryPlan(
        plan_id="unsafe",
        sample_rate_hz=20.0,
        segments=[
            TrajectorySegment(
                kind=TrajectoryKind.SWEEP,
                duration_s=1.0,
                start_position=-5.0,
                end_position=40.0,
            )
        ],
    )

    with pytest.raises(ReflectorSafetyError, match="position limits"):
        driver.execute(unsafe)

    assert driver.safety_state == SafetyState.ARMED


def test_simulation_is_deterministic_and_fail_closed() -> None:
    first = _driver(SimulatedReflectorDynamics(noise_standard_deviation=0.05, seed=9))
    second = _driver(SimulatedReflectorDynamics(noise_standard_deviation=0.05, seed=9))
    with pytest.raises(ReflectorSafetyError, match="armed"):
        first.execute(_plan())
    first.arm()
    second.arm()

    first_samples = first.execute(_plan())
    second_samples = second.execute(_plan())

    assert [item.model_dump() for item in first_samples] == [
        item.model_dump() for item in second_samples
    ]
    assert all(item.synthetic for item in first_samples)
    assert first_samples[-1].plan_complete
    assert first.safety_state == SafetyState.SAFE_HOLD


def test_estop_latches_and_requires_explicit_clear() -> None:
    driver = _driver()
    driver.engage_estop()

    assert driver.safety_state == SafetyState.ESTOPPED
    with pytest.raises(ReflectorSafetyError, match="e-stop"):
        driver.arm()

    driver.clear_estop()
    assert driver.safety_state == SafetyState.SAFE_HOLD
    driver.arm()
    assert driver.safety_state == SafetyState.ARMED


def test_analysis_recovers_lag_and_reports_trajectory_health() -> None:
    driver = _driver(
        SimulatedReflectorDynamics(
            response_lag_s=0.16,
            tracking_bias=0.1,
            noise_standard_deviation=0.01,
            seed=2,
        )
    )
    driver.arm()

    health = analyze_reflector_telemetry(
        driver.execute(_plan()),
        ReflectorHealthThresholds(
            maximum_lag_s=0.3,
            maximum_rmse=0.25,
            maximum_absolute_error=0.5,
            maximum_repeatability_rmse=0.1,
            minimum_samples=100,
        ),
    )

    assert health.valid_for_rf_calibration
    assert health.estimated_lag_s == pytest.approx(0.16, abs=0.02)
    assert health.trajectory_rmse is not None
    assert health.trajectory_rmse < 0.15
    assert health.repeatability_rmse is not None
    assert health.repeatability_rmse < 0.05
    assert len(health.synchronization_anchors) >= 4
    assert health.synthetic


def test_fault_invalidates_calibration_and_latches_driver() -> None:
    driver = _driver(SimulatedReflectorDynamics(fault_at_s=1.0))
    driver.arm()

    samples = driver.execute(_plan())
    health = analyze_reflector_telemetry(samples)

    assert driver.safety_state == SafetyState.FAULTED
    assert not health.valid_for_rf_calibration
    assert health.faults_observed == ["simulated_fault"]
    assert "actuator_fault" in health.invalidation_reasons
    assert "incomplete_trajectory" in health.invalidation_reasons
    assert not samples[-1].plan_complete
    with pytest.raises(ReflectorSafetyError, match="cannot arm"):
        driver.arm()
