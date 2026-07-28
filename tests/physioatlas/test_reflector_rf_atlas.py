
from physioatlas.calibration_reflector import (
    DeterministicSimulatedReflector,
    SimulatedReflectorDynamics,
    analyze_reflector_telemetry,
)
from physioatlas.calibration_reflector_schema import (
    MotionUnit,
    ReflectorDeviceIdentity,
    ReflectorInstallation,
    ReflectorSafetyLimits,
    ReflectorTrajectoryPlan,
    TrajectoryKind,
    TrajectorySegment,
)
from physioatlas.reflector_rf_atlas import (
    build_reflector_rf_atlas,
    compare_reflector_atlases,
)
from physioatlas.reflector_rf_atlas_schema import ReflectorRfSample


def _telemetry():
    driver = DeterministicSimulatedReflector(
        ReflectorDeviceIdentity(
            device_id="sim",
            hardware_revision="sim",
            firmware_version="sim",
            driver_name="sim",
            telemetry_clock_id="clock",
        ),
        ReflectorInstallation(
            environment_id="room",
            origin_m=(0.0, 0.0, 0.0),
            axis_unit_vector=(1.0, 0.0, 0.0),
            motion_unit=MotionUnit.DEGREES,
            home_position=0.0,
            reflector_diameter_m=0.25,
            reflector_material="simulated",
        ),
        ReflectorSafetyLimits(
            minimum_position=-20.0,
            maximum_position=20.0,
            maximum_velocity_per_s=100.0,
            maximum_acceleration_per_s2=500.0,
            maximum_motor_current_a=2.0,
        ),
        SimulatedReflectorDynamics(response_lag_s=0.0),
    )
    plan = ReflectorTrajectoryPlan(
        plan_id="dwell-map",
        sample_rate_hz=20.0,
        segments=[
            TrajectorySegment(kind=TrajectoryKind.DWELL, duration_s=1.0, target_position=-10.0),
            TrajectorySegment(kind=TrajectoryKind.DWELL, duration_s=1.0, target_position=0.0),
            TrajectorySegment(kind=TrajectoryKind.DWELL, duration_s=1.0, target_position=10.0),
        ],
    )
    driver.arm()
    telemetry = driver.execute(plan)
    return telemetry, analyze_reflector_telemetry(telemetry)


def _radio(telemetry, scale: float = 1.0):
    result = []
    for item in telemetry:
        if item.sequence % 2 or item.actual_velocity_per_s > 0.1:
            continue
        for source, band, link, multiplier in (
            ("wifi_csi", "2.4ghz", "csi-24", 1.0),
            ("wifi_csi", "5ghz", "csi-5", 1.5),
            ("wifi_bfi", "5ghz", "bfi-5", 1.3),
        ):
            response = scale * multiplier * (1.0 + abs(item.commanded_position) / 10.0)
            result.append(
                ReflectorRfSample(
                    timestamp_reflector_clock_s=item.monotonic_timestamp_s,
                    alignment_receipt_id="alignment-1",
                    source=source,
                    link_id=link,
                    frequency_band=band,
                    response_energy=response,
                    response_features=[response, response * 0.5],
                    quality=0.95,
                )
            )
    return result


def test_motor_off_dwell_atlas_and_drift_detection():
    telemetry, health = _telemetry()
    baseline = build_reflector_rf_atlas(
        telemetry,
        _radio(telemetry),
        health,
        atlas_id="baseline",
        environment_id="room",
    )
    assert baseline.motor_off_dwell_verified
    assert len(baseline.link_sensitivity) == 3
    assert baseline.cross_band_energy_correlation == 1.0
    stable = build_reflector_rf_atlas(
        telemetry,
        _radio(telemetry, 1.02),
        health,
        atlas_id="stable",
        environment_id="room",
    )
    stable_report = compare_reflector_atlases(baseline, stable)
    assert not stable_report.recalibration_required
    drifted = build_reflector_rf_atlas(
        telemetry,
        _radio(telemetry, 2.0),
        health,
        atlas_id="drifted",
        environment_id="room",
    )
    drift_report = compare_reflector_atlases(
        baseline, drifted, maximum_drift_score=0.2
    )
    assert drift_report.recalibration_required
    assert "reflector_transfer_drift_exceeded" in drift_report.reasons


def test_invalid_actuator_health_fails_closed():
    telemetry, health = _telemetry()
    invalid = health.model_copy(
        update={
            "valid_for_rf_calibration": False,
            "invalidation_reasons": ["manual_test"],
        }
    )
    try:
        build_reflector_rf_atlas(
            telemetry,
            _radio(telemetry),
            invalid,
            atlas_id="invalid",
            environment_id="room",
        )
    except ValueError as exc:
        assert "health" in str(exc)
    else:
        raise AssertionError("invalid actuator health should reject RF atlas")
