from pathlib import Path

from physioatlas.calibration_reflector import (
    DeterministicSimulatedReflector,
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
from physioatlas.research_hub import ResearchHubStore
from physioatlas.rf_fusion_schema import FusionDecision
from physioatlas.rf_hub import publish_fusion_decision, publish_reflector_state


def _reflector():
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
            reflector_diameter_m=0.2,
            reflector_material="simulated",
        ),
        ReflectorSafetyLimits(
            minimum_position=-20.0,
            maximum_position=20.0,
            maximum_velocity_per_s=100.0,
            maximum_acceleration_per_s2=500.0,
            maximum_motor_current_a=2.0,
        ),
    )
    driver.arm()
    telemetry = driver.execute(
        ReflectorTrajectoryPlan(
            plan_id="hub-test",
            sample_rate_hz=20.0,
            segments=[
                TrajectorySegment(
                    kind=TrajectoryKind.SINUSOID,
                    duration_s=2.0,
                    repeats=2,
                    center_position=0.0,
                    amplitude=5.0,
                    cycles=1.0,
                )
            ],
        )
    )
    return telemetry, analyze_reflector_telemetry(telemetry)


def test_reflector_and_fusion_publish_additive_hub_state(tmp_path: Path):
    store = ResearchHubStore(tmp_path / "hub.json")
    telemetry, health = _reflector()
    reflector = publish_reflector_state(store, telemetry=telemetry, health=health)
    assert reflector["synthetic"] is True
    assert reflector["hardware_control_available"] is False
    decision = FusionDecision(
        timestamp_s=1.0,
        status="abstained",
        reason="cross_modal_conflict",
        active_zones=[],
        fused_probabilities={"desk": 0.5},
        cross_modal_conflict=0.8,
        observability=0.8,
        sources_present=["wifi_csi", "wifi_bfi"],
        evidence_ids=["csi", "bfi"],
    )
    fusion = publish_fusion_decision(store, decision)
    assert fusion["abstention_summary"]["cross_modal_conflict"] == 1
    state = store.read()
    assert state["reflector"]["plan_id"] == "hub-test"
    assert state["rf_fusion"]["last_decision"]["reason"] == "cross_modal_conflict"
