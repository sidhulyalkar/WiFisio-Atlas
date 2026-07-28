from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from physioatlas.calibration_memory import (
    CalibrationProfile,
    load_profile,
    save_profile,
    select_calibration_profile,
)
from physioatlas.distillation import privileged_distillation_loss, teacher_removal_gap


def test_privileged_distillation_handles_missing_teachers() -> None:
    student = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], requires_grad=True)
    teacher = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    loss, report = privileged_distillation_loss(
        student,
        teacher,
        teacher_available=torch.tensor([1.0, 0.0, 1.0]),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert report.available_fraction == pytest.approx(2 / 3)
    assert teacher_removal_gap(0.2, 0.35) == pytest.approx(0.15)


def test_calibration_memory_hash_and_selection(tmp_path: Path) -> None:
    room_a = CalibrationProfile(
        profile_id="a",
        environment_id="room-a",
        hardware_ids=["esp32-1"],
        geometry_fingerprint=[0.0, 0.0, 1.0],
        uncertainty_profile={"median_interval_width": 0.2},
    )
    room_b = CalibrationProfile(
        profile_id="b",
        environment_id="room-b",
        hardware_ids=["esp32-2"],
        geometry_fingerprint=[1.0, 1.0, 1.0],
    )
    path = tmp_path / "profile.json"
    save_profile(path, room_a)
    loaded = load_profile(path)
    assert loaded.profile_id == "a"
    chosen, detail = select_calibration_profile(
        [room_b, loaded],
        environment_id="room-a",
        hardware_ids=["esp32-1"],
        geometry_fingerprint=np.asarray([0.0, 0.0, 1.0]),
    )
    assert chosen.profile_id == "a"
    assert detail["total_penalty"] == pytest.approx(0.0)

    payload = path.read_text(encoding="utf-8").replace('"room-a"', '"room-z"')
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_profile(path)
