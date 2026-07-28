from pathlib import Path

from physioatlas.verification import (
    doctor,
    run_fault_injection_test,
    run_smoke_test,
    verify_run,
)


def test_doctor_reports_repository_ready():
    report = doctor(Path.cwd())
    assert report["ready"] is True
    assert report["required_failures"] == 0
    assert any(check["name"] == "python_version" for check in report["checks"])


def test_smoke_test_creates_verifiable_run(tmp_path: Path):
    report = run_smoke_test(tmp_path / "smoke", seed=19)
    assert report["status"] == "passed"
    assert report["verification"]["valid"] is True
    assert Path(report["report_path"]).exists()
    assert (tmp_path / "smoke" / "run" / "experiment.json").exists()


def test_verify_run_detects_checkpoint_corruption(tmp_path: Path):
    report = run_smoke_test(tmp_path / "smoke", seed=23)
    run_root = Path(report["verification"]["run_root"])
    checkpoint = run_root / "neural" / "model.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"corruption")

    verification = verify_run(run_root)
    assert verification["valid"] is False
    assert any("checkpoint_hash" in error for error in verification["errors"])


def test_fault_injection_proves_fail_closed(tmp_path: Path):
    report = run_fault_injection_test(tmp_path / "faults", seed=31)
    assert report["status"] == "passed"
    assert all(case["detected"] for case in report["cases"])


def test_verify_run_requires_neural_record_file(tmp_path: Path):
    report = run_smoke_test(tmp_path / "smoke-missing-run", seed=37)
    run_root = Path(report["verification"]["run_root"])
    (run_root / "neural" / "run.json").unlink()
    verification = verify_run(run_root)
    assert verification["valid"] is False
    assert any("neural_record_file" in error for error in verification["errors"])
