from pathlib import Path

from physioatlas.dataset import load_windows_from_root, subject_split
from physioatlas.synthetic import create_synthetic_cohort
from physioatlas.training import TrainConfig, train_model
from physioatlas.validation import validate_dataset


def test_synthetic_dataset_and_subject_split(tmp_path: Path):
    data = tmp_path / "data"
    manifests = create_synthetic_cohort(
        data,
        subjects=3,
        duration_seconds=10.0,
        sample_rate_hz=10.0,
        seed=3,
    )
    assert len(manifests) == 3
    validation = validate_dataset(data)
    assert validation["valid"] is True
    assert validation["subjects"] == 3
    examples = load_windows_from_root(
        data,
        sample_rate_hz=10.0,
        window_seconds=4.0,
        stride_seconds=2.0,
        target_key="ecg",
    )
    train, held_out = subject_split(
        [example.subject_id for example in examples],
        validation_fraction=0.34,
        seed=4,
    )
    assert train.isdisjoint(held_out)


def test_short_end_to_end_training(tmp_path: Path):
    data = tmp_path / "data"
    create_synthetic_cohort(
        data,
        subjects=3,
        duration_seconds=8.0,
        sample_rate_hz=10.0,
        seed=5,
    )
    examples = load_windows_from_root(
        data,
        sample_rate_hz=10.0,
        window_seconds=4.0,
        stride_seconds=2.0,
        target_key="ecg",
    )
    result = train_model(
        examples,
        tmp_path / "run",
        TrainConfig(
            epochs=1,
            batch_size=4,
            hidden_dim=16,
            layers=1,
            validation_fraction=0.34,
            seed=2,
            device="cpu",
        ),
    )
    assert result["status"] == "completed"
    assert set(result["train_subjects"]).isdisjoint(result["validation_subjects"])
    assert (tmp_path / "run" / "model.pt").exists()
    assert (tmp_path / "run" / "run.json").exists()
    assert len(result["checkpoint_sha256"]) == 64


def test_synthetic_dataset_fingerprint_is_reproducible(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    create_synthetic_cohort(first, subjects=3, duration_seconds=8.0, sample_rate_hz=10.0, seed=101)
    create_synthetic_cohort(second, subjects=3, duration_seconds=8.0, sample_rate_hz=10.0, seed=101)
    assert validate_dataset(first)["dataset_sha256"] == validate_dataset(second)["dataset_sha256"]
