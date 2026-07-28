from pathlib import Path

import yaml

from physioatlas.experiment import run_experiment
from physioatlas.synthetic import create_synthetic_cohort


def test_declarative_experiment_writes_baselines(tmp_path: Path):
    data = tmp_path / "data"
    create_synthetic_cohort(
        data,
        subjects=3,
        duration_seconds=8.0,
        sample_rate_hz=10.0,
        seed=8,
    )
    config = {
        "name": "test-experiment",
        "hypothesis": "software pipeline works",
        "data": {
            "root": str(data),
            "target": "ecg",
            "sample_rate_hz": 10.0,
            "window_seconds": 4.0,
            "stride_seconds": 2.0,
        },
        "training": {
            "epochs": 1,
            "batch_size": 4,
            "hidden_dim": 16,
            "layers": 1,
            "validation_fraction": 0.34,
            "seed": 3,
            "device": "cpu",
        },
    }
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(config))
    output = tmp_path / "output"
    result = run_experiment(config_path, output)
    assert set(result["baselines"]) == {
        "mean",
        "ridge_rf",
        "ridge_rf_auxiliary",
        "null_ridge_rf_time_shift",
        "null_ridge_rf_label_permutation",
    }
    assert result["neural"]["status"] == "completed"
    assert (output / "experiment.json").exists()
    assert (output / "config.snapshot.yaml").exists()
