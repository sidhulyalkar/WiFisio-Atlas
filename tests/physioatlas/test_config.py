from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from physioatlas.experiment import load_experiment_config


def _base_config() -> dict:
    return {
        "name": "config-validation",
        "hypothesis": "A deterministic configuration should validate cleanly.",
        "data": {"root": "data/example"},
        "training": {"epochs": 1},
    }


def test_config_rejects_unknown_keys(tmp_path: Path):
    config = _base_config()
    config["training"]["epochz"] = 2
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_experiment_config(path)


def test_config_supplies_research_guardrails(tmp_path: Path):
    path = tmp_path / "good.yaml"
    path.write_text(yaml.safe_dump(_base_config()), encoding="utf-8")
    loaded = load_experiment_config(path)
    assert loaded["research_only"] is True
    assert "no anatomical or clinical claim" in loaded["claim_boundary"]
