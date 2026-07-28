from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import yaml

from .baselines import (
    evaluate_mean_baseline,
    evaluate_ridge_baseline,
    permute_target_windows,
    time_shift_targets,
)
from .config import ExperimentConfig
from .dataset import load_windows_from_root
from .physiology import analyze_dataset_propagation
from .privacy import audit_dataset_consent
from .splits import count_examples_by_domain, group_split
from .training import TrainConfig, train_model
from .utils import make_json_safe
from .validation import validate_dataset


def load_experiment_config(path: Union[str, Path]) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Experiment configuration must be a YAML mapping")
    return ExperimentConfig.model_validate(config).model_dump(mode="python")


def run_experiment(
    config_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
) -> dict:
    config_path = Path(config_path)
    config = load_experiment_config(config_path)
    data_cfg = config["data"]
    train_cfg = dict(config["training"])
    split_cfg = config["split"]
    data_root = Path(data_cfg["root"])
    if not data_root.is_absolute():
        data_root = (config_path.resolve().parent / data_root).resolve()
    if output_dir is not None:
        output = Path(output_dir).resolve()
    else:
        output = Path(config.get("output") or "outputs/physioatlas/experiment")
        if not output.is_absolute():
            output = (config_path.resolve().parent / output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    validation = validate_dataset(
        data_root,
        require_consent=bool(data_cfg.get("require_consent", False)),
    )
    if not validation["valid"]:
        raise ValueError(f"Dataset validation failed: {validation['errors']}")

    examples = load_windows_from_root(
        data_root,
        sample_rate_hz=float(data_cfg["sample_rate_hz"]),
        window_seconds=float(data_cfg["window_seconds"]),
        stride_seconds=float(data_cfg["stride_seconds"]),
        target_key=str(data_cfg["target"]),
        max_gap_s=float(data_cfg["max_gap_s"]),
    )
    train_cfg.update(
        {
            "validation_fraction": float(split_cfg["validation_fraction"]),
            "split_strategy": str(split_cfg["strategy"]),
            "held_out_groups": tuple(split_cfg.get("held_out_groups", [])),
        }
    )
    training = TrainConfig(**train_cfg)
    train_examples, validation_examples, split = group_split(
        examples,
        strategy=training.split_strategy,
        validation_fraction=training.validation_fraction,
        seed=training.seed,
        held_out_groups=training.held_out_groups,
    )

    baseline_results = {
        "mean": evaluate_mean_baseline(train_examples, validation_examples),
        "ridge_rf": evaluate_ridge_baseline(
            train_examples, validation_examples, include_auxiliary=False
        ),
        "ridge_rf_auxiliary": evaluate_ridge_baseline(
            train_examples, validation_examples, include_auxiliary=True
        ),
        "null_ridge_rf_time_shift": evaluate_ridge_baseline(
            time_shift_targets(train_examples),
            validation_examples,
            include_auxiliary=False,
        ),
        "null_ridge_rf_label_permutation": evaluate_ridge_baseline(
            permute_target_windows(train_examples, seed=training.seed),
            validation_examples,
            include_auxiliary=False,
        ),
    }
    neural_result = train_model(examples, output / "neural", training)
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    dataset_sha256 = validation.get("dataset_sha256") or "unavailable"
    identity_config = json.loads(json.dumps(config, default=str))
    identity_config["data"]["root"] = "<dataset>"
    identity_config["output"] = None
    identity_bytes = json.dumps(
        identity_config, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    experiment_identity_sha256 = hashlib.sha256(identity_bytes).hexdigest()
    run_id = hashlib.sha256(
        f"{experiment_identity_sha256}:{dataset_sha256}".encode("utf-8")
    ).hexdigest()[:16]
    neural_rmse = float(neural_result["final_metrics"]["rmse"])
    comparison = {
        name: {
            "rmse": float(metrics["rmse"]),
            "neural_minus_baseline_rmse": neural_rmse - float(metrics["rmse"]),
        }
        for name, metrics in baseline_results.items()
    }
    privacy = audit_dataset_consent(
        data_root,
        require_consent=bool(data_cfg.get("require_consent", False)),
    )
    propagation = analyze_dataset_propagation(
        data_root,
        sample_rate_hz=float(data_cfg["sample_rate_hz"]),
    )
    result = {
        "schema_version": "physioatlas.experiment.v2",
        "run_id": run_id,
        "research_only": True,
        "clinical_claim_allowed": False,
        "observability_required": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "name": config["name"],
        "hypothesis": config["hypothesis"],
        "claim_boundary": config["claim_boundary"],
        "config_sha256": config_sha256,
        "experiment_identity_sha256": experiment_identity_sha256,
        "dataset_sha256": dataset_sha256,
        "dataset_validation": validation,
        "privacy_audit": privacy,
        "propagation_analysis": propagation,
        "split": {
            "strategy": training.split_strategy,
            **split,
            "domain_counts": {
                "train": count_examples_by_domain(train_examples),
                "validation": count_examples_by_domain(validation_examples),
            },
        },
        "baselines": baseline_results,
        "diagnostic_comparisons": comparison,
        "neural": neural_result,
    }
    result = make_json_safe(result)
    with (output / "experiment.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    with (output / "config.snapshot.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)
    return result
