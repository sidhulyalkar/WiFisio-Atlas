import json
from pathlib import Path

import torch

from physioatlas.federated import aggregate_model_deltas
from physioatlas.privacy import (
    HashChainAuditLog,
    audit_dataset_consent,
    create_privacy_export,
    pseudonymize_subject_id,
)
from physioatlas.schema import Modality
from physioatlas.synthetic import create_synthetic_cohort


def test_consent_audit_and_privacy_export(tmp_path: Path):
    data = tmp_path / "data"
    create_synthetic_cohort(data, subjects=2, duration_seconds=5, sample_rate_hz=10, seed=2)
    assert audit_dataset_consent(data, require_consent=True)["valid"] is True
    exported = tmp_path / "exported"
    report = create_privacy_export(
        data, exported, secret="correct horse battery staple", drop_modalities=[Modality.POSE]
    )
    assert report["manifests_rewritten"] == 2
    manifest = json.loads(next(exported.rglob("manifest.json")).read_text())
    assert manifest["subject_id"].startswith("participant-")
    assert all(stream["modality"] != "pose" for stream in manifest["streams"])
    assert pseudonymize_subject_id("abc", "secret") == pseudonymize_subject_id("abc", "secret")


def test_hash_chain_detects_tampering(tmp_path: Path):
    audit = HashChainAuditLog(tmp_path / "audit.jsonl")
    audit.append("one", {"value": 1})
    audit.append("two", {"value": 2})
    assert audit.verify()["valid"] is True
    audit.path.write_text(audit.path.read_text().replace('"value": 1', '"value": 7', 1))
    assert audit.verify()["valid"] is False


def test_federated_delta_aggregation_is_weighted_and_clipped():
    result = aggregate_model_deltas(
        [
            {"w": torch.tensor([1.0, 2.0])},
            {"w": torch.tensor([3.0, 4.0])},
        ],
        weights=[1.0, 3.0],
        clip_norm=100.0,
    )
    torch.testing.assert_close(result["w"], torch.tensor([2.5, 3.5]))
    clipped = aggregate_model_deltas(
        [{"w": torch.tensor([100.0, 0.0])}], clip_norm=1.0
    )
    assert torch.linalg.vector_norm(clipped["w"]) <= 1.00001
