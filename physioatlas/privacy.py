from __future__ import annotations

import hashlib
import hmac
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from .io import dump_json, load_manifest
from .schema import Modality


class ConsentPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "physioatlas.consent.v1"
    consent_id: str
    subject_id: str
    granted_at_utc: str
    expires_at_utc: Optional[str] = None
    allowed_modalities: list[Modality] = Field(default_factory=list)
    allowed_targets: list[str] = Field(default_factory=list)
    allowed_uses: list[str] = Field(default_factory=lambda: ["research"])
    allow_model_training: bool = True
    allow_raw_export: bool = False
    allow_identity_enrollment: bool = False
    allow_live_tracking: bool = False
    allow_household_dashboard: bool = False
    participant_acknowledged: bool = False
    withdrawal_contact: Optional[str] = None


def pseudonymize_subject_id(subject_id: str, secret: str, *, prefix: str = "participant") -> str:
    if not secret:
        raise ValueError("Pseudonymization secret cannot be empty")
    digest = hmac.new(secret.encode(), subject_id.encode(), hashlib.sha256).hexdigest()[:16]
    return f"{prefix}-{digest}"


def load_consent(path: Union[str, Path]) -> ConsentPolicy:
    return ConsentPolicy.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def validate_consent(
    policy: ConsentPolicy,
    *,
    subject_id: str,
    modalities: Iterable[Modality],
    targets: Iterable[str],
    use: str = "research",
    at_utc: Optional[datetime] = None,
) -> dict:
    at = at_utc or datetime.now(timezone.utc)
    errors: list[str] = []
    if policy.subject_id != subject_id:
        errors.append("Consent subject does not match manifest subject")
    if use not in policy.allowed_uses:
        errors.append(f"Consent does not allow use {use!r}")
    allowed_modalities = set(policy.allowed_modalities)
    denied_modalities = sorted({item.value for item in modalities if item not in allowed_modalities})
    if denied_modalities:
        errors.append(f"Consent does not allow modalities: {denied_modalities}")
    denied_targets = sorted(set(targets) - set(policy.allowed_targets))
    if denied_targets:
        errors.append(f"Consent does not allow targets: {denied_targets}")
    if policy.expires_at_utc:
        expires = datetime.fromisoformat(policy.expires_at_utc.replace("Z", "+00:00"))
        if at >= expires:
            errors.append("Consent has expired")
    if use == "model_training" and not policy.allow_model_training:
        errors.append("Consent does not allow model training")
    if use == "identity_enrollment" and not policy.allow_identity_enrollment:
        errors.append("Consent does not allow identity enrollment")
    if use == "live_tracking" and not policy.allow_live_tracking:
        errors.append("Consent does not allow live tracking")
    if use == "household_dashboard" and not policy.allow_household_dashboard:
        errors.append("Consent does not allow household dashboard display")
    if use in {"identity_enrollment", "live_tracking", "household_dashboard"} and not policy.participant_acknowledged:
        errors.append("Participant acknowledgement is required for household identity or live tracking")
    return {
        "schema_version": "physioatlas.consent-check.v1",
        "valid": not errors,
        "consent_id": policy.consent_id,
        "errors": errors,
    }


class HashChainAuditLog:
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: str, payload: dict) -> dict:
        previous_hash = "0" * 64
        if self.path.exists():
            lines = [line for line in self.path.read_text(encoding="utf-8").splitlines() if line]
            if lines:
                previous_hash = json.loads(lines[-1])["entry_hash"]
        entry = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "payload": payload,
            "previous_hash": previous_hash,
        }
        canonical = json.dumps(entry, sort_keys=True, separators=(",", ":")).encode()
        entry["entry_hash"] = hashlib.sha256(canonical).hexdigest()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry

    def verify(self) -> dict:
        previous_hash = "0" * 64
        entries = 0
        errors: list[str] = []
        if not self.path.exists():
            return {"valid": True, "entries": 0, "errors": []}
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line:
                continue
            entry = json.loads(line)
            claimed = entry.pop("entry_hash")
            if entry.get("previous_hash") != previous_hash:
                errors.append(f"line {line_number}: previous hash mismatch")
            actual = hashlib.sha256(
                json.dumps(entry, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if claimed != actual:
                errors.append(f"line {line_number}: entry hash mismatch")
            previous_hash = claimed
            entries += 1
        return {"valid": not errors, "entries": entries, "errors": errors}


def audit_dataset_consent(root: Union[str, Path], *, require_consent: bool = True) -> dict:
    reports = []
    errors: list[str] = []
    for manifest_path in sorted(Path(root).rglob("manifest.json")):
        manifest = load_manifest(manifest_path, resolve_paths=True)
        if not manifest.consent_path:
            if require_consent:
                errors.append(f"{manifest.session_id}: missing consent_path")
            reports.append({"session_id": manifest.session_id, "valid": not require_consent, "missing": True})
            continue
        policy = load_consent(manifest.consent_path)
        report = validate_consent(
            policy,
            subject_id=manifest.subject_id,
            modalities=[stream.modality for stream in manifest.streams],
            targets=[target.name for target in manifest.targets],
            use="model_training",
        )
        report["session_id"] = manifest.session_id
        reports.append(report)
        errors.extend(f"{manifest.session_id}: {error}" for error in report["errors"])
    return {
        "schema_version": "physioatlas.privacy-audit.v1",
        "valid": not errors,
        "sessions": reports,
        "errors": errors,
    }


def create_privacy_export(
    source_root: Union[str, Path],
    output_root: Union[str, Path],
    *,
    secret: str,
    drop_modalities: Iterable[Modality] = (),
) -> dict:
    source = Path(source_root).resolve()
    output = Path(output_root).resolve()
    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(source, output)
    dropped = set(drop_modalities)
    rewritten = 0
    for manifest_path in sorted(output.rglob("manifest.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["subject_id"] = pseudonymize_subject_id(payload["subject_id"], secret)
        retained_streams = []
        for stream in payload.get("streams", []):
            if Modality(stream["modality"]) in dropped:
                stream_path = manifest_path.parent / stream["path"]
                stream_path.unlink(missing_ok=True)
            else:
                retained_streams.append(stream)
        payload["streams"] = retained_streams
        payload["consent_reference"] = None
        payload["consent_path"] = None
        payload.setdefault("metadata", {})["privacy_export"] = True
        dump_json(manifest_path, payload)
        rewritten += 1
    return {
        "schema_version": "physioatlas.privacy-export.v1",
        "source": str(source),
        "output": str(output),
        "manifests_rewritten": rewritten,
        "dropped_modalities": sorted(item.value for item in dropped),
    }
