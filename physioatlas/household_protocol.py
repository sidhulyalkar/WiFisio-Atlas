from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Union

import yaml

from .io import dump_json
from .privacy import ConsentPolicy
from .schema import Modality


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-").lower()
    if not slug:
        raise ValueError("Member alias must contain at least one letter or number")
    return slug


def initialize_household_workspace(
    output_root: Union[str, Path],
    *,
    household_id: str,
    members: Iterable[str],
) -> dict[str, Any]:
    """Create a consent-first household-study workspace without granting consent.

    Member strings may be ``subject_id=Display Alias`` or just a display alias.
    The generated consent documents remain drafts until each participant sets
    the three requested permissions and ``participant_acknowledged`` to true.
    """
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    member_records = []
    aliases = {}
    for index, item in enumerate(members, start=1):
        if "=" in item:
            subject_id, alias = item.split("=", 1)
            subject_id, alias = subject_id.strip(), alias.strip()
        else:
            alias = item.strip()
            subject_id = f"member-{index:02d}-{_slug(alias)}"
        if not subject_id or not alias:
            raise ValueError(f"Invalid member specification: {item!r}")
        member_dir = output / "members" / subject_id
        member_dir.mkdir(parents=True, exist_ok=True)
        consent = ConsentPolicy(
            consent_id=f"draft-consent-{subject_id}",
            subject_id=subject_id,
            granted_at_utc=datetime.now(timezone.utc).isoformat(),
            allowed_modalities=[
                Modality.WIFI_CSI,
                Modality.MMWAVE,
                Modality.ECG,
                Modality.PPG,
                Modality.RESP_BELT,
                Modality.POSE,
                Modality.ULTRASOUND,
                Modality.SYNC,
            ],
            allowed_targets=[
                "ecg_waveform",
                "respiratory_waveform",
                "ultrasound_diaphragm_displacement",
            ],
            allowed_uses=[
                "research",
                "model_training",
                "identity_enrollment",
                "live_tracking",
                "household_dashboard",
            ],
            allow_model_training=False,
            allow_identity_enrollment=False,
            allow_live_tracking=False,
            allow_household_dashboard=False,
            participant_acknowledged=False,
        )
        dump_json(member_dir / "consent.template.json", consent.model_dump(mode="json"))
        aliases[subject_id] = alias
        member_records.append(
            {
                "subject_id": subject_id,
                "display_alias": alias,
                "consent_template": str((member_dir / "consent.template.json").relative_to(output)),
                "sessions_dir": str((member_dir / "sessions").relative_to(output)),
            }
        )
        (member_dir / "sessions").mkdir(exist_ok=True)
    dump_json(output / "aliases.json", aliases)
    household_config = {
        "schema_version": "physioatlas.household-study.v1",
        "household_id": household_id,
        "members": member_records,
        "minimum_enrollment_sessions": 3,
        "calibration_protocols": [
            "quiet_seated",
            "quiet_supine",
            "paced_breathing",
            "speaking_and_small_motion",
            "room_reentry",
        ],
        "negative_controls": [
            "empty_room",
            "fan_or_moving_curtain",
            "unknown_consented_test_guest",
            "wrong_geometry_replay",
        ],
        "privacy": {
            "local_only": True,
            "raw_identity_embeddings_exported": False,
            "unknown_people_anonymous": True,
        },
    }
    (output / "household.yaml").write_text(yaml.safe_dump(household_config, sort_keys=False), encoding="utf-8")
    readme = f"""# {household_id} PhysioAtlas workspace

This workspace is research-only and local-first.

1. Review each `members/*/consent.template.json` with that participant.
2. Only after explicit agreement, set the requested permissions and
   `participant_acknowledged` to `true`, then save it as `consent.json`.
3. Record at least three sessions per person on different days or positions.
4. Keep visitors and unconsented people anonymous. Do not create profiles for them.
5. Run dataset validation, household enrollment, held-out evaluation, and the
   household smoke suite before enabling live identity labels.

A normal home router often does not expose CSI. RuView generally needs compatible
ESP32 or research-NIC capture hardware. RSSI-only network telemetry is not an
adequate substitute for the CSI experiments described here.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    return {
        "schema_version": "physioatlas.household-workspace.v1",
        "output": str(output),
        "household_id": household_id,
        "members": member_records,
        "next_step": "Review and explicitly acknowledge each consent template before recording.",
    }
