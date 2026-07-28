import json
from pathlib import Path

from physioatlas.household import enroll_household
from physioatlas.household_live import run_household_jsonl
from physioatlas.household_protocol import initialize_household_workspace
from physioatlas.research_hub import ResearchHubStore
from physioatlas.research_hub_worker import ResearchHubActionWorker
from physioatlas.synthetic import create_synthetic_cohort


def test_workspace_initialization_does_not_grant_consent(tmp_path: Path):
    result = initialize_household_workspace(
        tmp_path / "workspace",
        household_id="home-test",
        members=["member-a=Alice", "member-b=Bob"],
    )
    assert len(result["members"]) == 2
    consent = json.loads((tmp_path / "workspace/members/member-a/consent.template.json").read_text())
    assert consent["participant_acknowledged"] is False
    assert consent["allow_identity_enrollment"] is False


def test_jsonl_bridge_and_worker(tmp_path: Path):
    data = tmp_path / "data"
    create_synthetic_cohort(data, subjects=2, sessions_per_subject=2, duration_seconds=12, seed=101)
    registry_path = tmp_path / "registry.json"
    assert enroll_household(data, registry_path, minimum_sessions=2)["ready"]
    registry = json.loads(registry_path.read_text())
    vector = registry["members"][0]["profiles"][0]["centroid"]
    jsonl = tmp_path / "observations.jsonl"
    rows = []
    for step in range(5):
        rows.append(
            json.dumps(
                {
                    "timestamp_s": float(step),
                    "position_m": [1.0 + 0.05 * step, 1.0],
                    "embeddings": {"wifi_csi": vector},
                    "signal_quality": 0.95,
                    "observability": 0.9,
                }
            )
        )
    jsonl.write_text("\n".join(rows) + "\n")
    state_path = tmp_path / "hub.json"
    report = run_household_jsonl(jsonl, registry_path, state_path)
    assert report["tracks"] == 1
    assert report["identified"] == 1
    assert report["valid"] is True
    assert report["status"] == "passed"

    store = ResearchHubStore(state_path)
    store.enqueue_action("run_calibration")
    worker = ResearchHubActionWorker(store, dataset_root=data, registry_path=registry_path, output_root=tmp_path / "actions")
    result = worker.process_once()
    assert result["processed"] == 1
    assert result["results"][0]["status"] == "completed"


def test_household_udp_loopback(tmp_path: Path):
    import socket
    import threading
    import time

    from physioatlas.household_live import run_household_udp

    data = tmp_path / "udp-data"
    create_synthetic_cohort(data, subjects=2, sessions_per_subject=2, duration_seconds=10, seed=119)
    registry_path = tmp_path / "udp-registry.json"
    assert enroll_household(data, registry_path, minimum_sessions=2)["ready"]
    registry = json.loads(registry_path.read_text())
    vector = registry["members"][0]["profiles"][0]["centroid"]
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    result_holder = {}

    def run_server():
        result_holder["result"] = run_household_udp(
            registry_path,
            tmp_path / "udp-state.json",
            host="127.0.0.1",
            port=port,
            duration_s=0.9,
        )

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    time.sleep(0.15)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        for step in range(5):
            sender.sendto(
                json.dumps(
                    {
                        "timestamp_s": float(step),
                        "position_m": [1.0 + 0.02 * step, 1.0],
                        "embeddings": {"wifi_csi": vector},
                        "signal_quality": 0.95,
                        "observability": 0.9,
                    }
                ).encode(),
                ("127.0.0.1", port),
            )
            time.sleep(0.04)
    finally:
        sender.close()
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert result_holder["result"]["accepted_observations"] == 5
    assert result_holder["result"]["tracks"] == 1
    assert result_holder["result"]["valid"] is True
    assert result_holder["result"]["status"] == "passed"
