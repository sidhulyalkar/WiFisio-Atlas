import json
import threading
import urllib.request
from pathlib import Path

from physioatlas.dashboard import DashboardServer, LiveStateStore
from physioatlas.dataset import load_windows_from_root
from physioatlas.splits import example_group, group_split
from physioatlas.synthetic import create_synthetic_cohort


def test_environment_and_hardware_group_splits(tmp_path: Path):
    data = tmp_path / "data"
    create_synthetic_cohort(data, subjects=4, duration_seconds=8, sample_rate_hz=10, seed=9)
    examples = load_windows_from_root(
        data,
        sample_rate_hz=10,
        window_seconds=4,
        stride_seconds=2,
        target_key="ecg",
    )
    train, validation, summary = group_split(
        examples,
        strategy="environment",
        validation_fraction=0.5,
        seed=2,
    )
    assert set(summary["train_groups"]).isdisjoint(summary["validation_groups"])
    assert {example_group(item, "environment") for item in train}.isdisjoint(
        {example_group(item, "environment") for item in validation}
    )
    assert all(example.hardware_domain != "unknown" for example in examples)


def test_dashboard_serves_health_and_atomic_state(tmp_path: Path):
    store = LiveStateStore(tmp_path / "state.json")
    store.publish({"status": "ready", "observability": 0.8})
    server = DashboardServer(store, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.address
        with urllib.request.urlopen(f"http://{host}:{port}/api/health", timeout=2) as response:
            health = json.loads(response.read())
        with urllib.request.urlopen(f"http://{host}:{port}/api/state", timeout=2) as response:
            state = json.loads(response.read())
        assert health["status"] == "ok"
        assert state["status"] == "ready"
        assert state["research_only"] is True
    finally:
        server.shutdown()
        thread.join(timeout=2)
