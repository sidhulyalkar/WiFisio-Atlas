import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

from physioatlas.research_hub import ResearchHubServer, ResearchHubStore


def test_research_hub_endpoints_and_safe_actions(tmp_path: Path):
    store = ResearchHubStore(tmp_path / "state.json")
    store.update(status="live_test", household={"members": [], "tracks": []})
    server = ResearchHubServer(store, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.address
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/api/health") as response:
            payload = json.load(response)
        assert payload["local_only"]
        with urllib.request.urlopen(f"http://{host}:{port}/api/hub") as response:
            state = json.load(response)
        assert state["status"] == "live_test"

        request = urllib.request.Request(
            f"http://{host}:{port}/api/actions",
            data=json.dumps({"action": "run_calibration"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            queued = json.load(response)
        assert queued["status"] == "queued"
        assert store.action_path.exists()

        bad = urllib.request.Request(
            f"http://{host}:{port}/api/actions",
            data=json.dumps({"action": "run_arbitrary_shell"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(bad)
            assert False, "unsafe action should be rejected"
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
    finally:
        server.shutdown()
        thread.join(timeout=2)
