import subprocess
import sys
import threading

import pytest

from indextts.audition_worker import AuditionWorker


def worker(tmp_path, monkeypatch):
    script = tmp_path/"fake_sidecar.py"
    script.write_text('''import json, sys
waiting = None
for line in sys.stdin:
    r = json.loads(line)
    method = r["method"]
    if method == "synthesize":
        waiting = r["id"]
        continue
    if method == "crash":
        print("simulated model load failure", file=sys.stderr, flush=True)
        sys.exit(1)
    if method == "cancel":
        print(json.dumps({"id": waiting, "ok": False, "error": {"message": "cancelled"}}), flush=True)
    print(json.dumps({"id": r["id"], "ok": True, "result": {"method": method}}), flush=True)
    if method == "shutdown": break
''', encoding="utf-8")
    launch = subprocess.Popen
    monkeypatch.setattr(subprocess, "Popen", lambda args, **kwargs: launch([sys.executable, str(script)], **kwargs))
    return AuditionWorker("unused", "cuda:0", tmp_path/"cache", "all", 4, "native")


def test_jsonl_worker_cancel_and_following_request(tmp_path, monkeypatch):
    client = worker(tmp_path, monkeypatch)
    try:
        assert client.call("health")["method"] == "health"
        cancelled = threading.Event()
        timer = threading.Timer(.2, cancelled.set)
        timer.start()
        with pytest.raises(RuntimeError, match="cancelled|取消"):
            client.call("synthesize", cancelled=cancelled)
        timer.join()
        assert client.call("health")["method"] == "health"
    finally:
        client.close()
    assert client.process.poll() is not None


def test_jsonl_worker_exit_unblocks_pending_request(tmp_path, monkeypatch):
    client = worker(tmp_path, monkeypatch)
    try:
        with pytest.raises(RuntimeError, match="退出|failure"):
            client.call("crash", timeout=2)
    finally:
        client.close()


def test_jsonl_worker_startup_cancellation_stops_process(tmp_path, monkeypatch):
    client = worker(tmp_path, monkeypatch)
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(RuntimeError, match="取消"):
        client.call("health", cancelled=cancelled)
    client.close()
    assert client.process.poll() is not None
