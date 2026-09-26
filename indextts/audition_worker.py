"""Managed JSONL reader process. No generation networks are imported by the UI."""
import json
import contextlib
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import uuid


class AuditionWorker:
    def __init__(self, model_dir, device, cache_dir, optimizations, cpu_threads, allocator):
        cache = Path(cache_dir).resolve()
        cache.mkdir(parents=True, exist_ok=True)
        empty = cache / "empty-voices"
        empty.mkdir(exist_ok=True)
        self._pending = {}
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._log = (cache / "worker.log").open("a", encoding="utf-8")
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        command = [sys.executable, "-m", "indextts.runtime.cli", "serve", "--model-dir", str(model_dir),
                   "--voice-dir", str(empty), "--cache-dir", str(cache), "--device", device,
                   "--emotion-backend", "none", "--optimizations", optimizations,
                   "--cpu-threads", str(cpu_threads), "--allocator", allocator]
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self._log, text=True, encoding="utf-8", env=env,
                cwd=Path(__file__).resolve().parents[1],
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except Exception:
            self._log.close()
            raise
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                try:
                    response = json.loads(line)
                    with self._lock:
                        pending = self._pending.get(response.get("id"))
                    if pending:
                        pending.put(response)
                except (ValueError, AttributeError):
                    continue
        finally:
            with self._lock:
                for pending in self._pending.values():
                    pending.put({"ok": False, "error": {"message": "试听进程已退出，请检查模型目录或 worker.log 后重试"}})

    def _send(self, request_id, method, params):
        with self._write_lock:
            self.process.stdin.write(json.dumps({"id": request_id, "method": method, "params": params}, ensure_ascii=False) + "\n")
            self.process.stdin.flush()

    def failure(self):
        self._log.flush()
        tail = Path(self._log.name).read_text(encoding="utf-8", errors="replace")[-2500:]
        return RuntimeError("试听进程已退出，可以重试。后端日志：\n" + tail)

    def call(self, method, params=None, *, cancelled=None, timeout=600):
        request_id = uuid.uuid4().hex
        replies = queue.Queue()
        with self._lock:
            self._pending[request_id] = replies
        started = time.monotonic()
        cancel_sent = False
        cancel_started = None
        try:
            if self.process.poll() is not None:
                raise self.failure()
            self._send(request_id, method, params or {})
            while True:
                if cancelled is not None and cancelled.is_set() and not cancel_sent:
                    if method != "synthesize":
                        self.close()
                        raise RuntimeError("试听已取消")
                    self._send(uuid.uuid4().hex, "cancel", {"requestId": request_id})
                    cancel_sent, cancel_started = True, time.monotonic()
                if time.monotonic() - started > timeout or (cancel_started and time.monotonic() - cancel_started > 30):
                    self.close()
                    raise RuntimeError("试听已取消" if cancel_sent else "试听超时，进程已卸载；可以重试")
                try:
                    response = replies.get(timeout=0.2)
                except queue.Empty:
                    if self.process.poll() is not None:
                        raise self.failure()
                    continue
                if not response.get("ok"):
                    if self.process.poll() is not None:
                        raise self.failure()
                    raise RuntimeError(response["error"]["message"])
                if cancel_sent:
                    raise RuntimeError("试听已取消")
                return response["result"]
        finally:
            with self._lock:
                self._pending.pop(request_id, None)

    def close(self):
        if self.process.poll() is None:
            try:
                self._send(uuid.uuid4().hex, "shutdown", {})
                self.process.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                if os.name == "nt" and self.process.poll() is None:
                    # Windows venv launchers may own a second Python process.
                    # Terminate our process tree, not just its launcher.
                    subprocess.run(["taskkill", "/PID", str(self.process.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   creationflags=subprocess.CREATE_NO_WINDOW, timeout=10)
                elif self.process.poll() is None:
                    self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
        self._reader.join(timeout=1)
        with contextlib.suppress(OSError):
            self.process.stdin.close()
        with contextlib.suppress(OSError):
            self.process.stdout.close()
        self._log.close()
