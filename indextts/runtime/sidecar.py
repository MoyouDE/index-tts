"""Port-free JSONL sidecar protocol for Readest."""

from __future__ import annotations

import contextlib
import json
import queue
import sys
import threading
from dataclasses import dataclass
from typing import IO, Any

from .engine import ReaderRuntime, SynthesisCancelled


@dataclass
class _Task:
    request_id: str
    params: dict[str, Any]
    cancelled: threading.Event


class JsonlSidecar:
    def __init__(
        self,
        runtime: ReaderRuntime,
        *,
        stdin: IO[str] = sys.stdin,
        stdout: IO[str] = sys.stdout,
        stderr: IO[str] = sys.stderr,
    ) -> None:
        self.runtime = runtime
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr
        self._write_lock = threading.Lock()
        self._tasks_lock = threading.Lock()
        self._tasks: dict[str, _Task] = {}
        self._queue: queue.Queue[_Task | None] = queue.Queue()
        self._stopping = threading.Event()
        self._worker = threading.Thread(target=self._work, name="indextts-gpu-worker", daemon=True)

    def _write(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self._write_lock:
            self.stdout.write(line + "\n")
            self.stdout.flush()

    def _ok(self, request_id: Any, result: Any) -> None:
        self._write({"id": request_id, "ok": True, "result": result})

    def _error(self, request_id: Any, code: str, message: str, exc: Exception | None = None) -> None:
        error = {"code": code, "message": message}
        if exc is not None:
            error["type"] = type(exc).__name__
        self._write({"id": request_id, "ok": False, "error": error})

    def _work(self) -> None:
        while True:
            task = self._queue.get()
            if task is None:
                self._queue.task_done()
                return
            try:
                if task.cancelled.is_set():
                    raise SynthesisCancelled("排队任务已取消")
                params = task.params
                with contextlib.redirect_stdout(self.stderr):
                    result = self.runtime.synthesize(
                        params["text"],
                        params["voiceId"],
                        params.get("emotion", "base"),
                        params.get("durationFactor", 1.0),
                        _cancelled=task.cancelled.is_set,
                    )
                self._ok(task.request_id, result)
            except SynthesisCancelled as exc:
                self._error(task.request_id, "cancelled", str(exc), exc)
            except Exception as exc:
                self._error(task.request_id, "synthesis_failed", str(exc), exc)
            finally:
                with self._tasks_lock:
                    self._tasks.pop(task.request_id, None)
                self._queue.task_done()

    def _enqueue_synthesis(self, request_id: str, params: Any) -> None:
        if not isinstance(params, dict):
            raise ValueError("params 必须为对象")
        allowed = {"text", "voiceId", "emotion", "durationFactor"}
        unknown = set(params) - allowed
        if unknown:
            raise ValueError(f"synthesize 含不允许的参数: {', '.join(sorted(unknown))}")
        if not isinstance(params.get("text"), str) or not isinstance(params.get("voiceId"), str):
            raise ValueError("synthesize 需要 text 和 voiceId")
        emotion = params.get("emotion", "base")
        if isinstance(emotion, str):
            if emotion not in {"base", "auto"}:
                raise ValueError("emotion 字符串只能是 base 或 auto")
            if emotion == "auto" and self.runtime.emotion_provider is None:
                raise ValueError("未配置自动情感后端，不能使用 emotion=auto")
        elif isinstance(emotion, list):
            if len(emotion) != 8 or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in emotion
            ):
                raise ValueError("emotion 显式向量必须包含 8 个数值")
            if any(not 0.0 <= float(value) <= 1.2 for value in emotion):
                raise ValueError("emotion 显式向量分量必须位于 [0, 1.2]")
        else:
            raise ValueError("emotion 只能是 base、auto 或 8 维显式向量")
        task = _Task(request_id=request_id, params=params, cancelled=threading.Event())
        with self._tasks_lock:
            if request_id in self._tasks:
                raise ValueError(f"请求 id 正在使用: {request_id}")
            self._tasks[request_id] = task
        self._queue.put(task)

    def _cancel(self, params: Any) -> dict[str, Any]:
        if not isinstance(params, dict) or set(params) != {"requestId"}:
            raise ValueError("cancel 只接受 requestId")
        target = params["requestId"]
        with self._tasks_lock:
            task = self._tasks.get(target)
            if task is not None:
                task.cancelled.set()
        return {"requestId": target, "cancelled": task is not None}

    def serve(self) -> int:
        self._worker.start()
        try:
            for line in self.stdin:
                request_id = None
                try:
                    request = json.loads(line)
                    if not isinstance(request, dict):
                        raise ValueError("请求顶层必须为对象")
                    request_id = request.get("id")
                    if not isinstance(request_id, str) or not request_id:
                        raise ValueError("id 必须为非空字符串")
                    if set(request) - {"id", "method", "params"}:
                        raise ValueError("请求含未知顶层字段")
                    method = request.get("method")
                    params = request.get("params", {})
                    if method == "synthesize":
                        self._enqueue_synthesis(request_id, params)
                    elif method == "health":
                        with contextlib.redirect_stdout(self.stderr):
                            self._ok(request_id, self.runtime.health())
                    elif method == "voices.list":
                        self._ok(request_id, {"voices": self.runtime.list_voices()})
                    elif method == "voices.reload":
                        with contextlib.redirect_stdout(self.stderr):
                            voices = self.runtime.reload_voices()
                        self._ok(request_id, {"voices": voices})
                    elif method == "cancel":
                        self._ok(request_id, self._cancel(params))
                    elif method == "shutdown":
                        self._ok(request_id, {"shuttingDown": True})
                        self._stopping.set()
                        with self._tasks_lock:
                            for task in self._tasks.values():
                                task.cancelled.set()
                        break
                    else:
                        self._error(request_id, "method_not_found", f"未知方法: {method}")
                except json.JSONDecodeError as exc:
                    self._error(request_id, "invalid_json", str(exc), exc)
                except Exception as exc:
                    self._error(request_id, "invalid_request", str(exc), exc)
        finally:
            self._queue.put(None)
            self._worker.join(timeout=30)
        return 0
