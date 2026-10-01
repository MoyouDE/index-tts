"""Reader subprocess lifecycle and synthesis, independent of the voice library."""
import copy
import json
import math
import logging
from pathlib import Path
import secrets
import threading
import time
from contextlib import contextmanager
from .audition_worker import AuditionWorker
from .runtime.profiles import BF16, FP32, BF16_RUNTIME_ABI, InferenceOptimizations, synthesis_settings

logger = logging.getLogger(__name__)


class AuditionService:
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self._worker = self._worker_key = None
        self._model_lock = threading.RLock()
        self._tasks = {}
        self._tasks_lock = threading.Lock()

    def close(self):
        with self._model_lock:
            self._close_worker()

    def _close_worker(self):
        if self._worker is not None:
            logger.info("Audition worker release")
            self._worker.close()
        self._worker = self._worker_key = None
        for pattern in ("tts-*.wav", ".*.wav.part"):
            for path in self.cache_dir.glob(pattern):
                path.unlink(missing_ok=True)

    def cancel(self, session_id):
        with self._tasks_lock:
            event = self._tasks.get(session_id)
            if event:
                event.set()
                logger.info("Audition cancellation requested: session=%s", session_id)
        return "已请求取消；当前阶段结束后停止，必要时释放进程" if event else "当前会话没有进行中的试听"

    @contextmanager
    def task(self, session_id):
        cancelled = threading.Event()
        with self._tasks_lock:
            if session_id in self._tasks:
                raise ValueError("当前会话已有试听任务")
            self._tasks[session_id] = cancelled
        try:
            yield cancelled
        finally:
            with self._tasks_lock:
                self._tasks.pop(session_id, None)

    def has_task(self, session_id):
        with self._tasks_lock:
            return session_id in self._tasks

    @staticmethod
    def validate_inputs(profile, model_dir, device, text, emotion, speed, seed, settings, optimizations, cpu_threads, allocator):
        if not str(text).strip():
            raise ValueError("试听文本不能为空")
        if emotion != "base":
            if not isinstance(emotion, list) or len(emotion) != 8 or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1.2 for v in emotion
            ):
                raise ValueError("情感向量必须包含 8 个 [0, 1.2] 范围内的数值")
        if not str(model_dir).strip():
            raise ValueError("请选择已有的裁剪模型目录")
        if not 0.5 <= float(speed) <= 2:
            raise ValueError("语速倍率必须位于 [0.5, 2.0]")
        if int(cpu_threads) != cpu_threads or not 1 <= cpu_threads <= 64:
            raise ValueError("CPU 线程数必须为 1 到 64 的整数")
        if allocator not in {"native", "cudaMallocAsync"} or not str(device).startswith("cuda"):
            raise ValueError("试听需要 CUDA 设备及受支持的分配器")
        InferenceOptimizations.parse(optimizations)
        synthesis_settings(profile, settings)
        if seed is None:
            seed = secrets.randbits(32)
        if isinstance(seed, bool) or int(seed) != seed or not 0 <= seed <= 2**32 - 1:
            raise ValueError("随机种子必须为 0 到 2^32-1 的整数")
        seed = int(seed)
        return seed

    def synthesize(self, snapshot, voice_id, profile, model_dir, device, text, emotion, speed, seed, settings,
                   optimizations, cpu_threads, allocator, cancelled, progress=None):
        seed = self.validate_inputs(profile, model_dir, device, text, emotion, speed, seed, settings,
                                    optimizations, cpu_threads, allocator)
        with self._model_lock:
            return self._synthesize(snapshot, voice_id, profile, model_dir, device, text, emotion, speed, seed,
                                    settings, optimizations, cpu_threads, allocator, cancelled, progress)

    def _synthesize(self, snapshot, voice_id, profile, model_dir, device, text, emotion, speed, seed, settings,
                    optimizations, cpu_threads, allocator, cancelled, progress):
        from .voicepack.archive import load_voicepack
        from .voicepack.provenance import sha256_file
        settings, emotion = copy.deepcopy(settings), copy.deepcopy(emotion)
        root = Path(model_dir).expanduser().resolve()
        manifest_path = root / "runtime_model.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual_profile = BF16 if manifest.get("runtimeAbi") == BF16_RUNTIME_ABI else FP32
        if actual_profile != profile:
            raise ValueError("模型精度与选择的音色包精度不匹配")
        load_voicepack(snapshot, expected_model_fingerprint=manifest["sourceModelFingerprint"])
        key = (str(root), manifest_path.stat().st_mtime_ns, profile, device, optimizations, int(cpu_threads), allocator)
        if progress:
            progress(0.1, desc="等待 GPU / 加载裁剪模型")
        if cancelled.is_set():
            raise RuntimeError("试听已取消")
        started = time.perf_counter()
        try:
            reused = key == self._worker_key and self._worker is not None and self._worker.process.poll() is None
            logger.info("Audition worker: reused=%s profile=%s device=%s", reused, profile, device)
            if not reused:
                self.close()
                self._worker = AuditionWorker(root, device, self.cache_dir, optimizations, int(cpu_threads), allocator)
                self._worker.call("health", cancelled=cancelled)
                self._worker_key = key
            load_ms = (time.perf_counter() - started) * 1000 if not reused else 0
            self._worker.call("voices.load", {"path": str(snapshot)}, cancelled=cancelled)
            if progress:
                progress(0.3, desc="合成试听音频")
            result = self._worker.call("synthesize", {"text": text, "voiceId": voice_id,
                "emotion": emotion, "durationFactor": 1 / float(speed), "seed": seed,
                "generationSettings": settings}, cancelled=cancelled)
            health = self._worker.call("health", cancelled=cancelled)
        except Exception:
            logger.exception("Audition failed: voice=%s profile=%s", voice_id, profile)
            self.close()
            raise
        audio = Path(result["audioPath"])
        report = {"voiceId": voice_id, "profile": profile, "packSha256": sha256_file(snapshot),
                  "sourceModelFingerprint": manifest["sourceModelFingerprint"],
                  "runtimeManifestSha256": sha256_file(manifest_path),
                  "text": text, "emotion": emotion, "speed": speed, "seed": seed,
                  "modelLoadMs": round(load_ms, 2), "modelReused": reused,
                  "device": device, "cpuThreads": int(cpu_threads), "allocator": allocator,
                  "optimizations": optimizations, "result": {k: v for k, v in result.items() if k != "audioPath"},
                  "memory": {key: value for key, value in health.items() if "Memory" in key},
                  "memoryScope": "PyTorch 进程当前分配/保留及进程生命周期峰值；不是 WDDM dedicated 或系统总内存"}
        return str(audio), report
