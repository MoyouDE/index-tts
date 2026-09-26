"""Voice library and audition orchestration for the local validation UI."""
import atexit
import copy
import json
import math
from pathlib import Path
import secrets
import shutil
import tempfile
import threading
import time

from .audition_worker import AuditionWorker
from .runtime.profiles import BF16, FP32, BF16_RUNTIME_ABI, InferenceOptimizations, synthesis_settings
from .validation_service import ValidationService
from .voice_workspace import VoiceWorkspace
from .voicepack.archive import load_voicepack
from .voicepack.provenance import sha256_file


class WorkbenchService(ValidationService):
    def __init__(self, output_dir, workspace_dir):
        super().__init__(output_dir)
        self.library = VoiceWorkspace(workspace_dir)
        self._gpu_lock = threading.RLock()
        self._worker = self._worker_key = None
        self._tasks = {}
        self._tasks_lock = threading.Lock()
        self._jobs = self.library.root / ".jobs"
        self._jobs.mkdir(exist_ok=True)
        atexit.register(self.close)

    def _close_worker(self):
        if self._worker is not None:
            self._worker.close()
        self._worker = self._worker_key = None
        for pattern in ("tts-*.wav", ".*.wav.part"):
            for path in (self.library.root / "runtime").glob(pattern):
                path.unlink(missing_ok=True)

    def close(self):
        self._close_worker()
        self._close_producer()

    def unload_audition(self):
        with self._gpu_lock:
            self._close_worker()
        return "试听模型已卸载"

    def unload_producer(self):
        with self._gpu_lock:
            return super().unload_producer()

    def build(self, reference, voice_id, name, gender, profile, device, model_dir, session_id, progress=None, *, instance=None):
        if instance is None and (self.library.directory(voice_id) / "voice.json").exists():
            raise ValueError("音色 ID 已存在，请选择库中音色生成；更换参考音频请使用新 ID")
        path = None
        with self._gpu_lock:
            self._close_worker()
            try:
                path, report = super().build(reference, voice_id, name, gender, profile, device, model_dir, session_id, progress)
                copied_reference = next(Path(path).parent.glob("reference.*"))
                installed = self.library.install(path, reference=copied_reference, expected_instance=instance)
                return installed, report
            finally:
                if path:
                    parent = Path(path).resolve().parent
                    if not parent.is_relative_to(self.output_dir):
                        raise ValueError("任务输出路径越界")
                    shutil.rmtree(parent)

    def rebuild(self, voice_id, profile, device, model_dir, session_id, progress=None):
        with tempfile.TemporaryDirectory(dir=self._jobs) as job:
            with self.library.lock:
                record = self.library.record(voice_id)
                if not record["reference"]:
                    raise ValueError("导入包没有参考音频，无法补生成另一精度")
                source = (self.library.directory(voice_id) / record["reference"]).resolve()
                if source.parent != self.library.directory(voice_id):
                    raise ValueError("参考音频路径无效")
                reference = Path(job) / source.name
                shutil.copyfile(source, reference)
                record = copy.deepcopy(record)
            return self.build(reference, voice_id, record["displayName"], record["gender"], profile,
                              device, model_dir, session_id, progress, instance=record["instance"])

    def cancel(self, session_id):
        with self._tasks_lock:
            event = self._tasks.get(session_id)
            if event:
                event.set()
        return "已请求取消；当前阶段结束后停止，必要时释放进程" if event else "当前会话没有进行中的试听"

    def audition(self, voice_id, profile, model_dir, device, text, emotion, speed, seed, settings,
                 optimizations, cpu_threads, allocator, session_id, progress=None):
        self.session_dir(session_id)  # validate the server-generated session identifier
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
        cancelled = threading.Event()
        with self._tasks_lock:
            if session_id in self._tasks:
                raise ValueError("当前会话已有试听任务")
            self._tasks[session_id] = cancelled
        try:
            with tempfile.TemporaryDirectory(dir=self._jobs) as job:
                snapshot = Path(job) / "voice.ivp"
                instance, pack_hash = self.library.snapshot(voice_id, profile, snapshot)
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
                with self._gpu_lock:
                    if cancelled.is_set():
                        raise RuntimeError("试听已取消")
                    self._close_producer()
                    started = time.perf_counter()
                    try:
                        reused = key == self._worker_key and self._worker is not None and self._worker.process.poll() is None
                        if not reused:
                            self._close_worker()
                            self._worker = AuditionWorker(root, device, self.library.root / "runtime", optimizations, int(cpu_threads), allocator)
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
                        self._close_worker()
                        raise
                    audio = Path(result["audioPath"])
                    report = {"voiceId": voice_id, "profile": profile, "packSha256": pack_hash,
                              "sourceModelFingerprint": manifest["sourceModelFingerprint"],
                              "runtimeManifestSha256": sha256_file(manifest_path),
                              "text": text, "emotion": emotion, "speed": speed, "seed": seed,
                              "modelLoadMs": round(load_ms, 2), "modelReused": reused,
                              "device": device, "cpuThreads": int(cpu_threads), "allocator": allocator,
                              "optimizations": optimizations, "result": {k: v for k, v in result.items() if k != "audioPath"},
                              "memory": {key: value for key, value in health.items() if "Memory" in key},
                              "memoryScope": "PyTorch 进程当前分配/保留及进程生命周期峰值；不是 WDDM dedicated 或系统总内存"}
                    try:
                        wav, details = self.library.save_preview(voice_id, profile, instance, pack_hash, audio, report)
                    finally:
                        audio.unlink(missing_ok=True)
                    self.library.save_settings({"models": {**self.library.settings().get("models", {}), profile: str(root)},
                                                "device": device, "cpuThreads": int(cpu_threads), "allocator": allocator})
                    if progress:
                        progress(1, desc="试听完成，已保存最近一次结果")
                    return wav, details, report
        finally:
            with self._tasks_lock:
                self._tasks.pop(session_id, None)
