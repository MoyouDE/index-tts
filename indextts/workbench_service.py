"""Application facade: compose services, GPU scheduling and local persistence."""
import atexit
import copy
from pathlib import Path
import shutil
import tempfile
import threading
from .gpu_coordinator import GpuCoordinator
from .validation_service import ValidationService
from .web_modules import parse_modules


class WorkbenchService:
    def __init__(self, output_dir, workspace_dir, *, modules=None, cpu_threads=4):
        self.modules = parse_modules(modules)
        self.validation = ValidationService(output_dir, cpu_threads=cpu_threads)
        self.output_dir = self.validation.output_dir
        self.gpu = GpuCoordinator()
        self._audition_service = None
        self._service_lock = threading.Lock()
        if "producer" in self.modules or "audition" in self.modules:
            from .voice_workspace import VoiceWorkspace
            self.library = VoiceWorkspace(workspace_dir)
            self._jobs = self.library.root / ".jobs"
            self._jobs.mkdir(exist_ok=True)
        if "producer" in self.modules:
            self.gpu.bind("producer", self.validation.unload_producer)
        if "audition" in self.modules:
            self.gpu.bind("audition", self._release_audition)
        atexit.register(self.close)

    def require(self, name):
        if name not in self.modules:
            raise ValueError(f"功能未启用: {name}")

    @property
    def audition_service(self):
        self.require("audition")
        with self._service_lock:
            if self._audition_service is None:
                from .audition_service import AuditionService
                self._audition_service = AuditionService(self.library.root / "runtime")
            return self._audition_service

    def session_dir(self, session_id):
        return self.validation.session_dir(session_id)

    def close(self):
        self.gpu.close()
        self.validation.unload_emotion()

    def _release_audition(self):
        if self._audition_service is not None:
            self._audition_service.close()

    def unload_audition(self):
        self.gpu.unload("audition")
        return "试听模型已卸载"

    def unload_producer(self):
        self.gpu.unload("producer")
        return "制包模型已卸载"

    def unload_emotion(self):
        return self.validation.unload_emotion()

    def inspect(self, uploaded, model_dir, session_id):
        self.require("producer")
        return self.validation.inspect(uploaded, model_dir, session_id)

    def emotion_default(self, model_dir):
        self.require("emotion")
        return self.validation.emotion_default(model_dir)

    def analyze(self, model_dir, mode, text, kind, rows, target, use_default, threshold, session_id):
        self.require("emotion")
        return self.validation.analyze(model_dir, mode, text, kind, rows, target, use_default, threshold, session_id)

    def build(self, reference, voice_id, name, gender, profile, device, model_dir, session_id, progress=None, *, instance=None):
        self.require("producer")
        if instance is None and (self.library.directory(voice_id) / "voice.json").exists():
            raise ValueError("音色 ID 已存在，请选择库中音色生成；更换参考音频请使用新 ID")
        path = None
        with self.gpu.use("producer"):
            try:
                path, report = self.validation.build(reference, voice_id, name, gender, profile, device, model_dir, session_id, progress)
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
        self.require("producer")
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
        if self._audition_service is None:
            return "当前会话没有进行中的试听"
        return self._audition_service.cancel(session_id)

    def audition(self, voice_id, profile, model_dir, device, text, emotion, speed, seed, settings,
                 optimizations, cpu_threads, allocator, session_id, progress=None):
        self.session_dir(session_id)
        service = self.audition_service
        seed = service.validate_inputs(profile, model_dir, device, text, emotion, speed, seed, settings,
                                       optimizations, cpu_threads, allocator)
        with service.task(session_id) as cancelled, tempfile.TemporaryDirectory(dir=self._jobs) as job:
            snapshot = Path(job) / "voice.ivp"
            instance, pack_hash = self.library.snapshot(voice_id, profile, snapshot)
            settings, emotion = copy.deepcopy(settings), copy.deepcopy(emotion)
            if progress:
                progress(0.1, desc="等待 GPU / 加载裁剪模型")
            with self.gpu.use("audition", cancelled):
                audio, report = service.synthesize(snapshot, voice_id, profile, model_dir, device, text, emotion,
                    speed, seed, settings, optimizations, cpu_threads, allocator, cancelled, progress)
                try:
                    wav, details = self.library.save_preview(voice_id, profile, instance, pack_hash, audio, report)
                finally:
                    Path(audio).unlink(missing_ok=True)
                self.library.save_settings({"models": {**self.library.settings().get("models", {}),
                    profile: str(Path(model_dir).expanduser().resolve())}, "device": device,
                    "cpuThreads": int(cpu_threads), "allocator": allocator})
                if progress:
                    progress(1, desc="试听完成，已保存最近一次结果")
                return wav, details, report
