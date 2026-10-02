"""Reference-only production. No voice-library or audition dependency."""
import gc
import json
import logging
import threading
import time
import uuid
from pathlib import Path
from .validation_common import SessionFiles, audio_details

logger = logging.getLogger(__name__)


class ProducerService:
    def __init__(self, output_dir, *, cpu_threads=4):
        self.files = SessionFiles(output_dir)
        self.cpu_threads = cpu_threads
        self._producer = self._producer_key = None
        self._producer_lock = threading.RLock()

    def _close_model(self):
        if self._producer is not None:
            logger.info("Producer model release: %s", self._producer_key)
            self._producer.close()
            gc.collect()
        self._producer = self._producer_key = None

    def close(self):
        with self._producer_lock:
            self._close_model()
        return "制包模型已卸载"

    def build(self, reference, voice_id, name, gender, profile, device, model_dir, session_id, progress=None, *, reference_selection=None, references=None):
        from .voicepack.schema import validate_identity
        from .voicepack.archive import load_voicepack
        validate_identity(voice_id, name, gender)
        if not reference:
            raise ValueError("请先上传参考音频")
        details = audio_details(reference)
        root = Path(model_dir).expanduser().resolve()
        if not (root / "config.yaml").is_file():
            raise ValueError("源模型目录缺少 config.yaml")
        job = self.files.session_dir(session_id) / uuid.uuid4().hex
        job.mkdir()
        uploaded = self.files.upload(reference, session_id, job / ("reference" + Path(reference).suffix))
        started = time.perf_counter()
        device = None if device == "auto" else device
        key = (str(root), profile, device)
        with self._producer_lock:
            if progress:
                progress(0.1, desc="加载参考编码器并生成音色条件")
            if key != self._producer_key:
                logger.info("Producer model load: profile=%s device=%s threads=%s", profile, device, self.cpu_threads)
                self._close_model()
                import torch
                torch.set_num_threads(self.cpu_threads)
                from .voicepack.builder import VoicePackBuilder
                self._producer = VoicePackBuilder(model_dir=root, profile=profile, device=device)
                self._producer_key = key
            try:
                kwargs = {"reference_selection": reference_selection, "references": references} if reference_selection is not None else {}
                path = self._producer.build(uploaded, {"voiceId": voice_id, "displayName": name, "gender": gender}, job / (voice_id + ".ivp"), **kwargs)
                pack = load_voicepack(path, expected_model_fingerprint=self._producer.source_model_fingerprint())
            except Exception:
                logger.exception("Producer failed: session=%s voice=%s", session_id, voice_id)
                self._close_model()
                raise
        report = {"ok": True, "seconds": time.perf_counter() - started,
                  "audio": details, "manifest": pack.manifest}
        (job / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if progress:
            progress(1, desc="生成完成，完整性与模型指纹检查通过")
        logger.info("Producer completed: session=%s voice=%s seconds=%.3f", session_id, voice_id, report["seconds"])
        return str(path), report

    def inspect(self, uploaded, model_dir, session_id):
        from .voicepack.builder import model_fingerprint
        from .voicepack.archive import load_voicepack
        if not uploaded:
            raise ValueError("请上传 .ivp 音色包")
        job = self.files.session_dir(session_id) / uuid.uuid4().hex
        job.mkdir()
        path = self.files.upload(uploaded, session_id, job / "inspect.ivp")
        expected = model_fingerprint(model_dir) if str(model_dir).strip() else None
        pack = load_voicepack(path, expected_model_fingerprint=expected)
        return {"ok": True, "sourceCompatibilityChecked": expected is not None, "manifest": pack.manifest}
