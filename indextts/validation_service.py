"""Local validation operations, independent of the Gradio presentation layer."""
import gc
import json
from pathlib import Path
import re
import shutil
import threading
import time
import uuid

from indextts.runtime.emotion import OnnxEmotionProvider
from indextts.voicepack.builder import VoicePackBuilder, model_fingerprint
from indextts.voicepack.archive import load_voicepack
from indextts.voicepack.schema import validate_identity


def audio_details(path):
    import soundfile as sf
    info = sf.info(path)
    if info.frames <= 0 or info.samplerate <= 0:
        raise ValueError("参考音频为空")
    return {"durationSeconds": info.duration, "sampleRate": info.samplerate,
            "channels": info.channels, "usedSeconds": min(info.duration, 15),
            "truncated": info.duration > 15}


def context_rows(rows):
    sentences = []
    for row in rows or []:
        if len(row) != 5:
            raise ValueError("每行需要句子 ID、章节、段落、类型、正文五列")
        sid, section, line, kind, text = row
        if not any(str(x or "").strip() for x in row):
            continue
        try:
            if isinstance(line, bool):
                raise ValueError()
            number = float(line)
            if not number.is_integer() or number < 0:
                raise ValueError()
        except (ValueError, TypeError):
            raise ValueError("段落编号必须为非负整数") from None
        kind = {"对白": "dialogue", "旁白": "narration"}.get(kind, kind)
        sentences.append({"sentenceId": str(sid or "").strip(), "sectionId": str(section or "").strip(),
                          "lineIndex": int(number), "sentenceType": kind, "text": str(text or "").strip()})
    return sentences


class ValidationService:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir).resolve()
        self._producer = None
        self._producer_key = None
        self._producer_lock = threading.Lock()
        self._emotion = None
        self._emotion_key = None
        self._emotion_lock = threading.Lock()

    def session_dir(self, session_id):
        if not isinstance(session_id, str) or not re.fullmatch(r"[0-9a-f]{32}", session_id):
            raise ValueError("无效会话")
        path = (self.output_dir / session_id).resolve()
        if not path.is_relative_to(self.output_dir):
            raise ValueError("无效会话目录")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _upload(self, path, session_id, destination):
        source = Path(path).resolve()
        # Results from another server-side session must not be reused as inputs.
        if source.is_relative_to(self.output_dir) and not source.is_relative_to(self.session_dir(session_id)):
            raise ValueError("不能访问其他会话的文件")
        shutil.copyfile(source, destination)
        return destination

    def _close_producer(self):
        if self._producer is not None:
            self._producer.close()
        self._producer = self._producer_key = None
        gc.collect()

    def unload_producer(self):
        with self._producer_lock:
            self._close_producer()
        return "制包模型已卸载"

    def build(self, reference, voice_id, name, gender, profile, device, model_dir, session_id, progress=None):
        validate_identity(voice_id, name, gender)
        if not reference:
            raise ValueError("请先上传参考音频")
        details = audio_details(reference)
        root = Path(model_dir).expanduser().resolve()
        if not (root / "config.yaml").is_file():
            raise ValueError("源模型目录缺少 config.yaml")
        job = self.session_dir(session_id) / uuid.uuid4().hex
        job.mkdir()
        uploaded = self._upload(reference, session_id, job / ("reference" + Path(reference).suffix))
        started = time.perf_counter()
        device = None if device == "auto" else device
        key = (str(root), profile, device)
        with self._producer_lock:
            if progress:
                progress(0.1, desc="加载参考编码器并生成音色条件")
            if key != self._producer_key:
                self._close_producer()
                self._producer = VoicePackBuilder(model_dir=root, profile=profile, device=device)
                self._producer_key = key
            try:
                path = self._producer.build(uploaded, {"voiceId": voice_id, "displayName": name, "gender": gender}, job / (voice_id + ".ivp"))
                pack = load_voicepack(path, expected_model_fingerprint=self._producer.source_model_fingerprint())
            except Exception:
                self._close_producer()
                raise
        report = {"ok": True, "seconds": time.perf_counter() - started,
                  "audio": details, "manifest": pack.manifest}
        (job / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if progress:
            progress(1, desc="生成完成，完整性与模型指纹检查通过")
        return str(path), report

    def inspect(self, uploaded, model_dir, session_id):
        if not uploaded:
            raise ValueError("请上传 .ivp 音色包")
        job = self.session_dir(session_id) / uuid.uuid4().hex
        job.mkdir()
        path = self._upload(uploaded, session_id, job / "inspect.ivp")
        expected = model_fingerprint(model_dir) if str(model_dir).strip() else None
        pack = load_voicepack(path, expected_model_fingerprint=expected)
        return {"ok": True, "sourceCompatibilityChecked": expected is not None, "manifest": pack.manifest}

    def _get_emotion(self, model_dir):
        if not str(model_dir).strip():
            raise ValueError("请指定包含 emotion_model.json 的情感模型目录")
        root = Path(model_dir).expanduser().resolve()
        manifest = root / "emotion_model.json"
        key = (str(root), manifest.stat().st_mtime_ns)
        if key != self._emotion_key:
            self._emotion = self._emotion_key = None
            provider = OnnxEmotionProvider(root)
            provider._load()
            self._emotion, self._emotion_key = provider, key
        return self._emotion

    def emotion_default(self, model_dir):
        with self._emotion_lock:
            model = self._get_emotion(model_dir)
            return model.model_threshold, model.model_manifest

    def analyze(self, model_dir, mode, text, kind, rows, target, use_default, threshold, session_id):
        if mode == "单句":
            sentences = [{"sentenceId": "target", "sectionId": "single", "lineIndex": 0,
                          "sentenceType": kind, "text": text}]
            target = "target"
        else:
            sentences = context_rows(rows)
        with self._emotion_lock:
            provider = self._get_emotion(model_dir)
            result = provider.analyze_window_details(sentences, target,
                neutral_threshold=provider.model_threshold if use_default else threshold)
            result["labels"] = provider.model_manifest["labels"]
            result["modelVersion"] = provider.model_manifest["version"]
            result["sourceCheckpointSha256"] = provider.model_manifest.get("sourceCheckpointSha256")
            result["modelFileSha256"] = provider.model_manifest["files"]["emotion.onnx"]["sha256"]
            result["modelThreshold"] = provider.model_threshold
            result["input"] = {"sentences": sentences, "targetSentenceId": target}
        path = self.session_dir(session_id) / ("emotion-" + uuid.uuid4().hex + ".json")
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result, str(path)

    def unload_emotion(self):
        with self._emotion_lock:
            self._emotion = self._emotion_key = None
            gc.collect()
        return "情感模型已卸载"
