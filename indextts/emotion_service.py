"""CPU ONNX emotion validation, independent of voice production and audition."""
import gc
import json
import threading
import uuid
from pathlib import Path
from .validation_common import SessionFiles, context_rows


class EmotionService:
    def __init__(self, output_dir):
        self.files = SessionFiles(output_dir)
        self._emotion = self._emotion_key = None
        self._emotion_lock = threading.RLock()

    def _get_emotion(self, model_dir):
        from .runtime.emotion import OnnxEmotionProvider
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
        path = self.files.session_dir(session_id) / ("emotion-" + uuid.uuid4().hex + ".json")
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result, str(path)

    def close(self):
        with self._emotion_lock:
            loaded = self._emotion is not None
            self._emotion = self._emotion_key = None
            if loaded:
                gc.collect()
        return "情感模型已卸载"
