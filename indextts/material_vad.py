"""Streaming CPU Silero v6.0 adapter; no producer or PyTorch dependency."""
from pathlib import Path
import hashlib
import json
import threading

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "voice-producer/models/vad/silero-v6.0.onnx"
SETTINGS = {"threshold": .5, "minSpeechMs": 250, "minSilenceMs": 500, "padMs": 150, "maxSeconds": 15}


def speech_intervals(probabilities, duration, *, step=.032):
    """Hysteresis and silence boundaries, then bounded clips with context."""
    import math
    values = [float(p) for p in probabilities]
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in values):
        raise ValueError("VAD 返回非法概率")
    raw, start, silence = [], None, None
    for i, p in enumerate(values):
        t = i * step
        if start is None:
            if p >= .5:
                start = t
        elif p < .35:
            if silence is None:
                silence = t
            if t + step - silence >= .5:
                if silence - start >= .25:
                    raw.append((start, min(silence, duration)))
                start = silence = None
        else:
            silence = None
    if start is not None:
        end = min(silence if silence is not None else duration, duration)
        if end - start >= .25:
            raw.append((start, end))
    clips = []
    for a, b in raw:
        left, right = max(0, a-.15), min(duration, b+.15)
        if clips and left < clips[-1][1]:
            middle = (left + clips[-1][1]) / 2
            clips[-1] = (clips[-1][0], middle)
            left = middle
        while right-left > 15:
            # Prefer the least speech-like boundary near the length limit.
            low, high = int((left+12)/step), int((left+14.7)/step)
            index = min(range(low, min(high+1, len(values))), key=lambda j: values[j])
            cut = min(left+15, max(left+1, index*step))
            clips.append((left, cut))
            left = cut
        clips.append((left, right))
    return [{"id": f"s{i+1:03d}", "start": round(a, 6), "end": round(b, 6), "selected": True}
            for i, (a, b) in enumerate(clips)], raw


class SileroVad:
    def __init__(self, model_path=DEFAULT_MODEL):
        self.model_path = Path(model_path)
        self._model = None
        self._model_sha = None
        self._lock = threading.RLock()

    def close(self):
        with self._lock:
            self._model = None
            self._model_sha = None

    def detect(self, audio_path, progress=None):
        import numpy as np
        import soundfile as sf
        with self._lock:
            if self._model is None:
                if not self.model_path.is_file():
                    raise FileNotFoundError("缺少 Silero VAD 权重，请按音色工具说明准备模型资产")
                spec = json.loads((ROOT/"voice-producer/assets.lock.json").read_text(encoding="utf-8"))["files"]["vad/silero-v6.0.onnx"]
                if self.model_path.stat().st_size != spec["bytes"] or hashlib.sha256(self.model_path.read_bytes()).hexdigest() != spec["sha256"]:
                    raise ValueError("VAD 权重校验失败")
                import onnxruntime as ort
                options = ort.SessionOptions()
                options.intra_op_num_threads = options.inter_op_num_threads = 1
                self._model = ort.InferenceSession(str(self.model_path), options, providers=["CPUExecutionProvider"])
                self._model_sha = spec["sha256"]
            state = np.zeros((2, 1, 128), dtype=np.float32)
            context = np.zeros((1, 64), dtype=np.float32)
            probabilities = []
            with sf.SoundFile(audio_path) as audio:
                if audio.samplerate != 16000 or audio.channels != 1:
                    raise ValueError("VAD 输入必须为 16kHz 单声道")
                duration = len(audio)/16000
                while True:
                    chunk = audio.read(512, dtype="float32")
                    if not len(chunk):
                        break
                    if not np.isfinite(chunk).all():
                        raise ValueError("音频包含非有限数值")
                    chunk = np.pad(chunk, (0, 512-len(chunk))).reshape(1, 512)
                    value = np.concatenate([context, chunk], axis=1)
                    out, state = self._model.run(None, {"input": value, "state": state, "sr": np.array(16000, dtype=np.int64)})
                    probabilities.append(float(out.reshape(-1)[0]))
                    context = value[:, -64:]
                    if progress and len(probabilities) % 100 == 0:
                        progress(min(.9, audio.tell()/len(audio)*.9), desc="CPU 人声检测")
            segments, raw = speech_intervals(probabilities, duration)
            return {"segments": segments, "speechIntervals": raw, "speechSeconds": sum(b-a for a,b in raw),
                    "settings": dict(SETTINGS), "modelSha256": self._model_sha}
