"""Streaming CPU Silero v6.0 adapter; no producer or PyTorch dependency."""
from pathlib import Path
import hashlib
import json
import math
import threading
from collections import OrderedDict

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "voice-producer/models/vad/silero-v6.0.onnx"
SETTINGS = {"threshold": .5, "minSpeechMs": 250, "minSilenceMs": 500, "padMs": 150, "maxSeconds": 15}


def detection_settings(min_silence_ms=500, min_volume_db=None):
    if (isinstance(min_silence_ms, bool) or not isinstance(min_silence_ms, (int, float))
            or not math.isfinite(min_silence_ms) or not 100 <= min_silence_ms <= 3000):
        raise ValueError("空白间隔须为 0.1～3 秒")
    if min_volume_db is not None and (isinstance(min_volume_db, bool)
            or not isinstance(min_volume_db, (int, float)) or not math.isfinite(min_volume_db)
            or not -80 <= min_volume_db <= 0):
        raise ValueError("最低音量须为 -80～0 dBFS")
    return {**SETTINGS, "minSilenceMs": min_silence_ms, "minVolumeDb": min_volume_db}


def speech_intervals(probabilities, duration, *, step=.032, min_silence_ms=500):
    """Hysteresis and silence boundaries, then bounded clips with context."""
    silence_seconds = detection_settings(min_silence_ms)["minSilenceMs"] / 1000
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
            if t + step - silence >= silence_seconds - 1e-9:
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
        self._analyses = OrderedDict()

    def close(self):
        with self._lock:
            self._model = None
            self._model_sha = None
            self._analyses.clear()

    def detect(self, audio_path, progress=None, *, min_silence_ms=500, min_volume_db=None):
        settings = detection_settings(min_silence_ms, min_volume_db)
        volume_floor = 10 ** (min_volume_db / 20) if min_volume_db is not None else None
        analysis = self.analyze(audio_path, progress)
        probabilities = [p if volume_floor is None or rms >= volume_floor else 0.
                         for p, rms in zip(analysis['probabilities'], analysis['rms'])]
        segments, raw = speech_intervals(probabilities, analysis['duration'], min_silence_ms=min_silence_ms)
        return {"segments": segments, "speechIntervals": raw, "speechSeconds": sum(b-a for a,b in raw),
                "settings": settings, "modelSha256": analysis['modelSha256']}

    def analyze(self, audio_path, progress=None):
        """Cache parameter-independent model scores; sliders never rerun inference."""
        import numpy as np
        import soundfile as sf
        audio_path = Path(audio_path)
        with self._lock:
            spec = json.loads((ROOT/"voice-producer/assets.lock.json").read_text(encoding="utf-8"))["files"]["vad/silero-v6.0.onnx"]
            stamp = audio_path.stat()
            signature = {"version": 1, "bytes": stamp.st_size, "mtimeNs": stamp.st_mtime_ns,
                         "modelSha256": self._model_sha or spec['sha256']}
            key = (str(audio_path.resolve()), *signature.values())
            if key in self._analyses:
                self._analyses.move_to_end(key)
                return self._analyses[key]
            cache = audio_path.with_name('vad-analysis-v1.json')
            try:
                value = json.loads(cache.read_text(encoding='utf-8'))
                if (value['signature'] == signature and value['modelSha256'] == signature['modelSha256']
                        and math.isfinite(value['duration']) and value['duration'] > 0
                        and len(value['probabilities']) == len(value['rms']) == math.ceil(value['duration']*16000/512 - 1e-9)
                        and all(math.isfinite(p) and 0 <= p <= 1 for p in value['probabilities'])
                        and all(math.isfinite(r) and r >= 0 for r in value['rms'])):
                    self._remember(key, value)
                    return value
            except (OSError, ValueError, KeyError, TypeError):
                pass
            if self._model is None:
                if not self.model_path.is_file():
                    raise FileNotFoundError("缺少 Silero VAD 权重，请按音色工具说明准备模型资产")
                if self.model_path.stat().st_size != spec["bytes"] or hashlib.sha256(self.model_path.read_bytes()).hexdigest() != spec["sha256"]:
                    raise ValueError("VAD 权重校验失败")
                import onnxruntime as ort
                options = ort.SessionOptions()
                options.intra_op_num_threads = options.inter_op_num_threads = 1
                self._model = ort.InferenceSession(str(self.model_path), options, providers=["CPUExecutionProvider"])
                self._model_sha = spec["sha256"]
            state = np.zeros((2, 1, 128), dtype=np.float32)
            context = np.zeros((1, 64), dtype=np.float32)
            probabilities, levels = [], []
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
                    # RMS is measured on original samples, before padding the final frame.
                    levels.append(float(np.sqrt(np.square(chunk, dtype=np.float64).mean())))
                    chunk = np.pad(chunk, (0, 512-len(chunk))).reshape(1, 512)
                    value = np.concatenate([context, chunk], axis=1)
                    out, state = self._model.run(None, {"input": value, "state": state, "sr": np.array(16000, dtype=np.int64)})
                    probability = float(out.reshape(-1)[0])
                    if not math.isfinite(probability) or not 0 <= probability <= 1:
                        raise ValueError("VAD 返回非法概率")
                    probabilities.append(probability)
                    context = value[:, -64:]
                    if progress and len(probabilities) % 100 == 0:
                        progress(min(.9, audio.tell()/len(audio)*.9), desc="CPU 人声检测")
            value = {"signature": signature, "duration": duration, "probabilities": probabilities,
                     "rms": levels, "modelSha256": self._model_sha}
            from .material_service import write_json
            write_json(cache, value)
            self._remember(key, value)
            return value

    def _remember(self, key, value):
        self._analyses[key] = value
        while len(self._analyses) > 2:
            self._analyses.popitem(last=False)
