"""Persistent material editing and immutable reference selections, CPU only."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import uuid


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def ffmpeg_executable():
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError("缺少 FFmpeg，请安装 validation-web extra 或在 PATH 提供 ffmpeg") from exc


class MaterialService:
    def __init__(self, workspace, *, detector=None):
        self.root = Path(workspace).resolve()/"materials"
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self._detector = detector

    def close(self):
        if self._detector is not None:
            self._detector.close()

    def directory(self, source_id):
        if not isinstance(source_id, str) or not re.fullmatch(r"[0-9a-f]{32}", source_id):
            raise ValueError("非法素材 ID")
        directory = (self.root/source_id).resolve()
        if directory.parent != self.root:
            raise ValueError("素材路径越界")
        return directory

    def record(self, source_id):
        return json.loads((self.directory(source_id)/"material.json").read_text(encoding="utf-8"))

    def items(self):
        with self.lock:
            return [self.record(p.parent.name) for p in sorted(self.root.glob("*/material.json"))
                    if re.fullmatch(r"[0-9a-f]{32}",p.parent.name)]

    def waveform(self, source_id):
        """Bounded display peaks, streamed on CPU; never alter reference samples."""
        import soundfile as sf
        import numpy as np
        directory = self.directory(source_id)
        record = self.record(source_id)
        cache = directory/"waveform-v1.json"
        with self.lock:
            if cache.exists():
                try:
                    value = json.loads(cache.read_text(encoding="utf-8"))
                    if value["audioSha256"] == record["audioSha256"] and value["version"] == 1:
                        return value
                except (ValueError, KeyError):
                    pass
            low, high = [], []
            with sf.SoundFile(directory/"audio.wav") as audio:
                if not audio.frames:
                    raise ValueError("音轨为空")
                span = max(1, math.ceil(audio.frames/12000))
                duration = audio.frames/audio.samplerate
                while True:
                    samples = audio.read(span*256, dtype="float32", always_2d=True)
                    if not len(samples):
                        break
                    samples = samples.mean(axis=1)
                    if not np.isfinite(samples).all():
                        raise ValueError("音轨含非法数值，无法显示波形")
                    samples = np.pad(samples, (0, (-len(samples)) % span)).reshape(-1, span)
                    low.extend(np.round(samples.min(axis=1), 5).tolist())
                    high.extend(np.round(samples.max(axis=1), 5).tolist())
            value = {"version": 1, "audioSha256": record["audioSha256"], "duration": duration,
                     "low": low, "high": high}
            write_json(cache, value)
            return value

    def suggest_segments(self, source_id, progress=None):
        """Return a fresh editing draft; never overwrite an existing selection."""
        audio = self.directory(source_id)/"vad.wav"
        if not audio.is_file():
            raise ValueError("素材缺少 VAD 音轨，请重新导入素材")
        with self.lock:
            if self._detector is None:
                from .material_vad import SileroVad
                self._detector = SileroVad()
            detector = self._detector
        return detector.detect(audio, progress)

    def import_media(self, uploaded, progress=None):
        if not uploaded or not Path(uploaded).is_file():
            raise ValueError("请上传本地视频或音频")
        executable = ffmpeg_executable()
        source_id = uuid.uuid4().hex
        staging = self.root/(".import-"+source_id)
        staging.mkdir()
        try:
            original = staging/("original"+Path(uploaded).suffix.lower())
            shutil.copyfile(uploaded, original)
            if progress:
                progress(.05, desc="FFmpeg 提取音轨")
            for name, rate in [("audio.wav", 22050), ("vad.wav", 16000)]:
                run = subprocess.run([executable, "-nostdin", "-v", "error", "-i", str(original), "-map", "0:a:0",
                    "-vn", "-ac", "1", "-ar", str(rate), "-c:a", "pcm_f32le", str(staging/name)],
                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
                if run.returncode:
                    raise ValueError("FFmpeg 提取失败：" + run.stderr[-1500:])
            import soundfile as sf
            details = sf.info(staging/"audio.wav")
            if not details.frames:
                raise ValueError("音轨为空")
            if self._detector is None:
                from .material_vad import SileroVad
                self._detector = SileroVad()
            detected = self._detector.detect(staging/"vad.wav", progress)
            from .voicepack.provenance import sha256_file
            record = {"sourceId": source_id, "name": Path(uploaded).name, "original": original.name,
                      "sourceSha256": sha256_file(original), "audioSha256": sha256_file(staging/"audio.wav"),
                      "durationSeconds": details.duration, "sampleRate": 22050, **detected,
                      "primary": max(detected["segments"],key=lambda s:s["end"]-s["start"])["id"] if detected["segments"] else None}
            record["revision"] = self.selection_revision(record)
            write_json(staging/"material.json", record)
            with self.lock:
                staging.rename(self.directory(source_id))
            return record
        finally:
            if staging.exists():
                resolved=staging.resolve()
                if resolved.parent != self.root or not resolved.name.startswith(".import-"):
                    raise ValueError("拒绝清理工作区外的导入目录")
                shutil.rmtree(resolved)

    @staticmethod
    def selection_revision(record):
        return digest_json({k: record[k] for k in ("sourceId", "sourceSha256", "audioSha256", "segments", "primary")})

    def save(self, source_id, rows, primary, *, expected_revision):
        with self.lock:
            record = self.record(source_id)
            if record["revision"] != expected_revision:
                raise ValueError("素材已在其他页面修改，请刷新后重试")
            if hasattr(rows, "values"):
                rows = rows.values.tolist()
            segments, ids = [], set()
            for row in rows:
                if len(row) != 4:
                    raise ValueError("片段需要 ID、起点、终点、是否选中")
                sid, a, b, selected = row
                if not isinstance(sid, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", sid) or sid in ids:
                    raise ValueError("片段 ID 非法或重复")
                if isinstance(a, bool) or isinstance(b, bool):
                    raise ValueError("片段边界必须为秒数")
                a, b = float(a), float(b)
                if not math.isfinite(a+b) or not 0 <= a < b <= record["durationSeconds"]:
                    raise ValueError("片段边界必须在音轨范围内且起点小于终点")
                if not isinstance(selected, bool):
                    raise ValueError("选中列必须为布尔值")
                segments.append({"id": sid, "start": a, "end": b, "selected": selected})
                ids.add(sid)
            segments.sort(key=lambda s: (s["start"],s["end"]))
            if primary not in ids or not next(s for s in segments if s["id"]==primary)["selected"]:
                raise ValueError("主参考必须是选中的片段")
            chosen = [s for s in segments if s["selected"]]
            if any(a["end"] > b["start"]+1e-6 for a,b in zip(chosen, chosen[1:])):
                raise ValueError("选中片段重叠，请调整边界，避免重复利用素材")
            record.update(segments=segments, primary=primary)
            record["revision"] = self.selection_revision(record)
            write_json(self.directory(source_id)/"material.json",record)
            return record

    def selection(self, source_id, revision, *, confirmed=False):
        if not confirmed:
            raise ValueError("请确认选中片段属于同一目标人物")
        with self.lock:
            record = self.record(source_id)
            if record["revision"] != revision:
                raise ValueError("选择记录已变更，请重新载入")
            chosen = [s for s in record["segments"] if s["selected"]]
            if not chosen or any(s["end"]-s["start"] > 15+1e-6 for s in chosen):
                raise ValueError("选中片段不能为空且每段不得超过 15 秒，请先拆分")
            return copy.deepcopy({"sourceId": source_id, "sourceSha256": record["sourceSha256"],
                "audioSha256": record["audioSha256"], "revision": revision, "segments": chosen,
                "primary": record["primary"], "method": "primary-only-v1", "confirmedTarget": True})

    def crop(self, source_id, start, end, destination):
        import soundfile as sf
        import numpy as np
        record = self.record(source_id)
        if not 0 <= start < end <= record["durationSeconds"] or end-start > 15+1e-6:
            raise ValueError("裁剪片段必须在素材范围内且不超过 15 秒")
        with sf.SoundFile(self.directory(source_id)/"audio.wav") as audio:
            audio.seek(round(start*audio.samplerate))
            samples = audio.read(round((end-start)*audio.samplerate), dtype="float32")
            if not len(samples) or not np.isfinite(samples).all():
                raise ValueError("片段为空或包含非法数值")
            sf.write(destination,samples,audio.samplerate,subtype="FLOAT")
        return str(destination)
