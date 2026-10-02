"""Persistent local-only voice library. Model files and audio never enter Git."""
import json
import os
from pathlib import Path
import shutil
import threading
import uuid

from .runtime.profiles import FP32, PROFILES
from .voicepack.archive import load_voicepack
from .voicepack.provenance import sha256_file
from .voicepack.schema import validate_identity


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class VoiceWorkspace:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.voices = self.root / "voices"
        self.voices.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()

    def directory(self, voice_id):
        validate_identity(voice_id, "voice", "unknown")
        path = (self.voices / voice_id).resolve()
        if not path.is_relative_to(self.voices) or path == self.voices:
            raise ValueError("音色路径越界")
        return path

    def record(self, voice_id):
        return json.loads((self.directory(voice_id) / "voice.json").read_text(encoding="utf-8"))

    def items(self):
        result = []
        with self.lock:
            for path in sorted(self.voices.glob("*/voice.json")):
                try:
                    record = self.record(path.parent.name)
                    record["variants"] = {}
                    for profile in PROFILES:
                        pack_path = path.parent / (profile + ".ivp")
                        if pack_path.exists():
                            try:
                                pack = load_voicepack(pack_path)
                                record["variants"][profile] = {"ready": True, "manifest": pack.manifest}
                            except Exception as exc:
                                record["variants"][profile] = {"ready": False, "error": str(exc)}
                        else:
                            record["variants"][profile] = {"ready": False, "status": "尚未生成"}
                    result.append(record)
                except Exception as exc:
                    result.append({"voiceId": path.parent.name, "displayName": path.parent.name, "error": str(exc)})
        return result

    def pack_path(self, voice_id, profile):
        if profile not in PROFILES:
            raise ValueError("未知精度")
        return self.directory(voice_id) / (profile + ".ivp")

    def install(self, pack_path, *, reference=None, expected_instance=None, reference_selection=None, references=None):
        pack = load_voicepack(pack_path)
        profile = pack.manifest.get("provenance", {}).get("profile")
        if profile is None:
            # Legacy packs are accepted only when their tensors really are FP32.
            if any(spec["dtype"] != "float32" for spec in pack.manifest["tensors"].values()):
                raise ValueError("旧包没有精度 provenance，且不是全 FP32；请从参考音频重新制包")
            profile = FP32
        if reference_selection is not None and reference_selection.get("method") == "speaker-mean-v1":
            if pack.manifest.get("provenance", {}).get("referenceSelection") != reference_selection:
                raise ValueError("包内来源与参考选择不一致")
        voice_id = pack.voice_id
        with self.lock:
            directory = self.directory(voice_id)
            exists = (directory / "voice.json").exists()
            if exists:
                record = self.record(voice_id)
                if expected_instance != record["instance"]:
                    raise ValueError("音色 ID 已存在；选择库中音色重新生成，导入不会覆盖已有条目")
                provenance = pack.manifest.get("provenance", {})
                if provenance.get("referenceSha256") != record.get("referenceSha256"):
                    raise ValueError("参考音频发生变化，请使用新的音色 ID")
                if reference_selection != record.get("referenceSelection"):
                    raise ValueError("参考选择发生变化，请使用新的音色 ID")
            elif expected_instance is not None:
                raise ValueError("音色已删除，任务结果不会重新创建该条目")
            else:
                record = {"voiceId": voice_id, "displayName": pack.manifest["displayName"],
                          "gender": pack.manifest["gender"], "notes": "", "instance": uuid.uuid4().hex,
                          "reference": None, "referenceSha256": None}
            staging = self.voices / (".install-" + uuid.uuid4().hex)
            staging.mkdir()
            try:
                staged_pack = staging / (profile + ".ivp")
                shutil.copyfile(pack_path, staged_pack)
                if not exists:
                    if reference is not None:
                        ref_name = "reference" + Path(reference).suffix.lower()
                        shutil.copyfile(reference, staging / ref_name)
                        record.update(reference=ref_name, referenceSha256=sha256_file(staging / ref_name))
                        if pack.manifest.get("provenance", {}).get("referenceSha256") != record["referenceSha256"]:
                            raise ValueError("参考音频哈希与制包结果不匹配")
                    if reference_selection is not None:
                        if reference_selection.get("method") not in {"primary-only-v1", "speaker-mean-v1"} or reference_selection.get("primarySha256") != record["referenceSha256"]:
                            raise ValueError("参考选择与主参考音频不匹配")
                        if reference_selection["method"] == "speaker-mean-v1":
                            from .voicepack.selection import validate_selection
                            validate_selection(reference_selection, references, staging / ref_name)
                            if pack.manifest.get("provenance", {}).get("referenceSelection") != reference_selection:
                                raise ValueError("包内来源与参考选择不一致")
                            (staging / "references").mkdir()
                            for sid, path in references.items():
                                shutil.copyfile(path, staging / "references" / (sid + ".wav"))
                        record["referenceSelection"] = reference_selection
                    atomic_json(staging / "voice.json", record)
                    # New entries are made visible together, never as half-created records.
                    os.rename(staging, directory)
                else:
                    os.replace(staged_pack, self.pack_path(voice_id, profile))
                    preview = directory / (profile + "-preview")
                    if preview.exists():
                        self._remove(preview)
            finally:
                if staging.exists():
                    self._remove(staging)
        return str(self.pack_path(voice_id, profile))

    def _remove(self, path):
        path = Path(path).resolve()
        if not path.is_relative_to(self.root) or path == self.root:
            raise ValueError("拒绝删除工作区外的路径")
        shutil.rmtree(path)

    def delete(self, voice_id, confirmation):
        with self.lock:
            record = self.record(voice_id)
            if confirmation != f"{voice_id}:{record['instance']}":
                raise ValueError("请先确认当前选中的音色及全部本地产物")
            self._remove(self.directory(voice_id))

    def notes(self, voice_id, value):
        with self.lock:
            record = self.record(voice_id)
            record["notes"] = str(value)
            atomic_json(self.directory(voice_id) / "voice.json", record)

    def snapshot(self, voice_id, profile, destination):
        with self.lock:
            record = self.record(voice_id)
            path = self.pack_path(voice_id, profile)
            if not path.exists():
                raise ValueError("尚未生成此精度，请到制包页选择库中音色并生成")
            load_voicepack(path)
            shutil.copyfile(path, destination)
            return record["instance"], sha256_file(destination)

    def save_preview(self, voice_id, profile, instance, pack_hash, audio, report):
        with self.lock:
            if self.record(voice_id)["instance"] != instance or sha256_file(self.pack_path(voice_id, profile)) != pack_hash:
                raise ValueError("试听期间音色已变更，结果不会覆盖新版音色的试听")
            directory = self.directory(voice_id) / (profile + "-preview")
            directory.mkdir(exist_ok=True)
            # Unique files + atomic pointer keep the previous result intact on failure.
            key = uuid.uuid4().hex
            wav = directory / (key + ".wav")
            details = directory / (key + ".json")
            try:
                shutil.copyfile(audio, wav)
                atomic_json(details, report)
                atomic_json(directory / "latest.json", {"wav": wav.name, "report": details.name})
            except Exception:
                wav.unlink(missing_ok=True)
                details.unlink(missing_ok=True)
                raise
            for path in directory.iterdir():
                if path.name not in {wav.name, details.name, "latest.json"}:
                    path.unlink()
            return str(wav), str(details)

    def latest(self, voice_id, profile):
        directory = self.directory(voice_id) / (profile + "-preview")
        if not (directory / "latest.json").exists():
            return None, None, {}
        data = json.loads((directory / "latest.json").read_text(encoding="utf-8"))
        paths = [(directory / data[key]).resolve() for key in ("wav", "report")]
        if any(path.parent != directory.resolve() for path in paths):
            raise ValueError("试听记录路径无效")
        return str(paths[0]), str(paths[1]), json.loads(paths[1].read_text(encoding="utf-8"))

    def settings(self):
        path = self.root / "settings.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def save_settings(self, updates):
        with self.lock:
            atomic_json(self.root / "settings.json", {**self.settings(), **updates})
