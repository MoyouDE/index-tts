"""Private, recoverable producer drafts and immutable build/audition snapshots.

The voice library is deliberately not a persistence dependency of this service.
"""
import copy
import json
import math
import os
from pathlib import Path
import re
import shutil
import threading
import time
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from .material_service import MaterialService, digest_json
from .material_web import prepare_selection, read_draft, table_rows, resolve_primary
from .runtime.profiles import FP32, BF16, PROFILES, synthesis_settings
from .voice_workspace import atomic_json
from .voicepack.archive import load_voicepack, combine_voicepacks, repackage_voicepack
from .voicepack.provenance import sha256_file


def now():
    return datetime.now(timezone.utc).isoformat()


def reference_rows(editor):
    """Retain staged ranges in the draft; pass only enabled ranges to the builder."""
    editor = editor or {}
    disabled = set(editor.get("disabled", []))
    return [[*r[:3], r[3] and r[0] not in disabled] for r in editor.get("rows", [])]


def reference_key(draft, *, legacy_profile=None):
    editor = draft.get("editor") or {}
    rows = [r for r in reference_rows(editor) if r[3]]
    main = resolve_primary(rows, editor.get("primary", "auto")) if rows else None
    return digest_json([draft.get("sourceId"), rows, main, legacy_profile or "dual-fp32-bf16-v1"])


def package_key(draft):
    return digest_json([reference_key(draft), draft["name"], draft["gender"]])


def audition_emotion(value):
    if value == "base":
        return "base"
    if not isinstance(value, list) or len(value) != 8 or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1.2
        for v in value
    ):
        raise ValueError("情感向量必须包含 8 个 [0, 1.2] 范围内的数值")
    return list(value) if any(value) else "base"


def repackage(source, destination, name, gender):
    """Preserve raw tensors, licenses and provenance byte for byte."""
    return repackage_voicepack(source, destination, name, gender)


class ProducerDrafts:
    def __init__(self, root, workbench, source_model_dir, runtime_root, cpu_threads=4):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.trash = self.root / ".trash"
        self.trash.mkdir(exist_ok=True)
        self.workbench = workbench
        self.source_model_dir = str(Path(source_model_dir).resolve())
        self.runtime_root = Path(runtime_root).resolve()
        self.cpu_threads = cpu_threads
        self.lock = threading.RLock()
        self._materials = {}
        self._tasks = {}
        self._runtime_verified = {}
        self.purge_expired()

    def directory(self, wid, *, recycled=False):
        if not isinstance(wid, str) or not re.fullmatch(r"[a-f0-9]{32}", wid):
            raise ValueError("工作区 ID 无效")
        root = self.trash if recycled else self.root
        path = (root / wid).resolve()
        if path.parent != root or path.is_symlink():
            raise ValueError("工作区路径越界")
        return path

    def read(self, wid):
        with self.lock:
            draft = json.loads((self.directory(wid) / "draft.json").read_text(encoding="utf-8"))
            draft.setdefault("emotion", "base")
            return draft

    def _write(self, draft):
        draft["savedAt"] = now()
        atomic_json(self.directory(draft["workspaceId"]) / "draft.json", draft)

    def items(self, *, recycled=False):
        with self.lock:
            result = []
            for p in (self.trash if recycled else self.root).glob("*/draft.json"):
                if re.fullmatch(r"[a-f0-9]{32}", p.parent.name):
                    result.append(json.loads(p.read_text(encoding="utf-8")))
            return sorted(result, key=lambda d: d["savedAt"], reverse=True)

    def activate(self, wid):
        with self.lock:
            draft = self.read(wid)
            atomic_json(self.root / "active.json", {"workspaceId": wid})
            return draft

    def last(self):
        items = self.items()
        return self.activate(items[0]["workspaceId"]) if items else self.new()

    def current(self):
        with self.lock:
            try:
                wid = json.loads((self.root / "active.json").read_text(encoding="utf-8"))["workspaceId"]
                return self.read(wid)
            except (OSError, ValueError, KeyError):
                return self.last()

    def new(self):
        with self.lock:
            wid = uuid.uuid4().hex
            directory = self.directory(wid)
            directory.mkdir()
            for name in ("packages", "previews"):
                (directory / name).mkdir()
            draft = dict(workspaceId=wid, voiceId="draft-" + wid, createdAt=now(), savedAt=now(),
                         revision=0, referenceRevision=0, sourceId=None, editor=None,
                         filters=dict(silence=.5, minimum=1, volume=-50, enabled=False),
                         name="", gender="unknown", profile=FP32, device="auto",
                         text="清晨的阳光照进了安静的书房。", emotion="base", packages=[], currentPackage=None,
                         previews=[], clients={}, lastClient=None)
            self._write(draft)
            return self.activate(wid)

    def materials(self, wid):
        with self.lock:
            self.read(wid)  # Never recreate a cleared workspace from a late request.
            if wid not in self._materials:
                self._materials[wid] = MaterialService(self.directory(wid))
            return self._materials[wid]

    def import_media(self, wid, path, progress=None):
        with self.lock:
            before = self.read(wid)
        f = before["filters"]
        record = self.materials(wid).import_media(path, progress, min_silence_ms=f["silence"]*1000,
                                                min_volume_db=f["volume"] if f["enabled"] else None)
        return self.attach(wid, record, expected_revision=before["revision"])

    def import_legacy(self, wid, source_id, *, editor=None):
        with self.lock:
            self.read(wid)
            old = self.workbench.materials
            record = old.record(source_id)
            target = self.materials(wid).directory(source_id)
            if not target.exists():
                staging = target.with_name(".copy-" + uuid.uuid4().hex)
                try:
                    shutil.copytree(old.directory(source_id), staging)
                    os.rename(staging, target)
                finally:
                    if staging.exists():
                        shutil.rmtree(staging)
            return self.attach(wid, record, editor=editor)

    def attach(self, wid, record, *, editor=None, expected_revision=None):
        with self.lock:
            d = self.read(wid)
            if expected_revision is not None and d["revision"] != expected_revision:
                raise ValueError("导入期间工作区已修改，素材已保留，请重新选择")
            d["sourceId"] = record["sourceId"]
            d["editor"] = editor or dict(sourceId=record["sourceId"], rows=table_rows(record), primary=record["primary"] or "auto")
            d["revision"] += 1
            d["referenceRevision"] += 1
            d["savedAt"] = now()
            self._write(d)
            return d

    def save(self, wid, payload):
        """Optimistic cross-client guard, with ordered requests from one browser."""
        if isinstance(payload, str):
            payload = json.loads(payload)
        if payload.get("workspaceId") != wid:
            raise ValueError("编辑内容不属于当前工作区")
        client = payload.get("clientId", "api")
        seq = payload.get("sequence", 0)
        with self.lock:
            d = self.read(wid)
            previous = d["clients"].get(client, -1)
            if seq <= previous:
                return d
            if payload.get("revision") != d["revision"] and d["lastClient"] != client:
                raise ValueError("工作区已在其他页面修改，请重新加载")
            updated = copy.deepcopy(d)
            for key in ("name", "gender", "profile", "device", "text", "emotion", "filters", "editor"):
                if key in payload:
                    updated[key] = copy.deepcopy(payload[key])
            if updated["profile"] not in PROFILES or updated["gender"] not in {"unknown", "male", "female"}:
                raise ValueError("音色设置无效")
            if not isinstance(updated["name"], str) or len(updated["name"].strip()) > 128:
                raise ValueError("名称最多 128 字")
            updated["name"] = updated["name"].strip()
            if not isinstance(updated["text"], str) or len(updated["text"]) > 20000:
                raise ValueError("试听文本过长")
            updated["emotion"] = audition_emotion(updated["emotion"])
            if updated["device"] not in {"auto", "cpu", "cuda:0", "cuda:1"}:
                raise ValueError("计算设备无效")
            f = updated["filters"]
            for key, lo, hi in (("silence", .1, 3), ("minimum", .1, 15), ("volume", -80, 0)):
                if isinstance(f.get(key), bool) or not isinstance(f.get(key), (int, float)) or not math.isfinite(f[key]) or not lo <= f[key] <= hi:
                    raise ValueError("过滤参数无效")
            if not isinstance(f.get("enabled"), bool):
                raise ValueError("过滤开关无效")
            if updated["sourceId"]:
                rows, primary = read_draft(updated["sourceId"], json.dumps(updated["editor"]))
                # Validate a draft without replacing the material's immutable source record.
                duration = self.materials(wid).record(updated["sourceId"])["durationSeconds"]
                ids = set()
                chosen = sorted([r for r in rows if r[3]], key=lambda r:r[1])
                for r in rows:
                    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", r[0]) or r[0] in ids or not 0 <= r[1] < r[2] <= duration:
                        raise ValueError("片段边界或 ID 无效")
                    ids.add(r[0])
                if any(r[2]-r[1] > 15.000001 for r in chosen) or any(a[2] > b[1]+1e-6 for a,b in zip(chosen, chosen[1:])):
                    raise ValueError("选中片段过长或重叠")
                disabled = updated["editor"].get("disabled", [])
                retained_ids = {r[0] for r in chosen}
                if (not isinstance(disabled, list) or any(not isinstance(s, str) or s not in retained_ids for s in disabled)
                        or len(disabled) != len(set(disabled))):
                    raise ValueError("片段启用状态无效")
                if updated["editor"].get("workspaceId", wid) != wid:
                    raise ValueError("片段来自其他工作区")
                for key, t in updated["editor"].get("previewStarts", {}).items():
                    if not isinstance(t, (int,float)) or isinstance(t,bool) or not math.isfinite(t) or not 0 <= t <= duration:
                        raise ValueError("试听起点无效")
            if reference_key(updated) != reference_key(d):
                updated["referenceRevision"] += 1
            changed = any(updated[k] != d[k] for k in ("name", "gender", "profile", "device", "text", "emotion", "filters", "editor"))
            if changed:
                updated["revision"] += 1
                updated["savedAt"] = now()
            updated["clients"][client] = seq
            updated["lastClient"] = client
            self._write(updated)
            return updated

    def cancel(self, wid, *, wait=False):
        with self.lock:
            task = self._tasks.get(wid)
            if task:
                task[0].set()
        if task and wait:
            task[1].wait()
        return "正在停止，当前计算结束后释放" if task and not task[1].is_set() else "已停止；成功结果仍保留"

    @contextmanager
    def task(self, wid):
        with self.lock:
            snapshot = self.read(wid)
            if wid in self._tasks:
                raise ValueError("当前工作区已有任务，请等待或取消")
            cancelled, done = threading.Event(), threading.Event()
            self._tasks[wid] = (cancelled, done)
        try:
            yield snapshot, cancelled
        finally:
            with self.lock:
                self._tasks.pop(wid, None)
                done.set()

    def _guard(self, snapshot, cancelled):
        d = self.read(snapshot["workspaceId"])
        if cancelled.is_set():
            raise RuntimeError("任务已取消")
        return d

    def ensure_package(self, snapshot, cancelled, progress=None):
        wid = snapshot["workspaceId"]
        if not snapshot["name"]:
            raise ValueError("请填写音色名称")
        if not snapshot["sourceId"]:
            raise ValueError("请先上传素材并选择参考片段")
        enabled_rows = reference_rows(snapshot["editor"])
        if not any(r[3] for r in enabled_rows):
            raise ValueError("请至少启用一个已选片段；停用的片段仍保留在列表中")
        key, ref = package_key(snapshot), reference_key(snapshot)
        with self.lock:
            d = self._guard(snapshot, cancelled)
            exact = next((p for p in reversed(d["packages"]) if p["key"] == key), None)
            same_ref = next((p for p in reversed(d["packages"]) if p["referenceKey"] == ref), None)
        if exact:
            load_voicepack(self.directory(wid) / exact["file"])
            with self.lock:
                d = self._guard(snapshot, cancelled)
                if d["revision"] == snapshot["revision"]:
                    d["currentPackage"] = exact
                    self._write(d)
            return exact
        version = uuid.uuid4().hex
        path = self.directory(wid) / "packages" / (version + ".ivp")
        jobs = []
        assets = self.directory(wid) / "packages" / (version + "-data")
        try:
            if same_ref:
                repackage(self.directory(wid)/same_ref["file"], path, snapshot["name"], snapshot["gender"])
            else:
                materials = self.materials(wid)
                record = materials.record(snapshot["sourceId"])
                selection, _ = prepare_selection(materials, snapshot["sourceId"], enabled_rows,
                                                snapshot["editor"]["primary"], record["revision"])
                variants = {}
                with self.lock:
                    current=self._guard(snapshot,cancelled)
                    caches={precision:next((p for p in reversed(current['packages'])
                        if p['referenceKey']==reference_key(snapshot,legacy_profile=precision)),None) for precision in PROFILES}
                reference_snapshot=None
                leader=None
                for precision,cached in caches.items():
                    if cached and cached.get('referenceData'):
                        pack=load_voicepack(self.directory(wid)/cached['file'],profile=precision)
                        refs=self.directory(wid)/cached['referenceData']
                        provenance=pack.manifest['provenance']
                        if sha256_file(refs/'primary.wav')!=provenance['referenceSha256']:
                            raise ValueError('已缓存主参考完整性校验失败')
                        for seg in provenance.get('referenceSelection',{}).get('segments',[]):
                            if sha256_file(refs/(seg['id']+'.wav'))!=seg['sha256']:
                                raise ValueError('已缓存参考片段完整性校验失败')
                        reference_snapshot=refs;leader=provenance;break
                if leader:
                    for precision,cached in list(caches.items()):
                        if cached:
                            provenance=load_voicepack(self.directory(wid)/cached['file'],profile=precision).manifest['provenance']
                            if provenance['referenceSha256']!=leader['referenceSha256'] or provenance.get('referenceSelection',{}).get('segments')!=leader.get('referenceSelection',{}).get('segments'):
                                caches[precision]=None
                elif not all(caches.values()):
                    # An old pack without its raw snapshot cannot seed the other profile.
                    caches={precision:None for precision in PROFILES}
                with tempfile.TemporaryDirectory(dir=self.directory(wid)/"packages") as staging:
                    for precision in PROFILES:
                        with self.lock:
                            self._guard(snapshot,cancelled)
                        cached=caches[precision]
                        if cached:
                            source=self.directory(wid)/cached['file']
                            load_voicepack(source,profile=precision)
                            variants[precision]=repackage(source,Path(staging)/(precision+'.ivp'),snapshot['name'],snapshot['gender'])
                            refs=self.directory(wid)/cached['referenceData'] if cached.get('referenceData') else None
                        else:
                            if progress:progress(0,desc='生成 '+('FP32' if precision==FP32 else 'BF16')+' 音色数据')
                            job,report=self.workbench.build_selection(copy.deepcopy(selection),snapshot['voiceId'],snapshot['name'],
                                snapshot['gender'],precision,snapshot['device'],self.source_model_dir,
                                uuid.uuid4().hex,progress,materials=materials,temporary=True,cancelled=cancelled,
                                reference_snapshot=reference_snapshot)
                            jobs.append(job); variants[precision]=job
                            refs=Path(job).parent/'references'
                            if refs.exists():reference_snapshot=refs
                        if refs and refs.exists() and not assets.exists():shutil.copytree(refs,assets)
                    with self.lock:self._guard(snapshot,cancelled)
                    combine_voicepacks(path,variants[FP32],variants[BF16])
            pack = load_voicepack(path)
            entry = dict(version=version, file="packages/"+path.name, key=key, referenceKey=ref,
                         referenceRevision=snapshot["referenceRevision"], revision=snapshot["revision"],
                         profile=snapshot["profile"], profiles=list(PROFILES), sha256=sha256_file(path), createdAt=now(),
                         name=snapshot["name"], gender=snapshot["gender"],
                         editor=copy.deepcopy(snapshot["editor"]), sourceId=snapshot["sourceId"],
                         referenceData=(same_ref or {}).get("referenceData") if same_ref else
                            ("packages/"+assets.name if assets.exists() else None))
            with self.lock:
                d = self._guard(snapshot, cancelled)
                d["packages"].append(entry)
                # A task may add historical results, but never write newer editor fields.
                if d["revision"] == snapshot["revision"]:
                    d["currentPackage"] = entry
                self._write(d)
            return entry
        except Exception:
            path.unlink(missing_ok=True)
            path.with_suffix(".part").unlink(missing_ok=True)
            if assets.exists():
                shutil.rmtree(assets)
            raise
        finally:
            for job in jobs:
                parent = Path(job).resolve().parent
                if parent.is_relative_to(self.workbench.output_dir) and parent != self.workbench.output_dir:
                    shutil.rmtree(parent)

    def runtime(self, profile):
        from .runtime.model_export import verify_runtime_model
        root = self.runtime_root/profile
        if not (root/"runtime_model.json").is_file():
            raise ValueError("试听模型尚未准备，请运行独立模型准备命令")
        manifest = root/"runtime_model.json"
        token = (str(root), manifest.stat().st_mtime_ns)
        if token not in self._runtime_verified:
            verify_runtime_model(root)
            self._runtime_verified[token] = True
        return root

    def run(self, wid, kind, progress=None):
        if kind not in {"build", "audition", "export"}:
            raise ValueError("未知工作区任务")
        with self.task(wid) as (snapshot, cancelled):
            entry = self.ensure_package(snapshot, cancelled, progress)
            if kind == "audition":
                if not snapshot["text"].strip():
                    raise ValueError("请输入试听文本")
                models = self.runtime(snapshot["profile"])
                service = self.workbench.audition_service
                device = "cuda:0" if snapshot["device"] == "auto" else snapshot["device"]
                settings = synthesis_settings(snapshot["profile"])
                with self.workbench.gpu.use("audition", cancelled):
                    audio, report = service.synthesize(self.directory(wid)/entry["file"], snapshot["voiceId"],
                        snapshot["profile"], str(models), device, snapshot["text"], snapshot["emotion"], 1, 17,
                        {k:v for k,v in settings.items() if k != "num_return_sequences"},
                        "all", self.cpu_threads, "native", cancelled, progress)
                    preview = dict(version=entry["version"], packageKey=entry["key"],
                        profile=snapshot['profile'],
                        referenceRevision=snapshot["referenceRevision"], text=snapshot["text"],
                        settings=settings, seed=17, emotion=copy.deepcopy(snapshot["emotion"]), speed=1, createdAt=now(),
                        file="previews/"+uuid.uuid4().hex+".wav")
                    target = self.directory(wid)/preview["file"]
                    try:
                        with self.lock:
                            d = self._guard(snapshot, cancelled)
                            shutil.copyfile(audio, target)
                            atomic_json(target.with_suffix(".json"), report)
                            obsolete = d["previews"][1:]
                            d["previews"] = [preview] + d["previews"][:1]
                            self._write(d)
                            for old in obsolete:
                                oldfile = self.directory(wid)/old["file"]
                                oldfile.unlink(missing_ok=True)
                                oldfile.with_suffix(".json").unlink(missing_ok=True)
                    except Exception:
                        target.unlink(missing_ok=True)
                        target.with_suffix(".json").unlink(missing_ok=True)
                        raise
                    finally:
                        Path(audio).unlink(missing_ok=True)
            with self.lock:
                d = self._guard(snapshot, cancelled)
                if kind == "export" and package_key(d) != entry["key"]:
                    raise ValueError("制作期间参考或音色信息已修改，请再次下载最新版本")
                return entry, d

    def export_path(self, wid, entry):
        """A readable download name; the immutable version remains separately stored."""
        with self.lock:
            d = self.read(wid)
            if package_key(d) != entry["key"]:
                raise ValueError("音色信息已修改，请重新准备下载")
            source = self.directory(wid)/entry["file"]
            load_voicepack(source)
            name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', d["name"]).strip(' .')[:80]
            directory = self.directory(wid)/"exports"/entry["version"]
            directory.mkdir(parents=True, exist_ok=True)
            destination = directory/("音色-"+(name or "未命名")+".ivp")
            part = destination.with_suffix(".part")
            try:
                shutil.copyfile(source, part)
                os.replace(part, destination)
            finally:
                part.unlink(missing_ok=True)
            return str(destination)

    def clear(self, wid):
        self.cancel(wid, wait=True)
        with self.lock:
            d = self.read(wid)
            d["clearedAt"] = time.time()
            self._write(d)
            material = self._materials.pop(wid, None)
            if material:
                material.close()
            os.rename(self.directory(wid), self.directory(wid, recycled=True))
            return self.new()

    def restore(self, wid):
        with self.lock:
            source, destination = self.directory(wid, recycled=True), self.directory(wid)
            if destination.exists():
                raise ValueError("工作区已恢复")
            d = json.loads((source/"draft.json").read_text(encoding="utf-8"))
            if time.time()-d["clearedAt"] > 7*86400:
                raise ValueError("回收工作区已超过 7 天")
            os.rename(source, destination)
            d.pop("clearedAt", None)
            d["savedAt"] = now()
            self._write(d)
            return self.activate(wid)

    def purge_expired(self):
        for d in self.items(recycled=True):
            if time.time()-d.get("clearedAt", time.time()) > 7*86400:
                shutil.rmtree(self.directory(d["workspaceId"], recycled=True))

    def close(self):
        for wid in list(self._tasks):
            self.cancel(wid, wait=True)
        for m in self._materials.values():
            m.close()
