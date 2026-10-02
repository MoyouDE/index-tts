import copy
import json

import numpy as np
import pytest
import soundfile as sf

from tools.reference_fusion_experiment import control_fingerprint, resume_report
from indextts.voicepack.provenance import sha256_file


@pytest.fixture
def snapshot(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "runtime_model.json").write_text('{"sourceModelFingerprint":"test"}', encoding="utf-8")
    (tmp_path / "encoding.json").write_text('{"sourceModelFingerprint":"test"}', encoding="utf-8")
    config = {"recipeSha256": "reference", "texts": ["one", "two"], "seeds": [17],
              "generationSettings": {"do_sample": False}, "readerModels": str(model)}
    destination = tmp_path / "baseline"
    destination.mkdir()
    audio = destination / "seed-17-text-1.wav"
    sf.write(audio, np.ones(2205, dtype=np.float32) * .1, 22050, subtype="FLOAT")
    sample = {"seed": 17, "textIndex": 1, "file": audio.name, "waveformSha256": sha256_file(audio),
              "result": {"seed": 17, "profile": "compatible-fp32", "emotionMode": "base",
                         "generationSettings": {"do_sample": False}}}
    report = {"case": "baseline", "controlSha256": control_fingerprint(tmp_path, config), "samples": [sample]}
    path = destination / "synthesis.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return tmp_path, config, report, path, audio


def test_accepts_intact_partial_snapshot_without_rewriting_audio(snapshot):
    root, config, report, _, audio = snapshot
    before = audio.read_bytes()
    assert resume_report(root, "baseline", config) == report
    assert audio.read_bytes() == before
    assert resume_report(root, "not-started", config) is None


def test_rejects_changed_text_or_reader_manifest(snapshot):
    root, config, _, _, _ = snapshot
    changed = copy.deepcopy(config)
    changed["texts"][0] = "changed"
    with pytest.raises(ValueError, match="控制指纹"):
        resume_report(root, "baseline", changed)
    (root / "model/runtime_model.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="控制指纹"):
        resume_report(root, "baseline", config)


def test_rejects_corrupted_waveform(snapshot):
    root, config, _, _, audio = snapshot
    audio.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="哈希"):
        resume_report(root, "baseline", config)


@pytest.mark.parametrize("change", ["duplicate", "path", "settings"])
def test_rejects_invalid_sample_records(snapshot, change):
    root, config, report, path, _ = snapshot
    if change == "duplicate":
        report["samples"].append(copy.deepcopy(report["samples"][0]))
    elif change == "path":
        report["samples"][0]["file"] = "../other.wav"
    else:
        report["samples"][0]["result"]["generationSettings"]["do_sample"] = True
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        resume_report(root, "baseline", config)


def test_timeout_stops_owned_interpreter_children(tmp_path):
    import sys
    import psutil
    from tools.reference_fusion_experiment import isolated_step
    pid_file = tmp_path / "child.pid"
    code = ("import subprocess,sys,time;from pathlib import Path;"
            "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
            f"Path({str(pid_file)!r}).write_text(str(child.pid));time.sleep(60)")
    assert isolated_step([sys.executable, "-c", code], tmp_path / "timeout.log", timeout=2) == "timeout"
    assert pid_file.exists()
    pid = int(pid_file.read_text())
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


@pytest.mark.parametrize("partial,stem,label", [(False, "blind", "V01"), (True, "blind-partial", "U01")])
def test_blind_key_has_stable_name_separate_from_montage(tmp_path, monkeypatch, partial, stem, label):
    from tools import reference_fusion_experiment as experiment
    config = {"cases": ["baseline"], "seeds": [17], "texts": ["test"]}
    monkeypatch.setattr(experiment, "checked_configuration", lambda output: config)
    (tmp_path / "baseline").mkdir()
    sf.write(tmp_path / "baseline/seed-17-text-1.wav", np.ones(2205) * .1, 22050)
    assert experiment.make_blind(tmp_path, partial) == ["identity"]
    assert json.loads((tmp_path / (stem + "-key.json")).read_text()) == {label: "baseline"}
    assert (tmp_path / stem / "identity-seed-17-text-1.wav").is_file()
    assert not (tmp_path / "identity-seed-17-text-1-key.json").exists()
