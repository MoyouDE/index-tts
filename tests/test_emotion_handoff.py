import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "emotion-data" / "publish_emotion_handoff.py"
SPEC = importlib.util.spec_from_file_location("publish_emotion_handoff", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _source(tmp_path, *, status="approved"):
    source = tmp_path / "source"
    source.mkdir()
    files = {}
    for name in ("emotion.onnx", "tokenizer.json"):
        payload = name.encode("ascii")
        (source / name).write_bytes(payload)
        files[name] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    manifest = {
        "version": "user-selected-test",
        "releaseStatus": status,
        "releaseBasis": "user-selection",
        "releaseApproval": None,
        "sourceCheckpointSha256": "a" * 64,
        "neutralThreshold": 0.365,
        "qualitySummary": {
            "testSampleCount": 8255,
            "testMacroF1": 0.648,
            "testMacroSpearman": 0.585,
            "testIntensityMae": 0.098,
            "automaticChecksPassed": False,
            "failedChecks": ["addedNarrationMacroF1"],
        },
        "files": files,
    }
    (source / "emotion_model.json").write_text(json.dumps(manifest), encoding="utf-8")
    return source


def test_user_selected_handoff_preserves_failed_check_without_blocking(tmp_path):
    source = _source(tmp_path)
    target, backup = tmp_path / "emotion", tmp_path / "previous"
    result = MODULE.publish(source, target, backup)
    assert result["version"] == "user-selected-test"
    assert not backup.exists()
    assert "addedNarrationMacroF1" in (target / "README.md").read_text(encoding="utf-8")
    assert MODULE.load_manifest(target)["qualitySummary"]["automaticChecksPassed"] is False


def test_handoff_still_rejects_unselected_or_corrupt_package(tmp_path):
    source = _source(tmp_path, status="candidate-unvalidated")
    with pytest.raises(ValueError, match="approved"):
        MODULE.load_manifest(source)
    manifest_path = source / "emotion_model.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["releaseStatus"] = "approved"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (source / "emotion.onnx").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Source hash mismatch|Missing or truncated"):
        MODULE.load_manifest(source)
