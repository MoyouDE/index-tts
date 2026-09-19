import json

import pytest

from indextts.emotion.release import build_release_approval, validate_release_approval


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_release_gate_requires_training_metrics_and_experiment_audit(tmp_path):
    training = _write(
        tmp_path / "training.json",
        {
            "trainingObjective": {"loss_mode": "balanced-regression"},
            "finalEpoch": 8,
            "finalModelSha256": "a" * 64,
        },
    )
    dev = _write(
        tmp_path / "dev.json",
        {"macroF1": 0.64, "macroSpearman": 0.59, "intensityMae": 0.09, "sampleCount": 100},
    )
    test = _write(
        tmp_path / "test.json",
        {"macroF1": 0.65, "macroSpearman": 0.59, "intensityMae": 0.09, "sampleCount": 120},
    )
    audit = _write(
        tmp_path / "audit.json",
        {"passed": True, "recommendation": {"candidate": "A-loss"}},
    )
    approval = build_release_approval(
        training_report_path=training,
        dev_metrics_path=dev,
        test_metrics_path=test,
        experiment_audit_path=audit,
    )
    assert approval["qualityGatePassed"] is True
    assert approval["selectedCandidate"] == "A-loss"
    assert set(approval["inputSha256"]) == {
        "trainingReport", "devMetrics", "testMetrics", "experimentAudit"
    }


def test_release_approval_rejects_handwritten_incomplete_evidence():
    with pytest.raises(ValueError, match="证据集合"):
        validate_release_approval(
            {
                "schemaVersion": 2,
                "qualityGatePassed": True,
                "selectionPolicy": "experiment-comparison-and-user-selection",
                "selectedCandidate": "A-loss",
                "inputSha256": {},
            }
        )


def test_release_gate_rejects_failed_experiment_audit(tmp_path):
    training = _write(
        tmp_path / "training.json",
        {"experiment": {"loss_mode": "balanced-regression"}, "finalEpoch": 8, "finalModelSha256": "a" * 64},
    )
    metrics = {"macroF1": 0.6, "macroSpearman": 0.5, "intensityMae": 0.1, "sampleCount": 100}
    dev = _write(tmp_path / "dev.json", metrics)
    test = _write(tmp_path / "test.json", metrics)
    audit = _write(tmp_path / "audit.json", {"passed": False, "recommendation": {"candidate": "A-loss"}})
    with pytest.raises(ValueError, match="验收报告未通过"):
        build_release_approval(
            training_report_path=training,
            dev_metrics_path=dev,
            test_metrics_path=test,
            experiment_audit_path=audit,
        )
