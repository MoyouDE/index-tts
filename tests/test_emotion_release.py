import json

import pytest

from indextts.emotion.release import build_release_approval, validate_release_approval


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_release_gate_requires_data_quality_and_blind_test(tmp_path):
    training = _write(
        tmp_path / "training.json",
        {"releaseTraining": True, "releaseData": {"releaseEligible": True}},
    )
    model = _write(
        tmp_path / "model.json",
        {"macroF1": 0.8, "macroSpearman": 0.7, "sampleCount": 100},
    )
    qwen = _write(
        tmp_path / "qwen.json",
        {"macroF1": 0.75, "macroSpearman": 0.65, "sampleCount": 100, "warningCount": 2},
    )
    blind = _write(
        tmp_path / "blind.json",
        {"candidateWins": 50, "qwenWins": 40, "ties": 10},
    )
    audit = _write(
        tmp_path / "audit.json",
        {"commercialDataAuditPassed": True, "approvedBy": "data-owner"},
    )
    approval = build_release_approval(
        training_report_path=training,
        model_metrics_path=model,
        qwen_metrics_path=qwen,
        blind_test_path=blind,
        data_audit_path=audit,
    )
    assert approval["qualityGatePassed"] is True
    assert approval["humanBlindTestPassed"] is True
    assert approval["noQwenPseudoLabels"] is True
    assert approval["candidateNoWorseRate"] == pytest.approx(0.55)


def test_release_approval_rejects_handwritten_incomplete_evidence():
    with pytest.raises(ValueError, match="noQwenPseudoLabels"):
        validate_release_approval(
            {
                "schemaVersion": 1,
                "qualityGatePassed": True,
                "humanBlindTestPassed": True,
                "commercialDataAuditPassed": True,
            }
        )


def test_release_gate_rejects_model_below_qwen(tmp_path):
    training = _write(
        tmp_path / "training.json",
        {"releaseTraining": True, "releaseData": {"releaseEligible": True}},
    )
    model = _write(
        tmp_path / "model.json",
        {"macroF1": 0.6, "macroSpearman": 0.7, "sampleCount": 100},
    )
    qwen = _write(
        tmp_path / "qwen.json",
        {"macroF1": 0.7, "macroSpearman": 0.6, "sampleCount": 100, "warningCount": 0},
    )
    blind = _write(
        tmp_path / "blind.json",
        {"candidateWins": 100, "qwenWins": 0, "ties": 0},
    )
    audit = _write(
        tmp_path / "audit.json",
        {"commercialDataAuditPassed": True, "approvedBy": "data-owner"},
    )
    with pytest.raises(ValueError, match="Qwen 基线"):
        build_release_approval(
            training_report_path=training,
            model_metrics_path=model,
            qwen_metrics_path=qwen,
            blind_test_path=blind,
            data_audit_path=audit,
        )
