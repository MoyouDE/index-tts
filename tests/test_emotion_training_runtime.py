import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file
from torch import nn

from indextts.emotion import train as train_module
from indextts.emotion.model import MacBertEmotionModel
from indextts.emotion.schema import EMOTION_NAMES, EmotionExample
from indextts.emotion.training_state import (
    TrainingRunLock,
    build_training_fingerprint,
    epoch_batch_indices,
    find_resume_checkpoint,
)
from indextts.emotion.training_preflight import validate_resource_requirements


class _TinyConfig:
    hidden_size = 4

    def save_pretrained(self, target):
        (target / "config.json").write_text(
            json.dumps({"hidden_size": self.hidden_size}), encoding="utf-8"
        )


class _TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _TinyConfig()
        self.projection = nn.Linear(4, 4)

    def forward(self, input_ids, attention_mask, token_type_ids, return_dict):
        del attention_mask, token_type_ids, return_dict
        hidden = torch.nn.functional.one_hot(input_ids % 4, num_classes=4).float()
        return SimpleNamespace(last_hidden_state=self.projection(hidden))


class _TinyTokenizer:
    def __call__(self, previous, current, **kwargs):
        del kwargs
        rows = []
        for left, right in zip(previous, current):
            value = (sum(map(ord, left + right)) % 3) + 1
            rows.append([value, value + 1, value + 2])
        input_ids = torch.tensor(rows, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "token_type_ids": torch.zeros_like(input_ids),
        }

    def save_pretrained(self, target):
        (target / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")


def _example(index, work_id):
    label = [0.0] * len(EMOTION_NAMES)
    label[index % len(label)] = 0.67
    return EmotionExample(
        example_id=f"example-{work_id}-{index}",
        work_id=work_id,
        previous_text=f"previous-{index}",
        text=f"current-{index}",
        sentence_type="narration" if index % 2 else "dialogue",
        labels=tuple(label),
        label_mask=(1.0,) * len(label),
        intensity=0.67,
        license_id="PROPRIETARY-AUTHORIZED",
        source="test",
    )


def _patch_tiny_training(monkeypatch):
    tokenizer = _TinyTokenizer()
    monkeypatch.setattr(
        train_module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )
    monkeypatch.setattr(
        train_module.MacBertEmotionModel,
        "from_pretrained",
        lambda *args, **kwargs: MacBertEmotionModel(_TinyEncoder(), 4, dropout=0.1),
    )
    monkeypatch.setattr(
        train_module,
        "evaluate_model",
        lambda *args, **kwargs: {
            "macroF1": 0.5,
            "macroSpearman": 0.25,
            "neutralFalseActivationRate": 0.1,
        },
    )


def _run(output, train, dev, **overrides):
    options = {
        "base_model": "tiny",
        "epochs": 2,
        "batch_size": 2,
        "gradient_accumulation": 1,
        "learning_rate": 1e-3,
        "head_learning_rate": 2e-3,
        "max_length": 16,
        "seed": 1234,
        "device_name": "cpu",
        "checkpoint_steps": 1,
        "keep_checkpoints": 8,
        "progress": False,
    }
    options.update(overrides)
    return train_module.train_supervised(train, dev, output, **options)


def test_epoch_batch_indices_are_stable_and_epoch_specific():
    first = epoch_batch_indices(9, 3, 100, 1)
    assert first == epoch_batch_indices(9, 3, 100, 1)
    assert first != epoch_batch_indices(9, 3, 100, 2)
    assert sorted(item for batch in first for item in batch) == list(range(9))


def test_training_fingerprint_changes_with_route_or_examples():
    train = [_example(0, "train")]
    dev = [_example(1, "dev")]
    first = build_training_fingerprint(train, dev, {"epochs": 2}, {"train": "a"})
    assert first == build_training_fingerprint(train, dev, {"epochs": 2}, {"train": "a"})
    assert first != build_training_fingerprint(train, dev, {"epochs": 3}, {"train": "a"})
    assert first != build_training_fingerprint(train, dev, {"epochs": 2}, {"train": "b"})


def test_interrupted_resume_matches_uninterrupted_training(tmp_path, monkeypatch, capsys):
    _patch_tiny_training(monkeypatch)
    train = [_example(index, "train") for index in range(6)]
    dev = [_example(index, "dev") for index in range(2)]

    continuous = tmp_path / "continuous"
    continuous_report = _run(continuous, train, dev, resume="never", progress=True)
    progress_output = capsys.readouterr()
    assert "macroF1" in progress_output.out + progress_output.err
    assert "step" in progress_output.out + progress_output.err

    interrupted = tmp_path / "interrupted"
    original_save = train_module.save_training_checkpoint
    did_interrupt = False

    def stop_after_first_checkpoint(*args, **kwargs):
        nonlocal did_interrupt
        path = original_save(*args, **kwargs)
        if kwargs["global_step"] == 1 and not did_interrupt:
            did_interrupt = True
            raise train_module.TrainingInterrupted("test interruption")
        return path

    monkeypatch.setattr(train_module, "save_training_checkpoint", stop_after_first_checkpoint)
    with pytest.raises(train_module.TrainingInterrupted):
        _run(interrupted, train, dev, resume="auto")
    monkeypatch.setattr(train_module, "save_training_checkpoint", original_save)
    resumed_report = _run(interrupted, train, dev, resume="auto")

    assert resumed_report["history"] == continuous_report["history"]
    assert resumed_report["globalStep"] == continuous_report["globalStep"]
    continuous_weights = load_file(continuous / "best" / "model.safetensors")
    resumed_weights = load_file(interrupted / "best" / "model.safetensors")
    assert continuous_weights.keys() == resumed_weights.keys()
    for name in continuous_weights:
        assert torch.equal(continuous_weights[name], resumed_weights[name]), name

    completed = _run(interrupted, train, dev, resume="auto")
    assert completed["alreadyCompleted"] is True


def test_corrupt_latest_checkpoint_is_ignored_and_foreign_checkpoint_is_rejected(
    tmp_path, monkeypatch
):
    _patch_tiny_training(monkeypatch)
    train = [_example(index, "train") for index in range(4)]
    dev = [_example(index, "dev") for index in range(2)]
    output = tmp_path / "run"
    report = _run(output, train, dev, resume="never")
    fingerprint = report["inputFingerprint"]
    checkpoints = sorted((output / "checkpoints").glob("checkpoint-step-*"))
    assert len(checkpoints) >= 2
    (checkpoints[-1] / "optimizer.pt").write_bytes(b"corrupt")
    found = find_resume_checkpoint(output, fingerprint)
    assert found is not None
    assert found[0] == checkpoints[-2]
    with pytest.raises(ValueError, match="不匹配"):
        find_resume_checkpoint(output, "0" * 64)


def test_training_run_lock_rejects_concurrent_owner(tmp_path):
    with TrainingRunLock(tmp_path):
        with pytest.raises(RuntimeError, match="已有训练进程"):
            with TrainingRunLock(tmp_path):
                pass


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (
            {
                "cuda_available": False,
                "free_vram_bytes": 0,
                "free_disk_bytes": 20 * 1024**3,
                "require_cuda": True,
                "minimum_free_vram_bytes": 0,
                "minimum_free_disk_bytes": 10 * 1024**3,
            },
            "无法使用 CUDA",
        ),
        (
            {
                "cuda_available": True,
                "free_vram_bytes": 8 * 1024**3,
                "free_disk_bytes": 20 * 1024**3,
                "require_cuda": True,
                "minimum_free_vram_bytes": 9 * 1024**3,
                "minimum_free_disk_bytes": 10 * 1024**3,
            },
            "显存不足",
        ),
        (
            {
                "cuda_available": True,
                "free_vram_bytes": 10 * 1024**3,
                "free_disk_bytes": 9 * 1024**3,
                "require_cuda": True,
                "minimum_free_vram_bytes": 9 * 1024**3,
                "minimum_free_disk_bytes": 10 * 1024**3,
            },
            "空间不足",
        ),
    ],
)
def test_resource_preflight_rejects_unstable_training_environment(values, message):
    with pytest.raises(RuntimeError, match=message):
        validate_resource_requirements(**values)


def test_training_bat_is_gbk_crlf_and_has_fixed_quality_route():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    for filename in ("train-emotion.bat", "_train_emotion_inner.bat"):
        raw = (root / filename).read_bytes()
        text = raw.decode("gbk")
        assert b"\n" not in raw.replace(b"\r\n", b"")
        assert "\\outputs\\" not in text

    inner = (root / "_train_emotion_inner.bat").read_bytes().decode("gbk")
    for expected in (
        'set "NOVEL_TRAIN=outputs/emotion-data/training-ready-v3-novel-tgt-context512/train.jsonl"',
        'set "OUTPUT_DIR=outputs/emotion-data/macbert-training-v3-tgt-context512"',
        'set "EPOCHS=8"',
        'set "BATCH_SIZE=6"',
        'set "GRADIENT_ACCUMULATION=4"',
        'set "MAX_LENGTH=512"',
        'set "LEARNING_RATE=2e-5"',
        'set "HEAD_LEARNING_RATE=1e-4"',
        'set "INTENSITY_LOSS_WEIGHT=0.7"',
        'set "NEUTRAL_LOSS_WEIGHT=3.0"',
        'set "MIN_FREE_VRAM_GIB=9"',
        "--resume auto",
        "--checkpoint-steps %CHECKPOINT_STEPS%",
        "--require-complete-context",
        "--minimum-active-recall 0.8",
        '--threshold-output "%THRESHOLD_VALUE%"',
        "--neutral-threshold %NEUTRAL_THRESHOLD%",
        "--progress",
    ):
        assert expected in inner
    assert "BRIGHTER" not in inner
    assert "--release" not in inner
